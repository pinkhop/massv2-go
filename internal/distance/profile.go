// MIT License
//
// Copyright (c) 2025 David L Kinney <david@pinkhop.com> <david@kinney.io>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// Package distance computes the z-normalized Euclidean distance profile that
// the public massv2 package returns. It prepares the inputs, evaluates the
// MASS distance formula on FFT sliding dot products, checks every FFT-path
// entry against a forward roundoff bound, and recalculates directly any
// window the bound cannot certify within the accuracy target. Direct results
// do not receive a separate error-bound check.
//
// The package is internal to the module. Its exported identifiers exist so
// that the public package and its tests can reach them; they are not part of
// the module's supported API and may change without notice.
package distance

import (
	"errors"
	"math"
)

// ErrConstantQuery is returned by Compute when every query value is equal, so
// the query has no z-normalized form.
var ErrConstantQuery = errors.New("query has no variation")

// Profile is the result of Compute: the distances and the number of windows
// that were recalculated directly instead of being certified on the FFT path.
// Tests use the count to verify which path produced a result.
type Profile struct {
	// Distances holds one z-normalized Euclidean distance per window of the
	// series; entry i is the distance to the window starting at index i. A
	// constant window receives positive infinity.
	Distances []float64

	// DirectlyCalculatedWindows counts the windows whose FFT-path distance
	// could not be certified and were recalculated from the original values.
	// Constant windows excluded by equality checks do not count.
	DirectlyCalculatedWindows int
}

// preparedQuery holds the forms of the query that the two distance paths
// consume: the prepared values reversed for convolution, the z-normalized
// values for direct calculation, and the prepared statistics with bounds.
type preparedQuery struct {
	reversed   []float64
	normalized []float64
	statistics windowStatistics
	sigma      float64
}

// Compute calculates the distance profile of query against every window of
// timeSeries of the same length. FFT-path entries are certified against
// DistanceAccuracyTarget; direct results have the accuracy limitations
// described on directZNormalizedDistance.
//
// The caller must have validated the inputs: both must be nonempty and
// finite, and the query must not be longer than the series. Returns
// ErrConstantQuery for a query whose values are all equal. The inputs are not
// modified.
//
// The series and query are each prepared with centerAndScale, so the FFT
// receives O(1) magnitudes and large common offsets are removed before any
// dot product.
func Compute(timeSeries, query []float64) (Profile, error) {
	n := len(timeSeries)
	m := len(query)

	// Prepare the query, the series, its sliding statistics, and the sliding
	// dot products. The convolution's only error is for empty inputs, which
	// the caller has excluded, but it is still propagated rather than ignored.
	prepared, constant := prepareQuery(query)
	if constant {
		return Profile{}, ErrConstantQuery
	}
	profile := Profile{Distances: make([]float64, n-m+1)}
	preparedTimeSeries, constant := centerAndScale(timeSeries)
	if constant {
		for i := range profile.Distances {
			profile.Distances[i] = math.Inf(1)
		}
		return profile, nil
	}
	// Only original-value equality proves that a distance is undefined.
	// Global preparation can erase tiny, but representable, local variation.
	constantCandidates := constantWindows(timeSeries, m)
	statistics := slidingWindowStatistics(preparedTimeSeries, m)
	dotProducts, dotProductRoundoff, err := fftConvolutionLinear(preparedTimeSeries, prepared.reversed)
	if err != nil {
		return Profile{}, err
	}

	// Evaluate every window, certifying the FFT result or recalculating.
	context := prepared.fftContext(m, dotProductRoundoff)
	var directScratch []float64
	for i := range profile.Distances {
		if constantCandidates[i] {
			profile.Distances[i] = math.Inf(1)
			continue
		}
		sigma := statistics.sigmas[i]
		if sigma > 0 {
			distanceSquared, bounds := context.distanceSquared(
				dotProducts[m+i-1],
				statistics.means[i],
				statistics.meanErrors[i],
				sigma,
				statistics.sigmaRelativeErrors[i],
			)
			if !fftDistanceRequiresDirectVerification(distanceSquared, bounds) {
				profile.Distances[i] = math.Sqrt(distanceSquared)
				continue
			}
		}
		if directScratch == nil {
			directScratch = make([]float64, m)
		}
		profile.Distances[i] = directZNormalizedDistance(timeSeries[i:i+m], prepared.normalized, directScratch)
		profile.DirectlyCalculatedWindows++
	}

	return profile, nil
}

// prepareQuery centers and scales the query and derives the forms in
// preparedQuery. constant reports a query with no variation, which has no
// z-normalized form.
func prepareQuery(query []float64) (prepared preparedQuery, constant bool) {
	m := len(query)
	values, constant := centerAndScale(query)
	if constant {
		return preparedQuery{}, true
	}
	prepared.statistics = computeWindowStatistics(values)
	prepared.sigma = math.Sqrt(prepared.statistics.sumOfSquaredDeviations / float64(m))
	prepared.normalized = make([]float64, m)
	prepared.reversed = make([]float64, m)
	for index, value := range values {
		prepared.normalized[index] = value / prepared.sigma
		prepared.reversed[m-1-index] = value
	}
	return prepared, false
}

// fftContext returns the per-call constants of the FFT path for this query
// and the given convolution roundoff bound.
func (prepared preparedQuery) fftContext(windowSize int, dotProductRoundoff float64) fftPathContext {
	return fftPathContext{
		windowSize:              float64(windowSize),
		dotProductRoundoff:      dotProductRoundoff,
		queryMean:               prepared.statistics.mean,
		queryMeanError:          prepared.statistics.meanError,
		querySigma:              prepared.sigma,
		querySigmaRelativeError: prepared.statistics.sumError/(2*prepared.statistics.sumOfSquaredDeviations) + float64Epsilon,
	}
}

// directZNormalizedDistance calculates one distance from the original window
// values, preparing the window on its own so that its variation is
// represented at full precision regardless of the rest of the series.
//
// normalizedQuery must already have zero mean and unit population standard
// deviation. Because the window is anchored on its own first value, its
// prepared sigma is at least 1/√(2m), so the preparation and statistics
// rounding are expected to keep the result within DistanceAccuracyTarget for
// window lengths up to about one million. The final sum of squared differences
// uses ordinary summation, and no error bound is checked on the returned
// distance. Longer windows are accepted without guaranteeing the target.
// A constant window returns positive infinity. preparedWindow is scratch
// storage of the same length as the window and must not alias either input.
func directZNormalizedDistance(timeSeriesWindow, normalizedQuery, preparedWindow []float64) float64 {
	constant := centerAndScaleInto(timeSeriesWindow, preparedWindow)
	if constant {
		return math.Inf(1)
	}
	windowStatistics := computeWindowStatistics(preparedWindow)
	windowSigma := math.Sqrt(windowStatistics.sumOfSquaredDeviations / float64(len(preparedWindow)))

	var distanceSquared float64
	for index, windowValue := range preparedWindow {
		difference := windowValue/windowSigma - normalizedQuery[index]
		distanceSquared += difference * difference
	}
	return math.Sqrt(distanceSquared)
}

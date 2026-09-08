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

package testutil

import (
	"math"

	"gonum.org/v1/gonum/stat"
)

// NaiveSlidingMeanStddev recomputes the population mean and standard
// deviation of every window from scratch with gonum's two-pass statistics. It
// shares no code with the production sliding recurrence and returns nil slices
// when windowSize exceeds len(data).
func NaiveSlidingMeanStddev(data []float64, windowSize int) (means, sigmas []float64) {
	n := len(data)
	if windowSize > n {
		return nil, nil
	}

	means = make([]float64, n-windowSize+1)
	sigmas = make([]float64, n-windowSize+1)
	for i := 0; i <= n-windowSize; i++ {
		means[i], sigmas[i] = stat.PopMeanStdDev(data[i:i+windowSize], nil)
	}

	return means, sigmas
}

// OracleDistanceProfile independently calculates every distance with a
// two-pass mean and a scaled Euclidean norm. It does not reuse production
// preparation, rolling statistics, convolution, or fallback code. It is fast
// enough for long fixtures but can overflow on opposite-sign values near the
// float64 extremes; ExactDistanceProfile covers those. Constant windows
// receive positive infinity, and the query must not be longer than the series.
func OracleDistanceProfile(timeSeries, query []float64) []float64 {
	normalizedQuery, queryIsConstant := independentlyNormalize(query)
	distances := make([]float64, len(timeSeries)-len(query)+1)
	for index := range distances {
		normalizedWindow, windowIsConstant := independentlyNormalize(timeSeries[index : index+len(query)])
		if queryIsConstant || windowIsConstant {
			distances[index] = math.Inf(1)
			continue
		}

		var distanceSquared float64
		for valueIndex, windowValue := range normalizedWindow {
			difference := windowValue - normalizedQuery[valueIndex]
			distanceSquared += difference * difference
		}
		distances[index] = math.Sqrt(distanceSquared)
	}
	return distances
}

// independentlyNormalize translates by the first value, then uses an
// incremental mean and math.Hypot. It avoids the production implementation's
// maximum-difference scaling and compensated statistics. It reports constant
// when the values have no variation, in which case normalized is nil.
func independentlyNormalize(values []float64) (normalized []float64, constant bool) {
	anchor := values[0]
	differences := make([]float64, len(values))
	var meanDifference float64
	for index, value := range values {
		differences[index] = value - anchor
		meanDifference += (differences[index] - meanDifference) / float64(index+1)
	}

	centered := make([]float64, len(values))
	var centeredNorm float64
	for index, difference := range differences {
		centered[index] = difference - meanDifference
		centeredNorm = math.Hypot(centeredNorm, centered[index])
	}
	if centeredNorm == 0 {
		return nil, true
	}

	normalized = make([]float64, len(values))
	normalizationScale := math.Sqrt(float64(len(values))) / centeredNorm
	for index, value := range centered {
		normalized[index] = value * normalizationScale
	}
	return normalized, false
}

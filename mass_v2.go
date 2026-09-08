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

// Package massv2 implements version 2 of Mueen's Algorithm for Similarity
// Search (MASS_V2). MASS is an algorithm to create Distance Profile of a query
// to a long time series: given a query subsequence Q of length m and a long
// time-series T of length n, MASS computes the z-normalized Euclidean distance
// between Q and every subsequence in T of length m. Additionally, this package
// provides convenience functions to find the best match, or the top K matches,
// in the time-series.
//
// The absolute accuracy target is 1e-7 in z-normalized distance units. The
// FFT path checks each entry against a forward roundoff bound. Any entry it
// cannot certify within the target, including exact and near-exact matches,
// is recalculated directly from the original window. Direct recalculation
// is expected to meet the target for query lengths up to about one million
// values, but it does not check an error bound on the returned distance.
// Longer queries are accepted without guaranteeing the target for directly
// recalculated distances. The target is not an unconditional guarantee for
// every finite result; a nil error is not a separate accuracy certification.
//
// Citation: Abdullah Mueen, Sheng Zhong, Yan Zhu, Michael Yeh, Kaveh Kamgar,
// Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2022), The
// Fastest Similarity Search Algorithm for Time Series Subsequences under
// Euclidean Distance, URL:
// http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
package massv2

import (
	"errors"
	"fmt"
	"math"

	"github.com/pinkhop/massv2-go/internal/distance"
)

var (
	// ErrEmptyQuery is returned when the query is nil or empty.
	ErrEmptyQuery = errors.New("empty or nil query")
	// ErrEmptyTimeSeries is returned when the time-series is nil or empty.
	ErrEmptyTimeSeries = errors.New("empty or nil time-series")
	// ErrQueryHasZeroVariance is returned when every query value is equal,
	// so the query has no z-normalized form.
	ErrQueryHasZeroVariance = errors.New("query has zero variance (all values are the same)")
	// ErrQueryLongerThanTimeSeries is returned when the query has more
	// values than the time-series.
	ErrQueryLongerThanTimeSeries = errors.New("query length exceeds time-series length")
	// ErrQueryNotFinite is returned when the query contains NaN or an
	// infinite value.
	ErrQueryNotFinite = errors.New("query contains NaN or an infinite value")
	// ErrTimeSeriesNotFinite is returned when the time-series contains NaN
	// or an infinite value.
	ErrTimeSeriesNotFinite = errors.New("time-series contains NaN or an infinite value")
)

// MASSV2 computes the z-normalized Euclidean distance between the query and
// every subsequence of the same length in the time-series.
//
// The result has len(timeSeries) − len(query) + 1 entries; entry i is the
// distance to the window starting at index i. Each window and the query are
// normalized by their population mean and population standard deviation. A
// window whose values are all equal has no z-normalized form and receives
// positive infinity (+Inf), even when the returned error is nil. Callers
// must check for these entries (for example, with math.IsInf(d, 1)) before
// using distances in arithmetic or serialization that requires finite values.
// FindBestMatch and FindTopKMatches select only finite distances.
//
// The absolute accuracy target is 1e-7, subject to the direct-recalculation
// limits described in the package documentation.
//
// Time is O(n log n) for the FFT path plus O(m) for each window that the
// FFT roundoff bound cannot certify. Exact and near-exact matches are always
// recalculated, as are nonconstant windows whose variation is tiny compared
// with the series' overall range, so a profile in which every window is such
// a case costs O(n·m). Constant windows skip statistics rebuilding and direct
// recalculation after linear equality scans. An entirely constant series
// skips the FFT and takes O(n+m) time. Space is O(n) beyond the inputs,
// including one lazily allocated O(m) buffer reused for direct recalculation.
//
// The inputs are not modified and may be read concurrently by other calls;
// the caller must not modify them during the call. Both inputs must be
// nonempty and finite, the query must not be longer than the time-series, and
// the query must not be constant; the returned error identifies which
// condition failed.
func MASSV2(timeSeries, query []float64) (distances []float64, err error) {
	if err := validateInputs(timeSeries, query); err != nil {
		return nil, err
	}
	profile, err := distance.Compute(timeSeries, query)
	if err != nil {
		return nil, publicProfileError(err)
	}
	return profile.Distances, nil
}

// validateInputs checks the structural and finiteness preconditions of
// MASSV2 and returns the first violated one as its exported error.
func validateInputs(timeSeries, query []float64) error {
	switch {
	case len(query) == 0:
		return ErrEmptyQuery
	case len(timeSeries) == 0:
		return ErrEmptyTimeSeries
	case len(query) > len(timeSeries):
		return ErrQueryLongerThanTimeSeries
	case !allFinite(query):
		return ErrQueryNotFinite
	case !allFinite(timeSeries):
		return ErrTimeSeriesNotFinite
	}
	return nil
}

// allFinite reports whether no value is NaN or infinite.
func allFinite(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// publicProfileError maps an error from the internal profile computation to
// the exported error that describes it, so callers can match with errors.Is
// without depending on internal sentinels. A constant query is the only
// failure the computation reports for validated inputs; anything else is
// wrapped so that it is never dropped.
func publicProfileError(err error) error {
	if errors.Is(err, distance.ErrConstantQuery) {
		return ErrQueryHasZeroVariance
	}
	return fmt.Errorf("computing distance profile: %w", err)
}

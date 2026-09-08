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

package massv2_test

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"gonum.org/v1/gonum/stat"

	massv2 "github.com/pinkhop/massv2-go"
	"github.com/pinkhop/massv2-go/internal/testutil"
)

// distanceProfileTestTolerance is the tolerance for comparing a distance
// profile against an independent oracle. It independently pins the documented
// target; passing confirms agreement for the tested fixtures only.
const distanceProfileTestTolerance = 1e-7

func TestMASSV2_PlateauTransitions_MatchExactProfile(t *testing.T) {
	t.Parallel()
	for name, series := range map[string][]float64{
		"plateaus":              {2, 2, 2, 2, 3, -1, 4, 4, 4, 4, 0, -2, -2, -2, -2},
		"lost global variation": {1e300, 0, 0, 0, 1e-300, 2e-300, 3e-300, 0, 0, 0},
		"extreme scratch reuse": {-math.MaxFloat64, 0, math.MaxFloat64, 0, 0, 0, math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64, 3 * math.SmallestNonzeroFloat64},
		"signed zeros":          {0, math.Copysign(0, -1), 0, 1, 2, 0, 0, 0},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			// GIVEN plateaus and nonconstant windows, with an independent reference.
			query := []float64{1, 2, 3}
			original := slices.Clone(series)
			expected := testutil.ExactDistanceProfile(series, query)
			// WHEN the complete public profile is computed.
			actual, err := massv2.MASSV2(series, query)
			// THEN constants remain infinite and all other distances agree.
			if err != nil {
				t.Fatal(err)
			}
			testutil.AssertDistanceProfilesEqual(t, actual, expected, err, 1e-7)
			if !slices.Equal(series, original) || !slices.Equal(query, []float64{1, 2, 3}) {
				t.Fatal("inputs were modified")
			}
		})
	}
}

func TestMASSV2_FallbackAllocations_DoNotGrowPerWindow(t *testing.T) {
	// AllocsPerRun changes GOMAXPROCS, so this allocation test is not parallel.
	for _, constant := range []bool{true, false} {
		t.Run(fmt.Sprintf("constant=%t", constant), func(t *testing.T) {
			// GIVEN either constant windows or exact ramp matches requiring fallback.
			series := make([]float64, 2048)
			if !constant {
				for i := range series {
					series[i] = float64(i)
				}
			}
			query := []float64{1, 2, 3, 4, 5}
			var calculationError error
			// WHEN allocations are measured across calls with thousands of windows.
			allocations := testing.AllocsPerRun(3, func() {
				_, calculationError = massv2.MASSV2(series, query)
			})
			// THEN a generous fixed budget excludes a buffer allocation per window.
			if calculationError != nil {
				t.Fatal(calculationError)
			}
			if allocations > 100 {
				t.Fatalf("got %.0f allocations; want at most 100", allocations)
			}
		})
	}
}

func TestMASSV2_BasicFunctionality(t *testing.T) {
	t.Parallel()

	// GIVEN a simple series with a known exact occurrence of the query.
	timeSeries := []float64{1.1, 1.9, 4.1, 8.1, 15.8, 15.1, 12.9, 9.25, 1.2, 0.1}
	query := []float64{4.1, 8.1, 15.8}
	const (
		expectedLength     = 8
		expectedMotifIndex = 2
	)

	// WHEN the public distance-profile API is called.
	actualDistances, err := massv2.MASSV2(timeSeries, query)
	// THEN the profile has one entry per window, no negative entries, and its
	// minimum is the exact occurrence.
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}
	if len(actualDistances) != expectedLength {
		t.Errorf("expected %d distances, got %d", expectedLength, len(actualDistances))
	}
	for i, dist := range actualDistances {
		if math.IsNaN(dist) || math.IsInf(dist, 0) || dist < 0 {
			t.Errorf("expected finite nonnegative distance at index %d, got %v", i, dist)
		}
	}

	minIdx := 0
	minDist := actualDistances[0]
	for i, dist := range actualDistances {
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	if minIdx != expectedMotifIndex {
		t.Errorf("expected perfect match at index %d, got index %d with distance %f\ndistances=%#v", expectedMotifIndex, minIdx, minDist, actualDistances)
	}
	if !testutil.AlmostEqual(minDist, 0, distanceProfileTestTolerance) {
		t.Errorf("expected perfect match distance to be very close to 0.0, got %e", minDist)
	}
}

func TestMASSV2_SelfMatch(t *testing.T) {
	t.Parallel()

	// GIVEN a query copied from the series.
	timeSeries := testutil.GenerateSyntheticData(100, testutil.DefaultSeed0, testutil.DefaultSeed1)
	startIdx := 20
	queryLength := 10
	query := make([]float64, queryLength)
	copy(query, timeSeries[startIdx:startIdx+queryLength])

	// WHEN the public distance-profile API is called.
	distances, err := massv2.MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// THEN the copied window is an exact match.
	if !testutil.AlmostEqual(distances[startIdx], 0, distanceProfileTestTolerance) {
		t.Errorf("Self-match at index %d should have distance ~0, got %.12e", startIdx, distances[startIdx])
	}
}

func TestMASSV2_IdenticalElements(t *testing.T) {
	t.Parallel()

	// GIVEN repeated shapes at several offsets and scales.
	timeSeries := []float64{1, 2, 3, -3, -2, -1, 4, 5, 6, 12, 24}
	query := []float64{1, 2, 3}

	// WHEN the public distance-profile API is called.
	distances, err := massv2.MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// THEN each repeated shape is an exact match.
	for _, idx := range []int{0, 3, 6} {
		if !testutil.AlmostEqual(distances[idx], 0, distanceProfileTestTolerance) {
			t.Errorf("Expected perfect match at index %d, got distance %.12e", idx, distances[idx])
		}
	}
}

func TestMASSV2_ErrorCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		timeSeries  []float64
		query       []float64
		expectedErr error
	}{
		{
			name:        "Empty time series",
			timeSeries:  []float64{},
			query:       []float64{1, 2, 3},
			expectedErr: massv2.ErrEmptyTimeSeries,
		},
		{
			name:        "Empty query",
			timeSeries:  []float64{1, 2, 3, 4, 5},
			query:       []float64{},
			expectedErr: massv2.ErrEmptyQuery,
		},
		{
			name:        "Query longer than time series",
			timeSeries:  []float64{1, 2, 3},
			query:       []float64{1, 2, 3, 4, 5},
			expectedErr: massv2.ErrQueryLongerThanTimeSeries,
		},
		{
			name:        "Zero variance query",
			timeSeries:  []float64{1, 2, 3, 4, 5},
			query:       []float64{2, 2, 2},
			expectedErr: massv2.ErrQueryHasZeroVariance,
		},
		{
			name:       "Valid input",
			timeSeries: []float64{1, 2, 3, 4, 5},
			query:      []float64{2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := massv2.MASSV2(tt.timeSeries, tt.query)
			if !errors.Is(err, tt.expectedErr) {
				t.Errorf("expected error %v, got %v", tt.expectedErr, err)
			}
		})
	}
}

func TestMASSV2_NonfiniteInputs_ReturnIdentifyingErrors(t *testing.T) {
	t.Parallel()

	nonfiniteValues := map[string]float64{
		"NaN":  math.NaN(),
		"+Inf": math.Inf(1),
		"-Inf": math.Inf(-1),
	}
	positions := map[string]func(length int) int{
		"beginning": func(int) int { return 0 },
		"middle":    func(length int) int { return length / 2 },
		"end":       func(length int) int { return length - 1 },
	}
	baseSeries := []float64{7, 6, 1, 2, 4, 5, 3}
	baseQuery := []float64{1, 2, 4}

	for valueName, value := range nonfiniteValues {
		for positionName, position := range positions {
			t.Run(fmt.Sprintf("%s at %s of query", valueName, positionName), func(t *testing.T) {
				t.Parallel()
				query := slices.Clone(baseQuery)
				query[position(len(query))] = value
				assertAllPublicOperationsFail(t, baseSeries, query, massv2.ErrQueryNotFinite)
			})
			t.Run(fmt.Sprintf("%s at %s of time series", valueName, positionName), func(t *testing.T) {
				t.Parallel()
				timeSeries := slices.Clone(baseSeries)
				timeSeries[position(len(timeSeries))] = value
				assertAllPublicOperationsFail(t, timeSeries, baseQuery, massv2.ErrTimeSeriesNotFinite)
			})
		}
	}
}

func TestMASSV2_ZeroVarianceSubsequences(t *testing.T) {
	t.Parallel()

	// GIVEN a series containing two constant windows and one exact match.
	timeSeries := []float64{1, 1, 1, 2, 3, 4, 5, 5, 5, 6}
	query := []float64{2, 3, 4}
	expectedDistances := testutil.ExactDistanceProfile(timeSeries, query)

	// WHEN the public distance-profile API is called.
	actualDistances, err := massv2.MASSV2(timeSeries, query)

	// THEN constant windows remain +Inf, and all other distances match the oracle.
	testutil.AssertDistanceProfilesEqual(t, actualDistances, expectedDistances, err, distanceProfileTestTolerance)
}

func TestMASSV2_ConstantTimeSeries_ReturnsInfiniteProfile(t *testing.T) {
	t.Parallel()

	// GIVEN a constant series and a nonconstant query.
	timeSeries := []float64{2, 2, 2, 2, 2}
	query := []float64{1, 2, 4}

	// WHEN the public distance-profile API is called.
	actualDistances, err := massv2.MASSV2(timeSeries, query)

	// THEN every window is +Inf and no error is returned.
	testutil.AssertDistanceProfilesEqual(t, actualDistances, []float64{math.Inf(1), math.Inf(1), math.Inf(1)}, err, distanceProfileTestTolerance)
}

func TestMASSV2_SineWavePattern(t *testing.T) {
	t.Parallel()

	// GIVEN a sine wave and a query cut from it.
	const n = 1000
	timeSeries := testutil.GenerateSineWave(n, 0.1, 1.0, 0)
	queryStart := 100
	queryLength := 50
	query := timeSeries[queryStart : queryStart+queryLength]

	// WHEN the public distance-profile API is called.
	distances, err := massv2.MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// THEN the source window is an exact match and periodicity produces others.
	if !testutil.AlmostEqual(distances[queryStart], 0, distanceProfileTestTolerance) {
		t.Errorf("Expected perfect match at index %d, got distance %.12e", queryStart, distances[queryStart])
	}
	goodMatches := 0
	for _, dist := range distances {
		if dist < 0.1 {
			goodMatches++
		}
	}
	if goodMatches < 2 {
		t.Errorf("Expected multiple good matches in periodic data, found only %d", goodMatches)
	}
}

func TestMASSV2_NumericalStability(t *testing.T) {
	t.Parallel()

	scales := []float64{1e-200, 1e-8, 1e-5, 1e-3, 1, 1e3, 1e5, 1e200}

	for _, scale := range scales {
		t.Run(fmt.Sprintf("Scale_%e", scale), func(t *testing.T) {
			t.Parallel()

			// GIVEN the same shape at a given scale.
			baseData := []float64{-1, 1, 4, 5, 6, 6, 8, 12, 20, 36}
			timeSeries := make([]float64, len(baseData))
			for i, v := range baseData {
				timeSeries[i] = v * scale
			}
			query := []float64{3 * scale, 4 * scale, 5 * scale}

			// WHEN the public distance-profile API is called.
			distances, err := massv2.MASSV2(timeSeries, query)
			if err != nil {
				t.Fatalf("MASSV2 failed at scale %e: %v", scale, err)
			}

			// THEN the exact occurrence at index 2 is the minimum and is zero.
			minIdx := 0
			minDist := distances[0]
			for i, dist := range distances {
				if dist < minDist {
					minDist = dist
					minIdx = i
				}
			}
			if minIdx != 2 {
				t.Errorf("At scale %e: expected perfect match at index 2, got index %d", scale, minIdx)
			}
			if !testutil.AlmostEqual(minDist, 0, distanceProfileTestTolerance) {
				t.Errorf("At scale %e: perfect match distance %.12e should be close to 0", scale, minDist)
			}
		})
	}
}

func TestMASSV2_IndependentlyScaledInputs_PreserveNormalizedDistance(t *testing.T) {
	t.Parallel()

	// GIVEN independently scaled, nonconstant inputs with the same shape.
	timeSeries := []float64{1, 2, 4}
	query := []float64{1e-8, 2e-8, 4e-8}

	// WHEN the public distance-profile API is called.
	actualDistances, err := massv2.MASSV2(timeSeries, query)

	// THEN the nonconstant query is accepted and its normalized distance is zero.
	testutil.AssertDistanceProfilesEqual(t, actualDistances, []float64{0}, err, distanceProfileTestTolerance)
}

func TestMASSV2_ExtremeFiniteValues_MatchExactProfile(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		timeSeries []float64
		query      []float64
	}{
		{
			name:       "large offset exact match",
			timeSeries: []float64{10_000_001, 10_000_002, 10_000_004},
			query:      []float64{10_000_001, 10_000_002, 10_000_004},
		},
		{
			name:       "very large offset exact match",
			timeSeries: []float64{1_000_000_001, 1_000_000_002, 1_000_000_004},
			query:      []float64{1_000_000_001, 1_000_000_002, 1_000_000_004},
		},
		{
			name:       "large offset nonmatch is not a perfect match",
			timeSeries: []float64{100_000_002, 100_000_001, 100_000_004},
			query:      []float64{100_000_001, 100_000_002, 100_000_004},
		},
		{
			name:       "large finite magnitude exact match",
			timeSeries: []float64{1e200, 2e200, 4e200},
			query:      []float64{1, 2, 4},
		},
		{
			name:       "anticorrelated shape has maximum distance",
			timeSeries: []float64{1, 2, 4},
			query:      []float64{-1, -2, -4},
		},
		{
			name:       "window after large outlier remains accurate",
			timeSeries: []float64{1e9, 1, 2, 4, 8},
			query:      []float64{1, 2, 4},
		},
		{
			name:       "two element large offset exact match",
			timeSeries: []float64{134_217_727, 134_217_729},
			query:      []float64{-1, 1},
		},
		{
			name:       "adjacent values near maximum float remain distinct",
			timeSeries: []float64{math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64},
			query:      []float64{-1, 1},
		},
		{
			name:       "opposite sign extremes whose difference overflows",
			timeSeries: []float64{math.MaxFloat64, -math.MaxFloat64, 0, 1},
			query:      []float64{1, -1, 0},
		},
		{
			name:       "opposite sign extremes with ordinary values between",
			timeSeries: []float64{-math.MaxFloat64, 0, math.MaxFloat64},
			query:      []float64{-1, 0, 1},
		},
		{
			name:       "subnormal values",
			timeSeries: []float64{5e-324, 1e-323, 2e-323, 1.5e-323},
			query:      []float64{1, 2, 4},
		},
		{
			name:       "extreme value next to ordinary values",
			timeSeries: []float64{math.MaxFloat64, 1, 2, 4, 8},
			query:      []float64{1, 2, 4},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// GIVEN an exact profile that cannot overflow or underflow.
			expectedDistances := testutil.ExactDistanceProfile(testCase.timeSeries, testCase.query)

			// WHEN the public distance-profile API is called.
			actualDistances, err := massv2.MASSV2(testCase.timeSeries, testCase.query)

			// THEN every distance agrees with the exact profile.
			testutil.AssertDistanceProfilesEqual(t, actualDistances, expectedDistances, err, distanceProfileTestTolerance)
		})
	}
}

func TestMASSV2_LargeDepartingOutlier_MatchesIndependentProfile(t *testing.T) {
	t.Parallel()

	// GIVEN a large first observation followed by varied ordinary observations
	// and an exact query occurrence well after the outlier has departed.
	timeSeries := testutil.LargeOutlierSeries()
	query := slices.Clone(timeSeries[500:510])
	expectedDistances := testutil.OracleDistanceProfile(timeSeries, query)

	// WHEN the public distance-profile API is called.
	actualDistances, err := massv2.MASSV2(timeSeries, query)

	// THEN every window, including those after the outlier, agrees with the oracle.
	testutil.AssertDistanceProfilesEqual(t, actualDistances, expectedDistances, err, distanceProfileTestTolerance)
}

// Property-based test using random data
func TestMASSV2_Properties(t *testing.T) {
	t.Parallel()

	seed := []uint64{testutil.DefaultSeed0, testutil.DefaultSeed1}
	prng := testutil.NewPRNG(seed...)

	const numTests = 50
	for i := range numTests {
		n := prng.IntN(500) + 100 // 100 to 600
		m := prng.IntN(n/2) + 3   // 3 to n/2

		timeSeries := testutil.GenerateSyntheticData(n, seed...)
		query := testutil.GenerateSyntheticData(m, seed[1], seed[0])

		// Ensure query has non-zero variance
		querySigma := stat.StdDev(query, nil)
		if querySigma == 0 {
			continue // Skip this iteration
		}

		distances, err := massv2.MASSV2(timeSeries, query)
		if err != nil {
			t.Fatalf("MASSV2 failed on iteration %d (n=%d, m=%d): %v", i, n, m, err)
		}

		// 1. Correct number of distances
		expectedLength := n - m + 1
		if len(distances) != expectedLength {
			t.Errorf("Iteration %d: expected %d distances, got %d", i, expectedLength, len(distances))
		}

		// 2. All distances should be non-negative
		for j, dist := range distances {
			if math.IsNaN(dist) || math.IsInf(dist, 0) || dist < 0 {
				t.Errorf("Iteration %d: distance[%d] = %v should be finite and nonnegative", i, j, dist)
			}
		}

		// 3. Self-match test: insert query into time series and verify perfect match
		if n > 2*m {
			insertPos := m // Insert after first m elements
			testSeries := make([]float64, n)
			copy(testSeries[:insertPos], timeSeries[:insertPos])
			copy(testSeries[insertPos:insertPos+m], query)
			copy(testSeries[insertPos+m:], timeSeries[insertPos+m:])

			selfDistances, selfMatchErr := massv2.MASSV2(testSeries, query)
			if selfMatchErr != nil {
				t.Fatalf("MASSV2 self-match failed on iteration %d (n=%d, m=%d): %v", i, n, m, selfMatchErr)
			}

			if len(selfDistances) != expectedLength {
				t.Fatalf("Iteration %d: expected %d self-match distances, got %d", i, expectedLength, len(selfDistances))
			}
			if !testutil.AlmostEqual(selfDistances[insertPos], 0, distanceProfileTestTolerance) {
				t.Errorf("Iteration %d: self-match distance %f should be close to 0", i, selfDistances[insertPos])
			}
		}
	}
}

// assertAllPublicOperationsFail checks that every public operation returns
// the expected error for the given inputs and no successful result.
func assertAllPublicOperationsFail(t *testing.T, timeSeries, query []float64, expectedErr error) {
	t.Helper()

	distances, err := massv2.MASSV2(timeSeries, query)
	if !errors.Is(err, expectedErr) || distances != nil {
		t.Errorf("MASSV2: expected error %v and nil profile, got %v and %v", expectedErr, err, distances)
	}

	index, dist, err := massv2.FindBestMatch(timeSeries, query)
	if !errors.Is(err, expectedErr) || index != -1 {
		t.Errorf("FindBestMatch: expected error %v and index -1, got %v, index %d, distance %v", expectedErr, err, index, dist)
	}

	indices, distances, err := massv2.FindTopKMatches(timeSeries, query, 2)
	if !errors.Is(err, expectedErr) || indices != nil || distances != nil {
		t.Errorf("FindTopKMatches: expected error %v and nil results, got %v, %v, %v", expectedErr, err, indices, distances)
	}
}

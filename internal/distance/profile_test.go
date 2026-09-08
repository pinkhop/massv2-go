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

package distance

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/pinkhop/massv2-go/internal/testutil"
)

// profileTestTolerance is the tolerance for comparing a distance profile
// against an independent oracle. It independently pins the documented
// target; passing confirms agreement for the tested fixtures only.
const profileTestTolerance = 1e-7

func TestCompute_ConstantWindows_SkipDirectCalculation(t *testing.T) {
	t.Parallel()
	// GIVEN a constant series and a nonconstant query.
	series := make([]float64, 4096)
	// WHEN the profile is calculated.
	profile, err := Compute(series, []float64{1, 2, 3})
	// THEN constant exclusion does not count as direct recalculation.
	if err != nil {
		t.Fatal(err)
	}
	if profile.DirectlyCalculatedWindows != 0 {
		t.Fatalf("directly calculated %d constant windows", profile.DirectlyCalculatedWindows)
	}
	if len(profile.Distances) != len(series)-2 {
		t.Fatal("wrong profile length")
	}
	for _, value := range profile.Distances {
		if !math.IsInf(value, 1) {
			t.Fatalf("constant distance = %v", value)
		}
	}
}

func TestCompute_ConstantQuery_ReturnsErrConstantQuery(t *testing.T) {
	t.Parallel()

	// GIVEN a query whose values are all equal.
	timeSeries := []float64{1, 2, 3, 4, 5}
	query := []float64{2, 2, 2}

	// WHEN the distance profile is calculated.
	profile, err := Compute(timeSeries, query)

	// THEN the identifying error is returned with an empty profile.
	if !errors.Is(err, ErrConstantQuery) {
		t.Errorf("expected ErrConstantQuery, got %v", err)
	}
	if profile.Distances != nil || profile.DirectlyCalculatedWindows != 0 {
		t.Errorf("expected an empty profile, got %+v", profile)
	}
}

func TestCompute_ConstantQueryAndSeries_ReturnsErrConstantQuery(t *testing.T) {
	t.Parallel()
	// GIVEN constants in both inputs, including the whole-series fast path.
	series := []float64{7, 7, 7, 7}
	query := []float64{2, 2}
	// WHEN the profile is calculated.
	profile, err := Compute(series, query)
	// THEN query validation takes precedence over constant-window exclusion.
	if !errors.Is(err, ErrConstantQuery) || profile.Distances != nil {
		t.Fatalf("got %+v, %v; want empty profile and ErrConstantQuery", profile, err)
	}
}

func TestCompute_MixedConstantWindows_OnlyVaryingWindowsCanUseDirectPath(t *testing.T) {
	t.Parallel()
	// GIVEN a single spike separating two long constant runs.
	series := make([]float64, 4096)
	series[2048] = 1
	query := []float64{1, 2}
	expected := testutil.OracleDistanceProfile(series, query)
	// WHEN the profile is calculated.
	profile, err := Compute(series, query)
	// THEN only the two windows containing the spike can need direct work.
	testutil.AssertDistanceProfilesEqual(t, profile.Distances, expected, err, 1e-7)
	if profile.DirectlyCalculatedWindows > 2 {
		t.Fatalf("directly calculated %d windows; only two vary", profile.DirectlyCalculatedWindows)
	}
}

func TestCompute_ManyExactMatches_UseDirectPath(t *testing.T) {
	t.Parallel()

	// GIVEN a ramp whose every five-value window has the query's normalized shape.
	timeSeries := make([]float64, 10_000)
	for index := range timeSeries {
		timeSeries[index] = float64(index)
	}
	query := []float64{0, 1, 2, 3, 4}
	expectedDistances := testutil.OracleDistanceProfile(timeSeries, query)

	// WHEN the distance profile is calculated.
	profile, err := Compute(timeSeries, query)

	// THEN every window agrees with the oracle, and every exact match was
	// calculated directly because the FFT formula cannot certify a distance
	// near zero.
	testutil.AssertDistanceProfilesEqual(t, profile.Distances, expectedDistances, err, profileTestTolerance)
	if profile.DirectlyCalculatedWindows != len(expectedDistances) {
		t.Errorf("expected all %d exact matches to be calculated directly, got %d", len(expectedDistances), profile.DirectlyCalculatedWindows)
	}
}

func TestCompute_ModerateDepartingOutlier_StaysOnFFTPath(t *testing.T) {
	t.Parallel()

	// GIVEN an outlier of a few hundred standard deviations and an exact
	// query occurrence after it leaves the candidate window.
	timeSeries := make([]float64, 1_000)
	timeSeries[0] = 500
	for index := 1; index < len(timeSeries); index++ {
		timeSeries[index] = math.Cos(float64(index)*0.19) + float64(index%13)/9
	}
	query := slices.Clone(timeSeries[500:510])
	expectedDistances := testutil.OracleDistanceProfile(timeSeries, query)

	// WHEN the distance profile is calculated.
	profile, err := Compute(timeSeries, query)

	// THEN the complete profile agrees with the oracle and the outlier does
	// not force ordinary windows off the FFT path.
	testutil.AssertDistanceProfilesEqual(t, profile.Distances, expectedDistances, err, profileTestTolerance)
	assertDirectFractionAtMost(t, profile, 0.02)
}

func TestCompute_OrdinaryInputs_StayOnFFTPath(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		timeSeries []float64
		query      []float64
	}{
		{
			name:       "gaussian n=100000 m=100",
			timeSeries: testutil.GenerateSyntheticData(100_000, 42),
			query:      testutil.GenerateSyntheticData(100, 84),
		},
		{
			name:       "gaussian with a spike of one hundred sigma",
			timeSeries: spikedGaussian(100_000, 100),
			query:      testutil.GenerateSyntheticData(100, 84),
		},
		{
			name:       "gaussian with a spike of ten thousand sigma",
			timeSeries: spikedGaussian(100_000, 1e4),
			query:      testutil.GenerateSyntheticData(100, 84),
		},
		{
			name:       "gaussian with a linear trend of one thousand",
			timeSeries: trendingGaussian(100_000, 1e3),
			query:      testutil.GenerateSyntheticData(100, 84),
		},
		{
			name:       "gaussian with a constant offset of one million",
			timeSeries: offsetGaussian(100_000, 1e6),
			query:      testutil.GenerateSyntheticData(100, 84),
		},
		{
			name:       "well conditioned sinusoids",
			timeSeries: sinusoidFixture(512),
			query:      sinusoidQuery(31),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// GIVEN an independent oracle profile.
			expectedDistances := testutil.OracleDistanceProfile(testCase.timeSeries, testCase.query)

			// WHEN the distance profile is calculated.
			profile, err := Compute(testCase.timeSeries, testCase.query)

			// THEN the FFT path produces nearly every entry, and every entry
			// agrees with the oracle within the accuracy target.
			testutil.AssertDistanceProfilesEqual(t, profile.Distances, expectedDistances, err, profileTestTolerance)
			assertDirectFractionAtMost(t, profile, 0.01)
		})
	}
}

func TestCompute_MillionPointSeries_StaysOnFFTPath(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping million-point FFT path test in short mode")
	}
	t.Parallel()

	// GIVEN a million Gaussian observations and a short query.
	timeSeries := testutil.GenerateSyntheticData(1_000_000, 42)
	query := testutil.GenerateSyntheticData(100, 84)
	expectedDistances := testutil.OracleDistanceProfile(timeSeries, query)

	// WHEN the distance profile is calculated.
	profile, err := Compute(timeSeries, query)

	// THEN the FFT path handles nearly every window and agrees with the oracle.
	testutil.AssertDistanceProfilesEqual(t, profile.Distances, expectedDistances, err, profileTestTolerance)
	assertDirectFractionAtMost(t, profile, 0.01)
}

// assertDirectFractionAtMost fails when more than the given fraction of
// windows were calculated directly instead of on the FFT path.
func assertDirectFractionAtMost(t *testing.T, profile Profile, maximumFraction float64) {
	t.Helper()
	fraction := float64(profile.DirectlyCalculatedWindows) / float64(len(profile.Distances))
	t.Logf("%d of %d windows (%.2f%%) were calculated directly", profile.DirectlyCalculatedWindows, len(profile.Distances), 100*fraction)
	if fraction > maximumFraction {
		t.Errorf("expected at most %.2f%% of windows to leave the FFT path, got %.2f%%", 100*maximumFraction, 100*fraction)
	}
}

// spikedGaussian returns standard normal data with one value replaced by a
// spike of the given size.
func spikedGaussian(n int, spike float64) []float64 {
	data := testutil.GenerateSyntheticData(n, 42)
	data[n/3] = spike
	return data
}

// trendingGaussian returns standard normal data plus a linear trend that
// rises by the given total across the series.
func trendingGaussian(n int, rise float64) []float64 {
	data := testutil.GenerateSyntheticData(n, 42)
	for index := range data {
		data[index] += rise * float64(index) / float64(n)
	}
	return data
}

// offsetGaussian returns standard normal data shifted by a constant offset.
func offsetGaussian(n int, offset float64) []float64 {
	data := testutil.GenerateSyntheticData(n, 42)
	for index := range data {
		data[index] += offset
	}
	return data
}

// sinusoidFixture returns a well-conditioned sum of two sinusoids.
func sinusoidFixture(n int) []float64 {
	data := make([]float64, n)
	for index := range data {
		data[index] = math.Sin(float64(index)*0.17) + math.Cos(float64(index)*0.071)
	}
	return data
}

// sinusoidQuery returns a sinusoidal query that is not copied from
// sinusoidFixture, so no window is an exact match.
func sinusoidQuery(m int) []float64 {
	query := make([]float64, m)
	for index := range query {
		query[index] = math.Sin(float64(index)*0.29+0.4) - math.Cos(float64(index)*0.11)
	}
	return query
}

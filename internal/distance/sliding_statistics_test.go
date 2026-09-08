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
	"math"
	"testing"

	"github.com/pinkhop/massv2-go/internal/testutil"
)

const (
	// slidingMeanTestTolerance bounds the absolute disagreement allowed
	// between a sliding mean and the exact reference on the short
	// prepared-scale fixtures, whose values are O(1) and whose window counts
	// are in the hundreds.
	slidingMeanTestTolerance = 1e-12

	// slidingSigmaTestTolerance bounds the relative disagreement allowed
	// between a sliding standard deviation and the exact reference. The
	// recurrence rebuilds a window once its sum's relative bound exceeds
	// statisticsRebuildRelativeError, and the square root halves that, so
	// this is the accuracy the implementation is designed to hold.
	slidingSigmaTestTolerance = statisticsRebuildRelativeError
)

func TestSlidingWindowStatistics_MatchesExactReferenceOnEveryWindow(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		data       []float64
		windowSize int
	}{
		{
			name:       "basic functionality",
			data:       []float64{1, 2, 4, 8, 16},
			windowSize: 3,
		},
		{
			name:       "single-value windows",
			data:       []float64{7, 7, -3, 0, 1},
			windowSize: 1,
		},
		{
			name:       "entirely constant series",
			data:       []float64{7, 7, 7, 7, 7},
			windowSize: 3,
		},
		{
			name:       "large departing outlier",
			data:       preparedOutlierFixture(1e6, 300),
			windowSize: 10,
		},
		{
			name:       "window becomes constant and varies again",
			data:       []float64{0.3, -0.2, 0.5, 0.5, 0.5, 0.5, 0.5, -0.1, 0.4, 0.1},
			windowSize: 4,
		},
		{
			name:       "alternating high and low variance blocks",
			data:       alternatingVarianceFixture(400),
			windowSize: 25,
		},
		{
			name:       "slowly varying values with tiny local variance",
			data:       tinyVarianceFixture(500),
			windowSize: 20,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// GIVEN an exact reference for every window.
			expectedMeans, expectedSigmas := testutil.ExactSlidingMeanStddev(testCase.data, testCase.windowSize)

			// WHEN the sliding recurrence is evaluated.
			actual := slidingWindowStatistics(testCase.data, testCase.windowSize)

			// THEN every window agrees with the reference within tolerance and
			// within the recurrence's own reported error bound.
			assertSlidingStatisticsMatch(t, actual, expectedMeans, expectedSigmas)
		})
	}
}

func TestSlidingWindowStatistics_ReturnsEmptyWhenWindowExceedsData(t *testing.T) {
	t.Parallel()

	// GIVEN a window longer than the data.
	data := []float64{1, 2, 4, 8, 16, 32}

	// WHEN the sliding recurrence is evaluated.
	actual := slidingWindowStatistics(data, len(data)+1)

	// THEN no statistics are produced.
	if len(actual.means) != 0 || len(actual.sigmas) != 0 {
		t.Errorf("expected no statistics, got %d means and %d sigmas", len(actual.means), len(actual.sigmas))
	}
}

func TestSlidingWindowStatistics_ReportsLargeErrorBoundsOnRawOffsets(t *testing.T) {
	t.Parallel()

	// GIVEN raw data with a large common offset that was not prepared.
	const offset = 1e12
	data := make([]float64, 64)
	for index := range data {
		data[index] = offset + math.Sin(float64(index))
	}

	// WHEN the sliding recurrence is evaluated.
	actual := slidingWindowStatistics(data, 2)

	// THEN the reported bounds admit the loss of precision instead of hiding
	// it, so a caller cannot certify a distance from these statistics.
	expectedMeans, expectedSigmas := testutil.ExactSlidingMeanStddev(data, 2)
	for index := range actual.sigmas {
		relativeError := math.Abs(actual.sigmas[index]-expectedSigmas[index]) / expectedSigmas[index]
		if !(relativeError <= actual.sigmaRelativeErrors[index]) {
			t.Errorf("window %d: relative sigma error %.3e exceeds reported bound %.3e", index, relativeError, actual.sigmaRelativeErrors[index])
		}
		if !testutil.AlmostEqual(actual.means[index], expectedMeans[index], actual.meanErrors[index]) {
			t.Errorf("window %d: mean error %.3e exceeds reported bound %.3e", index, math.Abs(actual.means[index]-expectedMeans[index]), actual.meanErrors[index])
		}
	}
	if actual.sigmaRelativeErrors[1] < 1e-8 {
		t.Errorf("expected the raw offset to produce a large sigma error bound, got %.3e", actual.sigmaRelativeErrors[1])
	}
}

// `go test -short ./...` to skip this test
func TestSlidingWindowStatistics_FloatingPointAccumulationError(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping floating point accumulation error test in short mode")
	}
	t.Parallel()

	// GIVEN (set up)

	const (
		m = 60 * 24 // 1 day of observations @ 1-minute intervals

		// Generate a time-series with ~5.27 million values, which is enough to
		// hold 10 years of observations taken every minute.
		n = 60 * 24 * 366 * 10
	)

	seed := []uint64{testutil.DefaultSeed0, testutil.DefaultSeed1}
	ts := testutil.GenerateNoisySineWave(n, 97, 100, 0, seed...)
	expectedMeans, expectedSigmas := testutil.NaiveSlidingMeanStddev(ts, m)

	// WHEN (operation under test)

	actual := slidingWindowStatistics(ts, m)

	// THEN (assertions)

	if len(actual.means) != len(expectedMeans) {
		t.Fatalf("expected sliding means to have length %d, got %d", len(expectedMeans), len(actual.means))
	}
	if len(actual.sigmas) != len(expectedSigmas) {
		t.Fatalf("expected sliding standard deviations to have length %d, got %d", len(expectedSigmas), len(actual.sigmas))
	}

	// The reference is itself float64 and the raw amplitude is 100, so the
	// mean is compared at the amplitude-scaled bound of the recurrence.
	lastIndex := len(actual.means) - 1
	if !testutil.AlmostEqual(actual.means[lastIndex], expectedMeans[lastIndex], 100*slidingSigmaTestTolerance) {
		t.Errorf("last mean differs from the reference by %e [seed: %v]", math.Abs(actual.means[lastIndex]-expectedMeans[lastIndex]), seed)
	}
	if !testutil.AlmostEqual(actual.sigmas[lastIndex], expectedSigmas[lastIndex], slidingSigmaTestTolerance*expectedSigmas[lastIndex]) {
		t.Errorf("last standard deviation differs from the reference by %e [seed: %v]", math.Abs(actual.sigmas[lastIndex]-expectedSigmas[lastIndex]), seed)
	}
}

// assertSlidingStatisticsMatch checks every window against an exact reference
// and against the recurrence's self-reported error bounds.
func assertSlidingStatisticsMatch(t *testing.T, actual slidingStatistics, expectedMeans, expectedSigmas []float64) {
	t.Helper()
	if len(actual.means) != len(expectedMeans) || len(actual.sigmas) != len(expectedSigmas) {
		t.Fatalf("expected %d windows, got %d means and %d sigmas", len(expectedMeans), len(actual.means), len(actual.sigmas))
	}
	if len(actual.meanErrors) != len(expectedMeans) || len(actual.sigmaRelativeErrors) != len(expectedSigmas) {
		t.Fatalf("expected %d error bounds, got %d mean bounds and %d sigma bounds", len(expectedMeans), len(actual.meanErrors), len(actual.sigmaRelativeErrors))
	}

	for index := range expectedMeans {
		meanError := math.Abs(actual.means[index] - expectedMeans[index])
		if !(meanError <= slidingMeanTestTolerance) {
			t.Errorf("window %d: mean %.17g differs from exact %.17g by %.3e", index, actual.means[index], expectedMeans[index], meanError)
		}
		if !(meanError <= actual.meanErrors[index]) {
			t.Errorf("window %d: mean error %.3e exceeds reported bound %.3e", index, meanError, actual.meanErrors[index])
		}

		sigmaError := math.Abs(actual.sigmas[index] - expectedSigmas[index])
		if expectedSigmas[index] == 0 {
			if !testutil.AlmostEqual(actual.sigmas[index], 0, slidingMeanTestTolerance) {
				t.Errorf("window %d: expected constant window sigma near 0, got %.3e", index, actual.sigmas[index])
			}
			continue
		}
		if !(sigmaError <= slidingSigmaTestTolerance*expectedSigmas[index]) {
			t.Errorf("window %d: sigma %.17g differs from exact %.17g by %.3e", index, actual.sigmas[index], expectedSigmas[index], sigmaError)
		}
		if !(sigmaError <= actual.sigmaRelativeErrors[index]*expectedSigmas[index]) {
			t.Errorf("window %d: relative sigma error %.3e exceeds reported bound %.3e", index, sigmaError/expectedSigmas[index], actual.sigmaRelativeErrors[index])
		}
	}
}

// preparedOutlierFixture returns prepared-scale data with a leading outlier of
// the given size relative to the ordinary variation, then ordinary variation.
func preparedOutlierFixture(outlier float64, n int) []float64 {
	data := make([]float64, n)
	data[0] = outlier
	for index := 1; index < n; index++ {
		data[index] = math.Sin(float64(index)*0.37) + float64(index%11)/7
	}
	prepared, _ := centerAndScale(data)
	return prepared
}

// alternatingVarianceFixture returns prepared-scale data whose variance
// alternates between blocks of large and tiny amplitude.
func alternatingVarianceFixture(n int) []float64 {
	data := make([]float64, n)
	for index := range data {
		amplitude := 1.0
		if (index/50)%2 == 1 {
			amplitude = 1e-6
		}
		data[index] = amplitude * math.Cos(float64(index)*0.61)
	}
	prepared, _ := centerAndScale(data)
	return prepared
}

// tinyVarianceFixture returns prepared-scale data with a slow drift across the
// whole range and tiny local variation, so window sums are much smaller than
// the magnitudes of the values that produce them.
func tinyVarianceFixture(n int) []float64 {
	data := make([]float64, n)
	for index := range data {
		data[index] = float64(index)/float64(n) + 1e-7*math.Sin(float64(index)*1.3)
	}
	prepared, _ := centerAndScale(data)
	return prepared
}

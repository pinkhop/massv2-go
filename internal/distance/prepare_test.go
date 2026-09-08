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
	"fmt"
	"math"
	"testing"

	"github.com/pinkhop/massv2-go/internal/testutil"
)

func TestConstantWindows_AllBinaryPatterns_MatchDirectEquality(t *testing.T) {
	t.Parallel()
	for pattern := range 128 {
		for windowSize := 1; windowSize <= 7; windowSize++ {
			t.Run(fmt.Sprintf("pattern=%d/m=%d", pattern, windowSize), func(t *testing.T) {
				t.Parallel()
				// GIVEN every seven-value binary pattern and every valid window size.
				values := make([]float64, 7)
				for i := range values {
					values[i] = float64((pattern >> i) & 1)
				}
				expected := make([]bool, len(values)-windowSize+1)
				for start := range expected {
					expected[start] = true
					for _, value := range values[start : start+windowSize] {
						expected[start] = expected[start] && value == values[start]
					}
				}
				// WHEN constant windows are identified by the linear scan.
				actual := constantWindows(values, windowSize)
				// THEN every entry matches independent per-window comparisons.
				if len(actual) != len(expected) {
					t.Fatal("wrong window count")
				}
				for i := range expected {
					if actual[i] != expected[i] {
						t.Errorf("window %d: got %t, want %t", i, actual[i], expected[i])
					}
				}
			})
		}
	}
}

func TestCompensatedSum_ErrorWithinDocumentedBound(t *testing.T) {
	t.Parallel()

	// GIVEN values whose naive sum loses many digits.
	values := []float64{1, 1e-16, -1, 1e-16, 3, 1e-16, -3, 1e-16}

	// WHEN the compensated sum is calculated.
	actual := compensatedSum(values)

	// THEN the result is within the documented bound of the exact sum.
	var sumOfMagnitudes float64
	for _, value := range values {
		sumOfMagnitudes += math.Abs(value)
	}
	if !testutil.AlmostEqual(actual, 4e-16, compensatedSumErrorFactor*sumOfMagnitudes) {
		t.Errorf("expected 4e-16, got %.17g", actual)
	}
}

func TestComputeWindowStatistics_BoundsCoverExactError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name   string
		values []float64
	}{
		{name: "prepared gaussian", values: preparedGaussian(257, testutil.DefaultSeed0)},
		{name: "prepared spike", values: preparedSpiked(129, 1e8)},
		{name: "two values", values: []float64{-1, 1}},
		{name: "raw large offset", values: []float64{1e12 + 1, 1e12 + 3, 1e12 - 2}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// GIVEN the exact statistics.
			exactMean, exactSigma := testutil.ExactMeanStddev(testCase.values)
			expectedMean, _ := exactMean.Float64()
			expectedSigma, _ := exactSigma.Float64()
			expectedSum := expectedSigma * expectedSigma * float64(len(testCase.values))

			// WHEN the two-pass statistics and their bounds are calculated.
			actual := computeWindowStatistics(testCase.values)

			// THEN the observed errors are within the reported bounds.
			if meanError := math.Abs(actual.mean - expectedMean); !(meanError <= actual.meanError) {
				t.Errorf("mean error %.3e exceeds bound %.3e", meanError, actual.meanError)
			}
			if sumError := math.Abs(actual.sumOfSquaredDeviations - expectedSum); !(sumError <= actual.sumError+expectedSum*1e-15) {
				t.Errorf("sum of squared deviations error %.3e exceeds bound %.3e", sumError, actual.sumError)
			}
		})
	}
}

// preparedGaussian returns standard normal data after production preparation,
// so magnitudes match what the convolution receives inside Compute.
func preparedGaussian(n int, seed uint64) []float64 {
	prepared, _ := centerAndScale(testutil.GenerateSyntheticData(n, seed))
	return prepared
}

// preparedSpiked returns prepared Gaussian data whose middle value has been
// replaced by a spike of the given size.
func preparedSpiked(n int, spike float64) []float64 {
	data := testutil.GenerateSyntheticData(n, testutil.DefaultSeed0)
	data[n/2] = spike
	prepared, _ := centerAndScale(data)
	return prepared
}

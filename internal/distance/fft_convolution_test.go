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
	"fmt"
	"math"
	"testing"

	"github.com/pinkhop/massv2-go/internal/testutil"
)

// fftConvolutionTestTolerance bounds disagreement with small integer-valued
// expectations whose exact results are representable.
const fftConvolutionTestTolerance = 1e-9

func TestFFTConvolutionLinear(t *testing.T) {
	t.Parallel()

	type TestCase struct {
		Name                string
		InputSignal         []float64
		InputKernel         []float64
		ExpectedDotProducts []float64
		ExpectedError       error
	}

	testCases := []TestCase{
		{
			Name:                "basic functionality",
			InputSignal:         []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			InputKernel:         []float64{5, 4, 3},
			ExpectedDotProducts: []float64{5, 14, 26, 38, 50, 62, 74, 86, 98, 110, 67, 30},
		},
		{
			Name:          "empty signal should return error",
			InputSignal:   []float64{},
			InputKernel:   []float64{5, 4, 3},
			ExpectedError: errEmptyFFTConvolutionInputs,
		},
		{
			Name:          "nil signal should return error",
			InputSignal:   nil,
			InputKernel:   []float64{5, 4, 3},
			ExpectedError: errEmptyFFTConvolutionInputs,
		},
		{
			Name:          "empty kernel should return error",
			InputSignal:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			InputKernel:   []float64{},
			ExpectedError: errEmptyFFTConvolutionInputs,
		},
		{
			Name:          "nil kernel should return error",
			InputSignal:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			InputKernel:   nil,
			ExpectedError: errEmptyFFTConvolutionInputs,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// WHEN (operation under test)
			actualDotProducts, _, err := fftConvolutionLinear(tc.InputSignal, tc.InputKernel)

			// THEN (assertions)
			if tc.ExpectedError != nil {
				if !errors.Is(err, tc.ExpectedError) {
					t.Errorf("expected error %v, got %v", tc.ExpectedError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("fftConvolutionLinear failed: %v", err)
			}

			if len(actualDotProducts) != len(tc.ExpectedDotProducts) {
				t.Fatalf("expected returned slice to have length %d, got length %d", len(tc.ExpectedDotProducts), len(actualDotProducts))
			}
			for i, expected := range tc.ExpectedDotProducts {
				actual := actualDotProducts[i]
				if !testutil.AlmostEqual(actual, expected, fftConvolutionTestTolerance) {
					t.Errorf("expected returned slice index %d to be %f, got %f", i, expected, actual)
				}
			}
		})
	}
}

func TestFFTConvolutionLinear_RoundoffBoundCoversObservedError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name   string
		signal []float64
		kernel []float64
	}{
		{
			name:   "gaussian signal and kernel",
			signal: preparedGaussian(4_096, testutil.DefaultSeed0),
			kernel: preparedGaussian(64, testutil.DefaultSeed1),
		},
		{
			name:   "signal with a large spike",
			signal: preparedSpiked(4_096, 1e6),
			kernel: preparedGaussian(48, testutil.DefaultSeed1),
		},
		{
			name:   "non power of two lengths",
			signal: preparedGaussian(3_001, testutil.DefaultSeed1),
			kernel: preparedGaussian(37, testutil.DefaultSeed0),
		},
		{
			name:   "kernel as long as the signal",
			signal: preparedGaussian(1_024, testutil.DefaultSeed0),
			kernel: preparedGaussian(1_024, testutil.DefaultSeed1),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// GIVEN exact dot products for every output position.
			expected := testutil.ExactDotProducts(testCase.signal, testCase.kernel)

			// WHEN the convolution and its roundoff bound are calculated.
			actual, bound, err := fftConvolutionLinear(testCase.signal, testCase.kernel)
			if err != nil {
				t.Fatalf("fftConvolutionLinear failed: %v", err)
			}

			// THEN no element's error exceeds the bound, and the bound is
			// within a few orders of magnitude of the largest observed error
			// rather than being vacuous.
			var largestError float64
			for index := range expected {
				observed := math.Abs(actual[index] - expected[index])
				largestError = math.Max(largestError, observed)
				if !(observed <= bound) {
					t.Errorf("index %d: observed error %.3e exceeds bound %.3e", index, observed, bound)
				}
			}
			if bound <= 0 || math.IsInf(bound, 0) || math.IsNaN(bound) {
				t.Fatalf("expected a positive finite bound, got %v", bound)
			}
			t.Logf("largest observed error %.3e, bound %.3e, ratio %.1f", largestError, bound, bound/largestError)
			if bound > 1e5*largestError {
				t.Errorf("bound %.3e is more than 1e5 times the largest observed error %.3e", bound, largestError)
			}
		})
	}
}

func TestNextPow2(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		input    int
		expected int
	}{
		{input: 1, expected: 1},
		{input: 2, expected: 2},
		{input: 3, expected: 4},
		{input: 1_000, expected: 1_024},
		{input: 1_024, expected: 1_024},
		{input: 1_025, expected: 2_048},
	}

	for _, testCase := range testCases {
		t.Run(fmt.Sprintf("%d", testCase.input), func(t *testing.T) {
			if actual := nextPow2(testCase.input); actual != testCase.expected {
				t.Errorf("nextPow2(%d): expected %d, got %d", testCase.input, testCase.expected, actual)
			}
		})
	}
}

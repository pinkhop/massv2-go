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

package massv2

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"gonum.org/v1/gonum/stat"
)

const floatToleranceForMASSV2Test = 2e-6

var (
	defaultSeed0 uint64 = 0x7fa2_2276_889c_4782
	defaultSeed1 uint64 = 0xaf4f_33b8_2757_b871
)

////////////////////////////////////////////////////////////////////////////////
// BENCHMARKS

func BenchmarkMASSV2_withMEqual200(b *testing.B) {
	const queryLength = 200
	sizes := []int{50_000, 100_000, 200_000}

	for _, n := range sizes {
		timeSeries := generateSyntheticData(n, 42)
		query := generateSyntheticData(queryLength, 84)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := MASSV2(timeSeries, query)
				if err != nil {
					b.Fatalf("MASSV2 failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkMASSV2_withMEqual1000(b *testing.B) {
	const queryLength = 1_000
	sizes := []int{50_000, 100_000, 200_000}

	for _, n := range sizes {
		timeSeries := generateSyntheticData(n, 42)
		query := generateSyntheticData(queryLength, 84)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := MASSV2(timeSeries, query)
				if err != nil {
					b.Fatalf("MASSV2 failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkMASSV2_withMEqual5000(b *testing.B) {
	const queryLength = 5_000
	sizes := []int{50_000, 100_000, 200_000}

	for _, n := range sizes {
		timeSeries := generateSyntheticData(n, defaultSeed0, defaultSeed1)
		query := generateSyntheticData(queryLength, defaultSeed0, defaultSeed1)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := MASSV2(timeSeries, query)
				if err != nil {
					b.Fatalf("MASSV2 failed: %v", err)
				}
			}
		})
	}
}

////////////////////////////////////////////////////////////////////////////////
// TESTS

func TestMASSV2_BasicFunctionality(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// GIVEN (set up)

	// Simple test case with a known pattern
	timeSeries := []float64{1.1, 1.9, 4.1, 8.1, 15.8, 15.1, 12.9, 9.25, 1.2, 0.1}
	query := []float64{4.1, 8.1, 15.8}
	const (
		expectedLength     = 8
		expectedMotifIndex = 2
	)

	// WHEN (operation under test)

	actualDistances, err := MASSV2(timeSeries, query)

	// THEN (assertions)

	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	if len(actualDistances) != expectedLength {
		t.Errorf("expected %d distances, got %d", expectedLength, len(actualDistances))
	}

	// All distances should be non-negative
	for i, dist := range actualDistances {
		if dist < 0 {
			t.Errorf("found negative distance %f at index %d", dist, i)
		}
	}

	// The perfect match should be at index 2 (subsequence [3,4,5])
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

	// The perfect match should have a distance very close to 0
	if minDist > floatToleranceForMASSV2Test {
		t.Errorf("expected perfect match distance to be very close to 0.0, got %e", minDist)
	}
}

func TestMASSV2_SelfMatch(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Test with a query taken from the time series itself
	timeSeries := generateSyntheticData(100, defaultSeed0, defaultSeed1)
	startIdx := 20
	queryLength := 10
	query := make([]float64, queryLength)
	copy(query, timeSeries[startIdx:startIdx+queryLength])

	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// The perfect match should be at startIdx
	if !almostEqual(distances[startIdx], 0, floatToleranceForMASSV2Test) {
		t.Errorf("Self-match at index %d should have distance ~0, got %.12e", startIdx, distances[startIdx])
	}
}

func TestMASSV2_IdenticalElements(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Test with repeated patterns
	timeSeries := []float64{1, 2, 3, -3, -2, -1, 4, 5, 6, 12, 24}
	query := []float64{1, 2, 3}

	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// Should find perfect matches at indices 0, 3, 6
	perfectMatches := []int{0, 3, 6}
	for _, idx := range perfectMatches {
		if !almostEqual(distances[idx], 0, floatToleranceForMASSV2Test) {
			t.Errorf("Expected perfect match at index %d, got distance %.12e", idx, distances[idx])
		}
	}
}

func TestMASSV2_ErrorCases(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	tests := []struct {
		name       string
		timeSeries []float64
		query      []float64
		expectErr  bool
	}{
		{
			name:       "Empty time series",
			timeSeries: []float64{},
			query:      []float64{1, 2, 3},
			expectErr:  true,
		},
		{
			name:       "Empty query",
			timeSeries: []float64{1, 2, 3, 4, 5},
			query:      []float64{},
			expectErr:  true,
		},
		{
			name:       "Query longer than time series",
			timeSeries: []float64{1, 2, 3},
			query:      []float64{1, 2, 3, 4, 5},
			expectErr:  true,
		},
		{
			name:       "Zero variance query",
			timeSeries: []float64{1, 2, 3, 4, 5},
			query:      []float64{2, 2, 2},
			expectErr:  true,
		},
		{
			name:       "Valid input",
			timeSeries: []float64{1, 2, 3, 4, 5},
			query:      []float64{2, 3, 4},
			expectErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := MASSV2(tt.timeSeries, tt.query)
			if (err != nil) != tt.expectErr {
				t.Errorf("Expected error: %v, got error: %v", tt.expectErr, err != nil)
			}
		})
	}
}

func TestMASSV2_ZeroVarianceSubsequences(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Time series with constant subsequences
	timeSeries := []float64{1, 1, 1, 2, 3, 4, 5, 5, 5, 6}
	query := []float64{2, 3, 4}

	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// Subsequences with zero variance should have infinite distance
	// [1,1,1] at index 0, [5,5,5] at index 6
	if !math.IsInf(distances[0], 1) {
		t.Errorf("Expected infinite distance for zero variance subsequence at index 0, got %f", distances[0])
	}
	if !math.IsInf(distances[6], 1) {
		t.Errorf("Expected infinite distance for zero variance subsequence at index 6, got %f", distances[6])
	}

	// Perfect match should be at index 3
	perfectMatchIdx := 3
	if !almostEqual(distances[perfectMatchIdx], 0, floatToleranceForMASSV2Test) {
		t.Errorf("Expected perfect match at index %d, got distance %f", perfectMatchIdx, distances[perfectMatchIdx])
	}
}

func TestMASSV2_SineWavePattern(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Generate sine wave and search for a pattern within it
	const n = 1000
	timeSeries := generateSineWave(n, 0.1, 1.0, 0)

	// Extract a query from the sine wave
	queryStart := 100
	queryLength := 50
	query := timeSeries[queryStart : queryStart+queryLength]

	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		t.Fatalf("MASSV2 failed: %v", err)
	}

	// Should find the perfect match at queryStart
	if !almostEqual(distances[queryStart], 0, floatToleranceForMASSV2Test) {
		t.Errorf("Expected perfect match at index %d, got distance %.12e", queryStart, distances[queryStart])
	}

	// Due to periodicity of sine wave, should find other good matches
	goodMatches := 0
	for _, dist := range distances {
		if dist < 0.1 { // threshold for "good" match
			goodMatches++
		}
	}

	if goodMatches < 2 {
		t.Errorf("Expected multiple good matches in periodic data, found only %d", goodMatches)
	}
}

func TestMASSV2_NumericalStability(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Test with various scales to check numerical stability
	scales := []float64{1e-5, 1e-3, 1, 1e3, 1e5}

	for _, scale := range scales {
		t.Run(fmt.Sprintf("Scale_%e", scale), func(t *testing.T) {
			baseData := []float64{-1, 1, 4, 5, 6, 6, 8, 12, 20, 36}
			timeSeries := make([]float64, len(baseData))
			for i, v := range baseData {
				timeSeries[i] = v * scale
			}
			query := []float64{3 * scale, 4 * scale, 5 * scale}

			distances, err := MASSV2(timeSeries, query)
			if err != nil {
				t.Fatalf("MASSV2 failed at scale %e: %v", scale, err)
			}

			// Find the minimum distance (should be at index 2)
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

			if minDist > floatToleranceForMASSV2Test {
				t.Errorf("At scale %e: perfect match distance %.12e should be close to 0", scale, minDist)
			}
		})
	}
}

// Performance test to verify O(n log n) complexity
func TestMASSV2_TimeComplexity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping time complexity test in short mode")
	}
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// Test different sizes to verify O(n log n) complexity
	sizes := []int{1000, 2000, 4000, 8000}
	queryLength := 50
	times := make([]time.Duration, len(sizes))

	for i, n := range sizes {
		timeSeries := generateSyntheticData(n, 12345)
		query := generateSyntheticData(queryLength, 54321)

		start := time.Now()
		_, err := MASSV2(timeSeries, query)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("MASSV2 failed for size %d: %v", n, err)
		}

		times[i] = elapsed
		t.Logf("Size %d: %v", n, elapsed)
	}

	// Verify that execution time grows sub-quadratically
	// For O(n log n), doubling input size should roughly double time (plus log factor)
	for i := 1; i < len(sizes); i++ {
		ratio := float64(times[i]) / float64(times[i-1])
		sizeRatio := float64(sizes[i]) / float64(sizes[i-1])
		logRatio := math.Log(float64(sizes[i])) / math.Log(float64(sizes[i-1]))
		expectedRatio := sizeRatio * logRatio

		// Allow for some variance due to system factors
		if ratio > expectedRatio*3 {
			t.Errorf("Time complexity appears worse than O(n log n): size ratio %.1f, time ratio %.1f, expected ~%.1f",
				sizeRatio, ratio, expectedRatio)
		}

		t.Logf("Size ratio: %.1f, Time ratio: %.1f, Expected O(n log n) ratio: %.1f",
			sizeRatio, ratio, expectedRatio)
	}
}

// Property-based test using random data
func TestMASSV2_Properties(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	seed := []uint64{defaultSeed0, defaultSeed1}
	prng := newPRNG(seed...)

	const numTests = 50
	for i := 0; i < numTests; i++ {
		n := prng.IntN(500) + 100 // 100 to 600
		m := prng.IntN(n/2) + 3   // 3 to n/2

		timeSeries := generateSyntheticData(n, seed...)
		query := generateSyntheticData(m, seed[1], seed[0])

		// Ensure query has non-zero variance
		querySigma := stat.StdDev(query, nil)
		if querySigma == 0 {
			continue // Skip this iteration
		}

		distances, err := MASSV2(timeSeries, query)
		if err != nil {
			t.Fatalf("MASSV2 failed on iteration %d (n=%d, m=%d): %v", i, n, m, err)
		}

		// Properties that should always hold:

		// 1. Correct number of distances
		expectedLength := n - m + 1
		if len(distances) != expectedLength {
			t.Errorf("Iteration %d: expected %d distances, got %d", i, expectedLength, len(distances))
		}

		// 2. All distances should be non-negative
		for j, dist := range distances {
			if dist < 0 && !math.IsInf(dist, 1) {
				t.Errorf("Iteration %d: distance[%d] = %f should be non-negative", i, j, dist)
			}
		}

		// 3. Self-match test: insert query into time series and verify perfect match
		if n > 2*m {
			insertPos := m // Insert after first m elements
			testSeries := make([]float64, n)
			copy(testSeries[:insertPos], timeSeries[:insertPos])
			copy(testSeries[insertPos:insertPos+m], query)
			copy(testSeries[insertPos+m:], timeSeries[insertPos+m:])

			selfDistances, err := MASSV2(testSeries, query)
			if err != nil {
				continue // Skip if this fails due to numerical issues
			}

			if len(selfDistances) > insertPos && selfDistances[insertPos] > floatToleranceForMASSV2Test {
				t.Errorf("Iteration %d: self-match distance %f should be close to 0", i, selfDistances[insertPos])
			}
		}
	}
}

func TestFFTConvolutionLinear(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

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
			Name:          "nil signal should return error",
			InputSignal:   []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			InputKernel:   nil,
			ExpectedError: errEmptyFFTConvolutionInputs,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// WHEN (operation under test)
			actualDotProducts, err := fftConvolutionLinear(tc.InputSignal, tc.InputKernel)

			// THEN (assertions)

			if err != nil && tc.ExpectedError == nil {
				t.Fatalf("fftConvolutionLinear failed: %v", err)
			}

			// When an error is expected, assert that we received it
			if tc.ExpectedError != nil {
				if !errors.Is(err, tc.ExpectedError) {
					t.Errorf("expected error %v, got %v", tc.ExpectedError, err)
				}

				// Since we expected an error, do not evaluate the dot
				// products.
				return
			}

			if len(actualDotProducts) != len(tc.ExpectedDotProducts) {
				t.Fatalf("expected returned slice to have length %d, got length %d", len(tc.ExpectedDotProducts), len(actualDotProducts))
			}

			for i, expected := range tc.ExpectedDotProducts {
				actual := actualDotProducts[i]
				if !almostEqual(actual, expected, floatTolerance) {
					t.Errorf("expected returned slice index %d to be %f, got %f", i, expected, actual)
				}
			}
		})
	}
}

// Test helper functions
func TestSlidingMeanStddev(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	type TestCase struct {
		Name            string
		InputData       []float64
		InputWindowSize int
		ExpectedMeans   []float64
		ExpectedSigmas  []float64
	}

	testCases := []TestCase{
		{
			Name:            "basic functionality",
			InputData:       []float64{1, 2, 4, 8, 16},
			InputWindowSize: 3,
			ExpectedMeans:   []float64{2.3333333333, 4.6666666667, 9.3333333333}, // [1,2,4], [2,4,8], [4,8,16]
			ExpectedSigmas:  []float64{1.2472191289, 2.4944382578, 4.9888765157}, // [1,2,4], [2,4,8], [4,8,16]
		},
		{
			Name:            "empty or nil values when windowSize > len(data)",
			InputData:       []float64{1, 2, 4, 8, 16, 32},
			InputWindowSize: 7, // len(data) + 1
			ExpectedMeans:   nil,
			ExpectedSigmas:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// WHEN (operation under test)
			actualMeans, actualSigmas := slidingMeanStddev(tc.InputData, tc.InputWindowSize)

			// THEN (assertions

			if len(actualMeans) != len(tc.ExpectedMeans) {
				t.Errorf("expected %d sliding means, got %d", len(tc.ExpectedMeans), len(actualMeans))
			}
			for i, expected := range tc.ExpectedMeans {
				if !almostEqual(actualMeans[i], expected, floatTolerance) {
					t.Errorf("expected sliding mean at %d to be %f, got %f", i, expected, actualMeans[i])
				}
			}

			if len(actualSigmas) != len(tc.ExpectedSigmas) {
				t.Errorf("expected %d sliding means, got %d", len(tc.ExpectedSigmas), len(actualSigmas))
			}
			for i, expected := range tc.ExpectedSigmas {
				if !almostEqual(actualSigmas[i], expected, floatTolerance) {
					t.Errorf("expected sliding mean at %d to be %f, got %f", i, expected, actualSigmas[i])
				}
			}
		})
	}

}

// `go test -short ./...` to skip this test
func TestSlidingMeanStddev_FloatingPointAccumulationError(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping floating point accumulation error test in short mode")
	}
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	// GIVEN (set up)

	const (
		m = 60 * 24 // 1 day of observations @ 1-minute intervals

		// Generate a time-series with ~5.27 million values, which is enough to
		// hold 10 years of observations taken every minute.
		n = 60 * 24 * 366 * 10
	)

	// Create time-series
	seed := []uint64{defaultSeed0, defaultSeed1}
	prng := newPRNG(seed...)
	ts := generateNoisySineWave(n, 97, 100, 0, seed...)

	// Pick a subsequence of the time-series for the query
	queryIdx := prng.IntN(n - m)
	query := make([]float64, m)
	copy(query, ts[queryIdx:queryIdx+m])

	expectedMeans, expectedSigmas := naiveSlidingMeanStddev(ts, m)

	// WHEN (operation under test)

	actualMeans, actualSigmas := slidingMeanStddev(ts, m)

	// THEN (assertions)

	if len(actualMeans) != len(expectedMeans) {
		t.Fatalf("expected sliding means to have length %d, got %d", len(expectedMeans), len(actualMeans))
	}
	if len(actualSigmas) != len(expectedSigmas) {
		t.Fatalf("expected sliding standard deviations to have length %d, got %d", len(expectedSigmas), len(actualSigmas))
	}

	lastIndex := len(actualMeans) - 1

	lastActualMean := actualMeans[lastIndex]
	lastExpectedMean := expectedMeans[lastIndex]
	if !almostEqual(lastActualMean, lastExpectedMean, floatToleranceForMASSV2Test) {
		t.Errorf("last actual mean and the last expected mean differ by excessive margin: %e [seed: %v]", math.Abs(lastActualMean-lastExpectedMean), seed)
	}

	lastActualStddev := actualSigmas[lastIndex]
	lastExpectedStddev := expectedSigmas[lastIndex]
	if !almostEqual(lastActualStddev, lastExpectedStddev, floatToleranceForMASSV2Test) {
		t.Errorf("last actual standard deviation and the last expected standard deviation differ by excessive margin: %e [seed: %v]", math.Abs(lastActualStddev-lastExpectedStddev), seed)
	}
}

////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS

// almostEqual is a helper function to check if two float64 values are equal,
// allowing for a little tolerance due to floating point accumulation errors.
func almostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

// generateNoisySineWave is a helper function to generate sinusoidal data with
// noise up to +-10% of amplitude.
func generateNoisySineWave(n int, frequency, amplitude, phase float64, seed ...uint64) []float64 {
	noiseMax := math.Abs(amplitude) * 0.1
	noiseStddev := noiseMax * 0.333

	prng := newPRNG(seed...)
	data := make([]float64, n)
	for i := range data {
		data[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)/float64(n)+phase)

		noise := prng.NormFloat64() * noiseStddev
		if math.Abs(noise) <= noiseMax {
			data[i] += noise
		}
	}

	return data
}

// generateSineWave is a helper function to generate sinusoidal data.
func generateSineWave(n int, frequency, amplitude, phase float64) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)/float64(n)+phase)
	}
	return data
}

// generateSyntheticData is a helper function to generate synthetic time-series
// data.
func generateSyntheticData(n int, seed ...uint64) []float64 {
	prng := newPRNG(seed...)
	data := make([]float64, n)
	for i := range data {
		data[i] = prng.NormFloat64()
	}
	return data
}

// naiveSlidingMeanStddev computes the mean and standard deviation of every
// sliding window in data size windowSize by recomputing the entire content
// of each window. This approach avoids floating-point accumulation error.
//
// This function exists for testing the output of slidingMeanStddev and
// fine-tuning its algorithm to maximize efficiency while minimizing
// computational overhead.
func naiveSlidingMeanStddev(data []float64, windowSize int) (means, sigmas []float64) {
	n := len(data)
	if windowSize > n {
		return nil, nil
	}

	means = make([]float64, n-windowSize+1)
	sigmas = make([]float64, n-windowSize+1)

	// Slide the window and recompute the mean and standard deviation from
	// scratch for each window.
	for i := 0; i <= n-windowSize; i++ {
		means[i], sigmas[i] = stat.PopMeanStdDev(data[i:i+windowSize], nil)
	}

	return means, sigmas
}

func newPRNG(seed ...uint64) *rand.Rand {
	seed0 := defaultSeed0
	seed1 := defaultSeed1
	if len(seed) > 0 {
		seed0 = seed[0]
		if len(seed) > 1 {
			seed1 = seed[1]
		}
	}

	src := rand.NewPCG(seed0, seed1)
	return rand.New(src)
}

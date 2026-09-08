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

import "math"

const (
	// compensatedSumErrorFactor bounds the absolute error of compensatedSum
	// as a multiple of the sum of the input magnitudes. Neumaier's summation
	// has error at most (2u + O(n·u²))·Σ|xᵢ|; 2ε = 4u covers the second-order
	// term for any length this package can allocate.
	compensatedSumErrorFactor = 2 * float64Epsilon

	// twoPassSumErrorFactor bounds the relative error of a compensated sum of
	// squared deviations, excluding the effect of an inexact mean: each term
	// (x − mean)² rounds within 3u, and the compensated sum of nonnegative
	// terms adds 2u, for 5u < 3ε.
	twoPassSumErrorFactor = 3 * float64Epsilon
)

// windowStatistics holds the population mean and sum of squared deviations of
// one window together with forward error bounds on both. meanError bounds
// |mean − exact mean| and sumError bounds |sum − exact sum| in absolute
// terms, for the exact statistics of the same float64 values.
type windowStatistics struct {
	mean                   float64
	sumOfSquaredDeviations float64
	meanError              float64
	sumError               float64
}

// centerAndScale maps values through the affine transformation
// x ↦ (x − anchor)/scale − mean, choosing the anchor and scale so that every
// prepared value lies in [−2, 2] and no intermediate overflows. Z-normalized
// distance is invariant under exact affine maps, so the prepared data has the
// same distance profile as the input up to preparationErrorPerValue of
// rounding in each value.
//
// The anchor is the first value and the scale is the largest absolute
// difference from it, which keeps every subtraction exact for values within a
// factor of two of the anchor and bounds every other subtraction's error by
// the scale. When opposite-sign extremes make a difference overflow, the
// values are instead divided by the largest input magnitude, which cannot
// overflow and, because such inputs span the float64 range, does not discard
// variation around a common offset. The direct distance path prepares each
// window on its own, so the anchor choice affects only which windows the
// FFT-path bound can certify, never the returned distances.
//
// values must be nonempty and finite. constant reports that all values are
// equal, in which case prepared is all zeros.
func centerAndScale(values []float64) (prepared []float64, constant bool) {
	prepared = make([]float64, len(values))
	constant = centerAndScaleInto(values, prepared)
	return prepared, constant
}

// centerAndScaleInto performs centerAndScale using caller-owned storage.
// prepared must have the same length as values and must not overlap values.
// Every element is overwritten, including when the input is constant.
func centerAndScaleInto(values, prepared []float64) (constant bool) {
	// Translate by the anchor and find the scale.
	anchor := values[0]
	var maximumDifference float64
	for index, value := range values {
		difference := value - anchor
		prepared[index] = difference
		maximumDifference = math.Max(maximumDifference, math.Abs(difference))
	}

	// Fall back to magnitude scaling when translation overflowed.
	if math.IsInf(maximumDifference, 0) {
		inputMagnitude := maximumMagnitude(values)
		for index, value := range values {
			prepared[index] = value / inputMagnitude
		}
		maximumDifference = 1
	}
	if maximumDifference == 0 {
		return true
	}

	// Scale, then center on the compensated mean.
	for index := range prepared {
		prepared[index] /= maximumDifference
	}
	mean := compensatedSum(prepared) / float64(len(prepared))
	for index := range prepared {
		prepared[index] -= mean
	}
	return false
}

// constantWindows identifies exactly equal windows in O(len(values)) time.
// windowSize must be positive and no larger than len(values). Equality treats
// signed zeros alike; no variance estimate or small-difference threshold is used.
func constantWindows(values []float64, windowSize int) []bool {
	constant := make([]bool, len(values)-windowSize+1)
	runLength := 0
	for index, value := range values {
		if index == 0 || value != values[index-1] {
			runLength = 1
		} else {
			runLength++
		}
		if index >= windowSize-1 {
			constant[index-windowSize+1] = runLength >= windowSize
		}
	}
	return constant
}

// maximumMagnitude returns the largest absolute value in values, or zero for
// an empty slice.
func maximumMagnitude(values []float64) float64 {
	var maximum float64
	for _, value := range values {
		maximum = math.Max(maximum, math.Abs(value))
	}
	return maximum
}

// computeWindowStatistics returns the two-pass population mean and sum of
// squared deviations of values with forward error bounds. Both sums are
// compensated, so the bounds stay near a few epsilon of the result rather
// than growing with the window length. values must be nonempty and finite.
func computeWindowStatistics(values []float64) windowStatistics {
	count := float64(len(values))

	// First pass: compensated mean and its bound.
	var sumOfMagnitudes float64
	for _, value := range values {
		sumOfMagnitudes += math.Abs(value)
	}
	mean := compensatedSum(values) / count
	meanError := compensatedSumErrorFactor*sumOfMagnitudes/count + float64Epsilon*math.Abs(mean)

	// Second pass: compensated sum of squared deviations. Its exact value
	// exceeds the exact sum about the true mean by count·(mean error)², so
	// that term joins the rounding bound.
	var accumulator neumaierAccumulator
	for _, value := range values {
		deviation := value - mean
		accumulator.add(deviation * deviation)
	}
	sumOfSquaredDeviations := accumulator.total()
	sumError := twoPassSumErrorFactor*sumOfSquaredDeviations + count*meanError*meanError

	return windowStatistics{
		mean:                   mean,
		sumOfSquaredDeviations: sumOfSquaredDeviations,
		meanError:              meanError,
		sumError:               sumError,
	}
}

// compensatedSum returns the sum of values using Neumaier's compensated
// summation. The result is within compensatedSumErrorFactor times the sum of
// the input magnitudes of the exact sum.
func compensatedSum(values []float64) float64 {
	var accumulator neumaierAccumulator
	for _, value := range values {
		accumulator.add(value)
	}
	return accumulator.total()
}

// neumaierAccumulator accumulates a running sum together with the rounding
// error lost by each addition, in the manner of Neumaier's improvement to
// Kahan summation. It is a value type for use within one function.
type neumaierAccumulator struct {
	sum          float64
	compensation float64
}

// add folds value into the running sum, capturing the low-order bits that
// the addition rounds away whichever operand is larger.
func (accumulator *neumaierAccumulator) add(value float64) {
	updated := accumulator.sum + value
	if math.Abs(accumulator.sum) >= math.Abs(value) {
		accumulator.compensation += (accumulator.sum - updated) + value
	} else {
		accumulator.compensation += (value - updated) + accumulator.sum
	}
	accumulator.sum = updated
}

// total returns the compensated sum.
func (accumulator neumaierAccumulator) total() float64 {
	return accumulator.sum + accumulator.compensation
}

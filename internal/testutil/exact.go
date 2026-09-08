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
	"math/big"
)

// exactPrecision is the mantissa width, in bits, used by the big.Float
// references. It leaves more than 60 decimal digits of headroom over float64.
const exactPrecision = 256

// ExactMeanStddev returns the population mean and standard deviation of
// values in 256-bit arithmetic. values must be nonempty.
func ExactMeanStddev(values []float64) (mean, sigma *big.Float) {
	mean = newExact(0)
	for _, value := range values {
		mean.Add(mean, newExact(value))
	}
	mean.Quo(mean, newExact(float64(len(values))))

	sumOfSquaredDeviations := newExact(0)
	deviation := newExact(0)
	for _, value := range values {
		deviation.Sub(newExact(value), mean)
		deviation.Mul(deviation, deviation)
		sumOfSquaredDeviations.Add(sumOfSquaredDeviations, deviation)
	}
	variance := sumOfSquaredDeviations.Quo(sumOfSquaredDeviations, newExact(float64(len(values))))
	return mean, newExact(0).Sqrt(variance)
}

// ExactSlidingMeanStddev computes every window's population mean and standard
// deviation in 256-bit arithmetic, so its results are correct to far more
// digits than float64 can represent. It is intended for modest inputs and
// returns nil slices when windowSize exceeds len(data).
func ExactSlidingMeanStddev(data []float64, windowSize int) (means, sigmas []float64) {
	n := len(data)
	if windowSize > n {
		return nil, nil
	}

	means = make([]float64, n-windowSize+1)
	sigmas = make([]float64, n-windowSize+1)
	for i := range means {
		mean, sigma := ExactMeanStddev(data[i : i+windowSize])
		means[i], _ = mean.Float64()
		sigmas[i], _ = sigma.Float64()
	}
	return means, sigmas
}

// ExactDotProducts returns every sliding dot product of the query against the
// signal in 256-bit arithmetic, in the same layout as the FFT convolution's
// output for the reversed query: index m−1+i holds the dot product at window
// i, where m is the kernel length.
func ExactDotProducts(signal, reversedKernel []float64) []float64 {
	n := len(signal)
	m := len(reversedKernel)
	out := make([]float64, n+m-1)
	sum := newExact(0)
	term := newExact(0)
	for k := range out {
		sum.SetFloat64(0)
		for j := range m {
			signalIndex := k - j
			if signalIndex < 0 || signalIndex >= n {
				continue
			}
			term.Mul(newExact(signal[signalIndex]), newExact(reversedKernel[j]))
			sum.Add(sum, term)
		}
		out[k], _ = sum.Float64()
	}
	return out
}

// ExactDistanceProfile computes every z-normalized Euclidean distance in
// 256-bit arithmetic. It cannot overflow or underflow on any finite float64
// input, so it verifies fixtures near the extremes of the float64 range that a
// float64 oracle cannot. Constant windows receive positive infinity. The query
// must not be longer than the series.
func ExactDistanceProfile(timeSeries, query []float64) []float64 {
	normalizedQuery, queryIsConstant := exactNormalize(query)
	distances := make([]float64, len(timeSeries)-len(query)+1)
	difference := newExact(0)
	for index := range distances {
		normalizedWindow, windowIsConstant := exactNormalize(timeSeries[index : index+len(query)])
		if queryIsConstant || windowIsConstant {
			distances[index] = math.Inf(1)
			continue
		}

		distanceSquared := newExact(0)
		for valueIndex, windowValue := range normalizedWindow {
			difference.Sub(windowValue, normalizedQuery[valueIndex])
			difference.Mul(difference, difference)
			distanceSquared.Add(distanceSquared, difference)
		}
		distances[index], _ = newExact(0).Sqrt(distanceSquared).Float64()
	}
	return distances
}

// exactNormalize z-normalizes values in 256-bit arithmetic. It reports
// constant when the values have no variation, in which case normalized is nil.
func exactNormalize(values []float64) (normalized []*big.Float, constant bool) {
	mean, sigma := ExactMeanStddev(values)
	if sigma.Sign() == 0 {
		return nil, true
	}
	normalized = make([]*big.Float, len(values))
	for index, value := range values {
		normalized[index] = newExact(value)
		normalized[index].Sub(normalized[index], mean)
		normalized[index].Quo(normalized[index], sigma)
	}
	return normalized, false
}

// newExact returns a 256-bit big.Float holding value.
func newExact(value float64) *big.Float {
	return new(big.Float).SetPrec(exactPrecision).SetFloat64(value)
}

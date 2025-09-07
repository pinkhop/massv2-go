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
// Citation: Abdullah Mueen, Sheng Zhong, Yan Zhu, Michael Yeh, Kaveh Kamgar,
// Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2022), The
// Fastest Similarity Search Algorithm for Time Series Subsequences under
// Euclidean Distance, URL:
// http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
package massv2

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
	"gonum.org/v1/gonum/stat"
)

const floatTolerance = 1e-6

var (
	ErrEmptyQuery                = errors.New("empty or nil query")
	ErrEmptyTimeSeries           = errors.New("empty or nil time-series")
	ErrQueryHasZeroVariance      = errors.New("query has zero variance (all values are the same)")
	ErrQueryLongerThanTimeSeries = errors.New("query length exceeds time-series length")

	errEmptyFFTConvolutionInputs = errors.New("empty or nil inputs for FFT convolution")
)

// MASSV2 computes and returns the z-normalized Euclidean distance between the
// query sequence and every subsequence of the same length in the time-series.
// The algorithm operates in O(n log n) time with a space complexity of O(n).
//
// The time-series and query sequence must not be empty or nil. The query
// cannot have zero variance (constant values) or be longer than the
// time-series. An error is returned when the inputs are invalid.
func MASSV2(timeSeries, query []float64) (distances []float64, err error) {
	n := len(timeSeries)
	m := len(query)

	// Guard statements
	if m <= 0 {
		return nil, ErrEmptyQuery
	} else if n <= 0 {
		return nil, ErrEmptyTimeSeries
	} else if m > n {
		return nil, ErrQueryLongerThanTimeSeries
	}

	// Compute query statistics
	queryMean, querySigma := stat.PopMeanStdDev(query, nil)
	if querySigma < floatTolerance { // values very close to 0 are treated like 0
		return nil, ErrQueryHasZeroVariance
	}

	// Compute sliding window statistics for the time-series
	timeSeriesMeans, timeSeriesSigmas := slidingMeanStddev(timeSeries, m)

	// Prepare the query for convolution: reverse the query
	reversedQuery := make([]float64, m)
	for i := 0; i < m; i++ {
		reversedQuery[i] = query[m-1-i]
	}

	// Compute the dot products using linear FFT convolution. We know that
	// timeSeries and reversedQuery cannot be empty due to the checks we
	// performed above, so we do not need to check the error.
	dotProducts, _ := fftConvolutionLinear(timeSeries, reversedQuery)

	// Compute z-normalized Euclidean distances
	distances = make([]float64, n-m+1)
	for i := 0; i < len(distances); i++ {
		if timeSeriesSigmas[i] == 0 {
			distances[i] = math.Inf(1) // infinite distance for zero variance subsequences
			continue
		}

		// Apply z-normalized Euclidean distance formula.
		//
		// In the reference MATLAB implementation (which uses 1-based arrays),
		// the computation is:
		//     dist = 2*(m-(z(m:n)-m*meanx(m:n)*meany)./(sigmax(m:n)*sigmay));
		//     dist = sqrt(dist);
		normalizedDot := (dotProducts[m+i-1] - float64(m)*timeSeriesMeans[i]*queryMean) / (timeSeriesSigmas[i] * querySigma) // = m * rho
		distSquared := 2.0 * (float64(m) - normalizedDot)
		if distSquared < 0 {
			distSquared = 0
		}
		distances[i] = math.Sqrt(distSquared)
	}

	return distances, nil
}

// fftConvolutionLinear performs linear convolution of 'signal' (len n)
// with 'kernel' (effective len m), using FFT zero-padding. Returns an error
// when signal or kernel are empty or nil.
func fftConvolutionLinear(signal, kernel []float64) ([]float64, error) {
	n := len(signal)
	m := len(kernel)
	if n == 0 || m == 0 {
		return nil, errEmptyFFTConvolutionInputs
	}

	convLen := nextPow2(n + m - 1)

	fft := fourier.NewCmplxFFT(convLen)

	a := make([]complex128, convLen)
	b := make([]complex128, convLen)

	for i := 0; i < n; i++ {
		a[i] = complex(signal[i], 0)
	}
	for i := 0; i < m; i++ {
		b[i] = complex(kernel[i], 0)
	}

	A := fft.Coefficients(nil, a)
	B := fft.Coefficients(nil, b)

	for i := 0; i < convLen; i++ {
		A[i] *= B[i]
	}

	c := fft.Sequence(nil, A)

	out := make([]float64, n+m-1)
	scale := float64(convLen) // gonum FFT is unnormalized
	for i := 0; i < len(out); i++ {
		out[i] = real(c[i]) / scale
	}
	return out, nil
}

// nextPow2 returns the smallest power of two >= x
func nextPow2(x int) int {
	p := 1
	for p < x {
		p <<= 1
	}
	return p
}

// slidingMeanStddev computes the mean and standard deviation of every sliding
// window in data of size windowSize.
func slidingMeanStddev(data []float64, windowSize int) (means, sigmas []float64) {
	n := len(data)
	if windowSize > n {
		return nil, nil
	}

	means = make([]float64, n-windowSize+1)
	sigmas = make([]float64, n-windowSize+1)
	windowSizeF64 := float64(windowSize)

	// Initialize the first window
	var sum, sumOfSquares float64
	for i := 0; i < windowSize; i++ {
		sum += data[i]
		sumOfSquares += data[i] * data[i]
	}
	means[0] = sum / windowSizeF64
	variance := sumOfSquares/windowSizeF64 - means[0]*means[0]
	if variance < 0 { // handle floating point imprecision, prevent sqrt of negative value
		variance = 0
	}
	sigmas[0] = math.Sqrt(variance)

	// Slide the window and update mean and standard deviations incrementally
	for i := 1; i <= n-windowSize; i++ {
		// NOTE: Testing finds that there is not significant floating point
		// accumulation error due to the incremental updates to sum and
		// sumOfSquares for the lengths of time-series and queries an order of
		// magnitude greater than I am interested in evaluating, so this
		// algorithm does not attempt to limit floating point accumulation
		// error. If this is determined to be an issue at a future point, an
		// easy solution would be to fully recompute sum and sumOfSquares for
		// one out of every 100,000 (or 1 million, or whatever); e.g.,
		// if i%100_000 == 0.

		// Remove the element leaving the window
		oldValue := data[i-1]
		sum -= oldValue
		sumOfSquares -= oldValue * oldValue

		// Add the element entering the window
		newValue := data[i+windowSize-1]
		sum += newValue
		sumOfSquares += newValue * newValue

		// Compute the mean and standard deviation for the updated window
		means[i] = sum / windowSizeF64
		variance = sumOfSquares/windowSizeF64 - means[i]*means[i]
		if variance < 0 { // handle floating point imprecision, prevent sqrt of negative value
			variance = 0
		}
		sigmas[i] = math.Sqrt(variance)
	}

	return means, sigmas
}

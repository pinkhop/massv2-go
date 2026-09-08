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

	"gonum.org/v1/gonum/dsp/fourier"
)

// errEmptyFFTConvolutionInputs is returned by fftConvolutionLinear when
// either input is empty.
var errEmptyFFTConvolutionInputs = errors.New("empty or nil inputs for FFT convolution")

// fftConvolutionLinear returns the linear convolution of signal (length n)
// with kernel (length m) through zero-padded FFTs, together with a bound on
// the absolute rounding error of every returned element.
//
// The bound follows Higham's forward error analysis of the FFT (Accuracy and
// Stability of Numerical Algorithms, 2nd ed., Theorem 24.2): each transform
// of length N perturbs its result by at most θ = fftPerLevelRelativeError ·
// log₂N in relative 2-norm. The computed spectral product differs from the
// exact one by eA∘B̂ + A∘eB plus the product rounding, where eA and eB are
// the transform errors. Each output element is an inverse-transform sum of
// such a product divided by N, and the Cauchy–Schwarz inequality bounds it
// by the product of the factors' 2-norms, which Parseval's identity relates
// to the input norms. The inverse transform's own error is at most θ times
// the 2-norm of its input, the computed product spectrum P̂. Together,
//
//	2·(θ + ε)·(1 + θ)²·‖a‖₂·‖b‖₂ + (θ + ε)·‖P̂‖₂/√N
//
// bounds the absolute error of every output element. The first term is the
// classical FFT convolution bound; the second uses the computed product
// spectrum, whose norm equals that of the convolution itself, so neither a
// spike nor a trend in the signal inflates the bound beyond its real
// contribution to the dot products.
//
// Inputs must be finite and small enough that their sums of squares do not
// overflow; prepared data from centerAndScale satisfies this. Returns
// errEmptyFFTConvolutionInputs when either input is empty.
func fftConvolutionLinear(signal, kernel []float64) (out []float64, roundoffBound float64, err error) {
	n := len(signal)
	m := len(kernel)
	if n == 0 || m == 0 {
		return nil, 0, errEmptyFFTConvolutionInputs
	}

	// Pad both inputs to a power of two and transform in place.
	convLen := nextPow2(n + m - 1)
	fft := fourier.NewCmplxFFT(convLen)
	signalSpectrum := make([]complex128, convLen)
	kernelSpectrum := make([]complex128, convLen)
	for i := range n {
		signalSpectrum[i] = complex(signal[i], 0)
	}
	for i := range m {
		kernelSpectrum[i] = complex(kernel[i], 0)
	}
	signalSpectrum = fft.Coefficients(signalSpectrum, signalSpectrum)
	kernelSpectrum = fft.Coefficients(kernelSpectrum, kernelSpectrum)

	// Multiply in place, recording the product spectrum's norm for the bound,
	// then invert in place and normalize the unnormalized inverse.
	var productSquaredNorm float64
	for i := range convLen {
		signalSpectrum[i] *= kernelSpectrum[i]
		productSquaredNorm += real(signalSpectrum[i])*real(signalSpectrum[i]) + imag(signalSpectrum[i])*imag(signalSpectrum[i])
	}
	product := fft.Sequence(signalSpectrum, signalSpectrum)
	out = make([]float64, n+m-1)
	scale := float64(convLen)
	for i := range out {
		out[i] = real(product[i]) / scale
	}

	roundoffBound = fftConvolutionRoundoffBound(
		convLen,
		euclideanNorm(signal),
		euclideanNorm(kernel),
		math.Sqrt(productSquaredNorm),
	)
	return out, roundoffBound, nil
}

// fftConvolutionRoundoffBound evaluates the bound documented on
// fftConvolutionLinear for a transform of length fftLength, given the input
// 2-norms and the 2-norm of the computed product spectrum.
func fftConvolutionRoundoffBound(fftLength int, signalNorm, kernelNorm, productSpectrumNorm float64) float64 {
	transformRelativeError := fftPerLevelRelativeError * math.Log2(float64(fftLength))
	perTransform := transformRelativeError + float64Epsilon
	growth := 1 + transformRelativeError
	return 2*perTransform*growth*growth*signalNorm*kernelNorm +
		perTransform*productSpectrumNorm/math.Sqrt(float64(fftLength))
}

// euclideanNorm returns the 2-norm of values without overflow protection,
// which is adequate for prepared data of magnitude at most two.
func euclideanNorm(values []float64) float64 {
	var sumOfSquares float64
	for _, value := range values {
		sumOfSquares += value * value
	}
	return math.Sqrt(sumOfSquares)
}

// nextPow2 returns the smallest power of two that is at least x, or one for
// nonpositive x.
func nextPow2(x int) int {
	p := 1
	for p < x {
		p <<= 1
	}
	return p
}

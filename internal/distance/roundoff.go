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

// Every roundoff bound in this package is a forward error bound expressed in
// multiples of float64Epsilon. IEEE 754 double arithmetic rounds each
// operation to within a unit roundoff u = float64Epsilon / 2, so counting
// each operation as a full epsilon leaves a factor of two of slack that
// absorbs the second-order terms that a first-order analysis drops.
const (
	// float64Epsilon is the machine epsilon of IEEE 754 double precision: the
	// spacing between 1 and the next representable value.
	float64Epsilon = 0x1p-52

	// DistanceAccuracyTarget is the absolute accuracy target, in z-normalized
	// distance units, used to accept FFT-path entries. A window whose bound
	// cannot certify the target is recalculated directly, with the accuracy
	// limitations described on directZNormalizedDistance.
	DistanceAccuracyTarget = 1e-7

	// fftPerLevelRelativeError is the relative 2-norm error contributed by
	// each radix-2 level of an FFT, the constant η of Higham, Accuracy and
	// Stability of Numerical Algorithms, 2nd ed., Theorem 24.2. There
	// η = μ + γ₄(√2 + μ), where μ bounds the absolute error of the twiddle
	// factors. gonum's FFTPACK port computes each twiddle angle as an integer
	// multiple of a rounded 2π/N, so angles near 2π carry an absolute error
	// of roughly 20u and μ ≈ 20u; with γ₄(√2 + μ) ≈ 6u, η ≈ 26u = 13ε. The
	// port also runs power-of-two lengths as radix-4 passes, whose butterflies
	// round more values per level than the radix-2 model assumes. 16ε covers
	// both effects, and the convolution roundoff test checks the resulting
	// bound against exact dot products.
	fftPerLevelRelativeError = 16 * float64Epsilon

	// preparationErrorPerValue bounds the absolute rounding error of each
	// prepared value relative to the exact affine image of the input.
	// Preparation subtracts the anchor (error ≤ u per unit of result), divides
	// by the scale (u), and subtracts the mean (u on a value of magnitude at
	// most 2), for at most 4u = 2ε on the prepared scale.
	preparationErrorPerValue = 2 * float64Epsilon

	// statisticsRebuildRelativeError is the relative error bound on a sliding
	// window's sum of squared deviations above which the recurrence is
	// discarded and the window recomputed from scratch. Sigma errors are
	// amplified by the window length in the FFT distance formula, so this is
	// far tighter than the distance target itself.
	statisticsRebuildRelativeError = 1e-10
)

// fftPathErrorBounds collects the error bounds for one window's FFT-path
// distance. distanceSquared bounds the difference between the calculated
// squared distance and the exact squared distance of the prepared data.
// preparation bounds the difference, in distance units, between the exact
// distance of the prepared data and the exact distance of the original data.
type fftPathErrorBounds struct {
	distanceSquared float64
	preparation     float64
}

// fftPathContext holds the per-call quantities that every FFT-path window
// shares: the window length, the convolution's roundoff bound, and the
// prepared query's statistics with their error bounds.
type fftPathContext struct {
	windowSize              float64
	dotProductRoundoff      float64
	queryMean               float64
	queryMeanError          float64
	querySigma              float64
	querySigmaRelativeError float64
}

// distanceSquared evaluates the MASS distance formula for one window and
// returns the squared distance together with the bounds needed to decide
// whether the result meets the accuracy target.
//
// dotProduct is the FFT dot product of the prepared window with the prepared
// query. mean, meanError, sigma, and sigmaRelativeError describe the prepared
// window; sigma must be positive.
//
// The formula is the reference implementation's
// 2·(m − (z − m·meanx·meany) / (sigmax·sigmay)), with z the sliding dot
// product. The mean-product term is retained because the prepared query's
// stored mean is not exactly zero, and dropping it would leave an error of
// order m²·ε in the dot product.
func (context fftPathContext) distanceSquared(
	dotProduct, mean, meanError, sigma, sigmaRelativeError float64,
) (distanceSquared float64, bounds fftPathErrorBounds) {
	meanProduct := context.windowSize * mean * context.queryMean
	numerator := dotProduct - meanProduct
	sigmaProduct := sigma * context.querySigma
	normalizedDot := numerator / sigmaProduct
	distanceSquared = 2 * (context.windowSize - normalizedDot)

	// Numerator: convolution roundoff, propagated mean errors, and the
	// rounding of two products and one subtraction.
	numeratorError := context.dotProductRoundoff +
		context.windowSize*(math.Abs(mean)*context.queryMeanError+math.Abs(context.queryMean)*meanError) +
		2*float64Epsilon*(math.Abs(dotProduct)+math.Abs(meanProduct))

	// Normalized dot product: numerator error scaled by the sigma product,
	// plus the relative sigma errors and the rounding of a product and a
	// quotient.
	normalizedDotError := numeratorError/sigmaProduct +
		math.Abs(normalizedDot)*(sigmaRelativeError+context.querySigmaRelativeError+2*float64Epsilon)

	bounds.distanceSquared = 2*normalizedDotError + float64Epsilon*math.Abs(distanceSquared)
	bounds.preparation = preparationDistanceError(context.windowSize, sigma, context.querySigma)
	return distanceSquared, bounds
}

// preparationDistanceError bounds how far the exact z-normalized distance of
// the prepared window and query can lie from that of the original data.
//
// Perturbing a vector by δ changes its z-normalized form by at most
// 2‖δ‖₂/σ, and ‖δ‖₂ ≤ preparationErrorPerValue·√m for each of the two
// vectors, so the distance moves by at most the sum of the two terms.
func preparationDistanceError(windowSize, sigma, querySigma float64) float64 {
	return 2 * preparationErrorPerValue * math.Sqrt(windowSize) * (1/sigma + 1/querySigma)
}

// fftDistanceRequiresDirectVerification reports whether the FFT-path squared
// distance cannot be certified to within DistanceAccuracyTarget, so the window
// must be recalculated directly.
//
// A nonpositive or nonfinite squared distance is never certified: the
// formula 2·(m − m·ρ) cancels catastrophically as ρ approaches one, so exact
// and near-exact matches are always recalculated directly. Otherwise the
// preparation error is subtracted from the target, leaving t, and the squared
// bound B must satisfy |d̂ − d| ≤ t whenever |d̂² − d²| ≤ B. For d̂ ≥ t that
// requires B ≤ 2·d̂·t − t², and for d̂ < t the lower side is automatic and
// B ≤ 2·d̂·t + t² suffices.
func fftDistanceRequiresDirectVerification(distanceSquared float64, bounds fftPathErrorBounds) bool {
	if math.IsNaN(distanceSquared) || math.IsInf(distanceSquared, 0) || distanceSquared <= 0 {
		return true
	}
	remainingTarget := DistanceAccuracyTarget - bounds.preparation
	if !(remainingTarget > 0) {
		return true
	}

	distance := math.Sqrt(distanceSquared)
	acceptableSquaredError := 2*distance*remainingTarget + remainingTarget*remainingTarget
	if distance >= remainingTarget {
		acceptableSquaredError = 2*distance*remainingTarget - remainingTarget*remainingTarget
	}
	// A NaN bound compares false and therefore requires verification.
	return !(bounds.distanceSquared <= acceptableSquaredError)
}

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

// slidingStatistics holds the population mean and standard deviation of every
// window of a series together with error bounds. meanErrors[i] bounds the
// absolute error of means[i]; sigmaRelativeErrors[i] bounds the relative
// error of sigmas[i]. A constant window has sigma zero and no meaningful
// relative bound, which is recorded as zero. All four slices have one entry
// per window.
type slidingStatistics struct {
	means               []float64
	sigmas              []float64
	meanErrors          []float64
	sigmaRelativeErrors []float64
}

// slidingWindowStatistics computes the population mean and standard deviation
// of every window of windowSize consecutive values in data, with forward error
// bounds that the FFT distance path uses to decide whether a window's result
// can be certified.
//
// Exactly constant windows use their common value and zero variance in O(1)
// work per window after an O(n) equality scan. The first nonconstant window,
// including after a constant run, uses compensated two-pass statistics. Each
// later nonconstant window updates the previous one with the replacement form
// of the centered second moment, tracking a running bound on the accumulated
// rounding error from the actual magnitudes involved. When that bound exceeds
// statisticsRebuildRelativeError of the sum and a fresh two-pass calculation
// would be tighter, the window is recomputed from scratch, which is what
// prevents a departing outlier's rounding residue from contaminating later
// windows. On data of ordinary conditioning the bound grows by about one
// epsilon per window, so rebuilds are rare and total work is O(n). Data whose
// local variation is tiny relative to its magnitudes can rebuild often, up to
// O(n·windowSize) in the worst case; such windows are also the ones the FFT
// path cannot certify.
//
// data must be finite. The bounds are valid for any finite magnitudes but are
// only small for data whose values are O(1), such as the output of
// centerAndScale; raw data with a large common offset yields large bounds.
// Returns zero-value statistics when windowSize is not positive or exceeds
// len(data).
func slidingWindowStatistics(data []float64, windowSize int) slidingStatistics {
	n := len(data)
	if windowSize <= 0 || windowSize > n {
		return slidingStatistics{}
	}

	windowCount := n - windowSize + 1
	windowSizeF64 := float64(windowSize)
	result := slidingStatistics{
		means:               make([]float64, windowCount),
		sigmas:              make([]float64, windowCount),
		meanErrors:          make([]float64, windowCount),
		sigmaRelativeErrors: make([]float64, windowCount),
	}

	// Equality here concerns the prepared data's statistics only; it must not
	// be used to classify original candidate windows as constant distances.
	constant := constantWindows(data, windowSize)
	var state windowStatistics
	for i := range windowCount {
		if constant[i] {
			state = windowStatistics{mean: data[i]}
		} else if i == 0 || constant[i-1] {
			state = computeWindowStatistics(data[i : i+windowSize])
		} else {
			state = advanceWindowStatistics(state, data[i-1], data[i+windowSize-1], windowSizeF64)
			if shouldRebuildWindowStatistics(state, windowSizeF64) {
				state = computeWindowStatistics(data[i : i+windowSize])
			}
		}
		result.record(i, state, windowSizeF64)
	}

	return result
}

// record stores one window's statistics, converting the sum of squared
// deviations and its absolute bound into a standard deviation and a relative
// bound. The square root halves a relative error, and the division and root
// each round once, which the added epsilon covers.
func (statistics slidingStatistics) record(index int, state windowStatistics, windowSize float64) {
	statistics.means[index] = state.mean
	statistics.meanErrors[index] = state.meanError
	if state.sumOfSquaredDeviations <= 0 {
		return
	}
	statistics.sigmas[index] = math.Sqrt(state.sumOfSquaredDeviations / windowSize)
	statistics.sigmaRelativeErrors[index] = state.sumError/(2*state.sumOfSquaredDeviations) + float64Epsilon
}

// advanceWindowStatistics returns the statistics of the window obtained by
// dropping oldValue and appending newValue, using the exact identities
//
//	mean' = mean + (new − old)/m
//	S'    = S + (new − old)·(new − mean' + old − mean)
//
// and accumulating a first-order forward error bound for every rounded
// operation from the magnitudes of the computed intermediates. The bound on
// the correction term includes the error already present in both means, which
// is how mean drift is charged against the sum.
func advanceWindowStatistics(previous windowStatistics, oldValue, newValue, windowSize float64) windowStatistics {
	// Mean update.
	delta := newValue - oldValue
	deltaError := float64Epsilon * math.Abs(delta)
	step := delta / windowSize
	stepError := deltaError/windowSize + float64Epsilon*math.Abs(step)
	mean := previous.mean + step
	meanError := previous.meanError + stepError + float64Epsilon*math.Abs(mean)

	// Correction term for the sum of squared deviations.
	newDeviation := newValue - mean
	partial := newDeviation + oldValue
	bracket := partial - previous.mean
	bracketError := meanError + previous.meanError +
		float64Epsilon*(math.Abs(newDeviation)+math.Abs(partial)+math.Abs(bracket))
	correction := delta * bracket
	correctionError := math.Abs(delta)*bracketError + math.Abs(bracket)*deltaError + float64Epsilon*math.Abs(correction)

	// Sum update.
	sum := previous.sumOfSquaredDeviations + correction
	sumError := previous.sumError + correctionError + float64Epsilon*math.Abs(sum)

	return windowStatistics{
		mean:                   mean,
		sumOfSquaredDeviations: sum,
		meanError:              meanError,
		sumError:               sumError,
	}
}

// shouldRebuildWindowStatistics reports whether the recurrence's error bound
// has grown past statisticsRebuildRelativeError of the sum, or the sum is no
// longer positive, and a two-pass recalculation would give a tighter bound.
// The fresh bound is estimated from the current mean and sigma, since the
// mean absolute value of a window is at most |mean| + sigma. Skipping a
// futile rebuild keeps ill-conditioned windows, which the distance path will
// recalculate directly anyway, from paying for the statistics twice.
func shouldRebuildWindowStatistics(state windowStatistics, windowSize float64) bool {
	if state.sumOfSquaredDeviations <= 0 {
		return true
	}
	if state.sumError <= statisticsRebuildRelativeError*state.sumOfSquaredDeviations {
		return false
	}

	sigma := math.Sqrt(state.sumOfSquaredDeviations / windowSize)
	freshMeanError := compensatedSumErrorFactor*(math.Abs(state.mean)+sigma) + float64Epsilon*math.Abs(state.mean)
	freshSumError := twoPassSumErrorFactor*state.sumOfSquaredDeviations + windowSize*freshMeanError*freshMeanError
	return state.sumError > 2*freshSumError
}

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
)

func TestFFTDistanceRequiresDirectVerification(t *testing.T) {
	t.Parallel()

	const target = DistanceAccuracyTarget
	testCases := []struct {
		name            string
		distanceSquared float64
		bounds          fftPathErrorBounds
		expected        bool
	}{
		{
			name:            "well within budget",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: 1e-9, preparation: 1e-12},
			expected:        false,
		},
		{
			name:            "squared bound exactly at the budget is accepted",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: 2*10*target - target*target},
			expected:        false,
		},
		{
			name:            "squared bound just above the budget requires verification",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: 2*10*target + target*target},
			expected:        true,
		},
		{
			name:            "preparation error consumes the budget",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: 0, preparation: target},
			expected:        true,
		},
		{
			name:            "preparation error shrinks the squared budget",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: 2 * 10 * (target / 2), preparation: target / 2},
			expected:        true,
		},
		{
			name:            "distance below the target uses the widened budget",
			distanceSquared: 1e-16,
			bounds:          fftPathErrorBounds{distanceSquared: 2*1e-8*target + target*target},
			expected:        false,
		},
		{
			name:            "zero squared distance",
			distanceSquared: 0,
			bounds:          fftPathErrorBounds{},
			expected:        true,
		},
		{
			name:            "negative squared distance",
			distanceSquared: -1e-12,
			bounds:          fftPathErrorBounds{},
			expected:        true,
		},
		{
			name:            "NaN squared distance",
			distanceSquared: math.NaN(),
			bounds:          fftPathErrorBounds{},
			expected:        true,
		},
		{
			name:            "infinite squared distance",
			distanceSquared: math.Inf(1),
			bounds:          fftPathErrorBounds{},
			expected:        true,
		},
		{
			name:            "NaN bound",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: math.NaN()},
			expected:        true,
		},
		{
			name:            "infinite bound",
			distanceSquared: 100,
			bounds:          fftPathErrorBounds{distanceSquared: math.Inf(1)},
			expected:        true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			actual := fftDistanceRequiresDirectVerification(testCase.distanceSquared, testCase.bounds)
			if actual != testCase.expected {
				t.Errorf("expected %v, got %v", testCase.expected, actual)
			}
		})
	}
}

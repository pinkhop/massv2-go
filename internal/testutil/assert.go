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
	"testing"
)

// maximumReportedMismatches caps how many individual mismatches an assertion
// prints before summarizing the rest.
const maximumReportedMismatches = 10

// AlmostEqual reports whether two float64 values differ by no more than
// tolerance. It is false when either value is NaN.
func AlmostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

// AssertDistanceProfilesEqual fails the test when the call returned an error,
// the lengths differ, or any entry disagrees with the expectation beyond
// tolerance. Expected positive infinity must be reproduced exactly, and every
// other actual entry must be finite. All mismatches are counted; the first
// few are printed individually.
func AssertDistanceProfilesEqual(t testing.TB, actual, expected []float64, err error, tolerance float64) {
	t.Helper()
	if err != nil {
		t.Fatalf("distance profile failed: %v", err)
	}
	if len(actual) != len(expected) {
		t.Fatalf("expected %d distances, got %d", len(expected), len(actual))
	}

	mismatches := 0
	report := func(format string, args ...any) {
		mismatches++
		if mismatches <= maximumReportedMismatches {
			t.Errorf(format, args...)
		}
	}
	for index, expectedDistance := range expected {
		actualDistance := actual[index]
		switch {
		case math.IsInf(expectedDistance, 1):
			if !math.IsInf(actualDistance, 1) {
				report("distance at index %d: expected +Inf, got %.16g", index, actualDistance)
			}
		case math.IsNaN(actualDistance) || math.IsInf(actualDistance, 0):
			report("distance at index %d must be finite, got %v", index, actualDistance)
		case !AlmostEqual(actualDistance, expectedDistance, tolerance):
			report("distance at index %d: expected %.16g, got %.16g (difference %.3e)", index, expectedDistance, actualDistance, math.Abs(actualDistance-expectedDistance))
		}
	}
	if mismatches > maximumReportedMismatches {
		t.Errorf("%d of %d distances mismatched; only the first %d were printed", mismatches, len(expected), maximumReportedMismatches)
	}
}

package massv2

import (
	"errors"
	"math"
	"slices"
	"testing"
)

const floatToleranceForFuncsTest = 2e-6

func TestFindBestMatch(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	type TestCase struct {
		Name                        string
		InputTimeSeries             []float64
		InputQuery                  []float64
		ExpectedIndex               int
		ExpectedZNormalizedDistance float64
		ExpectedErr                 error
	}

	testCases := []TestCase{
		{
			Name:                        "basic exact match",
			InputTimeSeries:             []float64{1.1, 1.9, 4.1, 8.1, 15.8, 15.1, 12.9, 9.25, 1.2, 0.1},
			InputQuery:                  []float64{4.1, 8.1, 15.8},
			ExpectedIndex:               2,
			ExpectedZNormalizedDistance: 0.0,
		},
		{
			Name:                        "tie between multiple perfect matches returns earliest index",
			InputTimeSeries:             []float64{1, 2, 3, 1, 2, 3},
			InputQuery:                  []float64{1, 2, 3},
			ExpectedIndex:               0,
			ExpectedZNormalizedDistance: 0.0,
		},
		{
			Name:                        "initial distance is +Inf, later finite best match is chosen",
			InputTimeSeries:             []float64{2, 2, 2, 1, 2, 3}, // the first window [2,2,2] has zero variance -> +Inf distance
			InputQuery:                  []float64{1, 2, 3},
			ExpectedIndex:               3,
			ExpectedZNormalizedDistance: 0.0,
		},
		{
			Name:            "error when query longer than time-series",
			InputTimeSeries: []float64{1, 2, 3},
			InputQuery:      []float64{1, 2, 3, 4},
			ExpectedErr:     ErrQueryLongerThanTimeSeries,
		},
		{
			Name:            "error when query has zero variance",
			InputTimeSeries: []float64{1, 2, 3, 4, 5},
			InputQuery:      []float64{2, 2, 2},
			ExpectedErr:     ErrQueryHasZeroVariance,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// GIVEN (set up)

			// WHEN (operation under test)
			actualIndex, actualDistance, actualErr := FindBestMatch(tc.InputTimeSeries, tc.InputQuery)

			// THEN (assertions)

			// Don't make assertions about the results if FindBestMatch
			// returned an error when one was not expected.
			if actualErr != nil && tc.ExpectedErr == nil {
				t.Fatalf("FindBestMatch failed: %v", actualErr)
			}

			// When we expect an error, test that we received that error.
			if tc.ExpectedErr != nil {
				if !errors.Is(actualErr, tc.ExpectedErr) {
					t.Errorf("expected error %v, got %v", tc.ExpectedErr, actualErr)
				}

				// Since this test case was about the returned error value, do
				// not make assertions about the indices and distances.
				return
			}

			// If an error was expected, skip further assertions
			if tc.ExpectedErr != nil {
				return
			}

			if actualIndex != tc.ExpectedIndex {
				t.Errorf("expected index %d, got %d", tc.ExpectedIndex, actualIndex)
			}
			if math.Abs(actualDistance-tc.ExpectedZNormalizedDistance) > floatToleranceForFuncsTest {
				t.Errorf("expected z-normalized distance %.10e, got %.10e (difference of %.10e)", tc.ExpectedZNormalizedDistance, actualDistance, math.Abs(actualDistance-tc.ExpectedZNormalizedDistance))
			}
		})
	}
}

func TestFindTopKMatches(t *testing.T) {
	t.Parallel() // this test is stateless and can be run in parallel with other tests

	type TestCase struct {
		Name                         string
		InputTimeSeries              []float64
		InputQuery                   []float64
		InputK                       int
		ExpectedIndices              []int
		ExpectedZNormalizedDistances []float64
		ExpectedErr                  error
		AllowIndicesInAnyOrder       bool // when all the distances are "the same", floating point imprecision can cause "random" ordering
	}

	testCases := []TestCase{
		{
			Name: "basic exact match when k=1",
			InputTimeSeries: []float64{
				1.1, 1.9,
				4.1, 8.1, 15.8, // starting at index 2, exact match
				15.1, 12.9, 9.25, 31.2,
			},
			InputQuery:                   []float64{4.1, 8.1, 15.8},
			InputK:                       1,
			ExpectedIndices:              []int{2},
			ExpectedZNormalizedDistances: []float64{0.0},
		},
		{
			Name: "basic exact matches when k>1",
			InputTimeSeries: []float64{
				1.1, 1.9,
				4.1, 8.1, 15.8, // starting at index 2, exact match
				15.1, 12.9, 12.9,
				4.1, 8.1, 15.8, // starting at index 8, exact match
				9.25, 1.2, 0.1,
			},
			InputQuery:                   []float64{4.1, 8.1, 15.8}, // exact matches at 2 and 7
			InputK:                       2,
			ExpectedIndices:              []int{2, 8},
			ExpectedZNormalizedDistances: []float64{0.0, 0.0},
			AllowIndicesInAnyOrder:       true, // k>1 and all matching distances are the same
		},
		{
			Name: "matches different amplitudes (z-normalization)",
			InputTimeSeries: []float64{
				1.1, 1.9,
				8.2, 16.2, 31.6, // starting at index 2, query values * 2
				15.1, 88.8, 77.7, 12.9, -1.2,
				41, 81, 158, // starting at index 10, query values * 10
				9.25, 1.2, 0.1,
			},
			InputQuery:                   []float64{4.1, 8.1, 15.8},
			InputK:                       2,
			ExpectedIndices:              []int{2, 10},
			ExpectedZNormalizedDistances: []float64{0.0, 0.0},
			AllowIndicesInAnyOrder:       true, // k>1 and all matching distances are the same
		},
		{
			Name: "matches different offsets (z-normalization)",
			InputTimeSeries: []float64{
				1.1, 1.9, 0.2,
				10.0, 14.0, 21.7, // starting at index 3, matches query +5.9
				16.3, 12.9,
				-22.1, -18.1, -10.4, // starting at index 8, matches query -26.2
				9.25, 1.2, 0.1,
			},
			InputQuery:                   []float64{4.1, 8.1, 15.8},
			InputK:                       2,
			ExpectedIndices:              []int{3, 8},
			ExpectedZNormalizedDistances: []float64{0.0, 0.0},
			AllowIndicesInAnyOrder:       true, // k>1 and all matching distances are the same
		},
		{
			Name: "matches different amplitudes and offsets (z-normalization)",
			InputTimeSeries: []float64{
				1.1, 1.9, 0.2,
				42, 82, 159, // starting at index 3, query *10 +1
				15.1,
				-6.67, 6.53, 31.94, // starting at index 7, query *3.3 -20.2
				9.25, 1.2, 0.1},
			InputQuery:                   []float64{4.1, 8.1, 15.8},
			InputK:                       2,
			ExpectedIndices:              []int{3, 7},
			ExpectedZNormalizedDistances: []float64{0.0, 0.0},
			AllowIndicesInAnyOrder:       true, // k>1 and all matching distances are the same
		},
		{
			Name:                         "no exact match, k=2 returns smallest distances in index order when tied",
			InputTimeSeries:              []float64{0, 1, 4, 9, 3, 6, 10},
			InputQuery:                   []float64{1, 2, 4},
			InputK:                       2,
			ExpectedIndices:              []int{1, 0}, // z-norm distances: [0.157, 0.081, 2.834, 2.822, 0.186]
			ExpectedZNormalizedDistances: []float64{0.08101454540986468, 0.15740530704225447},
		},
		{
			Name:                         "k greater than number of windows is capped (all ties by index)",
			InputTimeSeries:              []float64{1, 2, 4, 8, 16, 32},
			InputQuery:                   []float64{1, 2, 4},
			InputK:                       10,
			ExpectedIndices:              []int{0, 1, 2, 3},
			ExpectedZNormalizedDistances: []float64{0, 0, 0, 0},
			AllowIndicesInAnyOrder:       true,
		},
		{
			Name:                         "when k=2 and there is a tie between multiple perfect matches, returns both",
			InputTimeSeries:              []float64{1, 2, 3, 1, 2, 3},
			InputQuery:                   []float64{1, 2, 3},
			InputK:                       2,
			ExpectedIndices:              []int{0, 3},
			ExpectedZNormalizedDistances: []float64{0.0, 0.0},
			AllowIndicesInAnyOrder:       true, // k>1 and all matching distances are the same
		},
		{
			Name:            "error when k=0",
			InputTimeSeries: []float64{1, 2, 3, 4, 5},
			InputQuery:      []float64{2, 3},
			InputK:          0,
			ExpectedErr:     ErrKMustBePositive,
		},
		{
			Name:            "error when k<0",
			InputTimeSeries: []float64{1, 2, 3, 4, 5},
			InputQuery:      []float64{2, 3},
			InputK:          -1,
			ExpectedErr:     ErrKMustBePositive,
		},
		{
			Name:            "error when query longer than time-series",
			InputTimeSeries: []float64{1, 2, 3},
			InputQuery:      []float64{1, 2, 3, 4},
			InputK:          1,
			ExpectedErr:     ErrQueryLongerThanTimeSeries,
		},
		{
			Name:            "error when query has zero variance",
			InputTimeSeries: []float64{1, 2, 3, 4, 5},
			InputQuery:      []float64{2, 2, 2},
			InputK:          1,
			ExpectedErr:     ErrQueryHasZeroVariance,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// GIVEN (set up)

			// WHEN (operation under test)
			actualIndices, actualDistances, actualErr := FindTopKMatches(tc.InputTimeSeries, tc.InputQuery, tc.InputK)

			// THEN (assertions)

			// Don't make assertions about the results if FindTopKMatches
			// returned an error when one was not expected.
			if actualErr != nil && tc.ExpectedErr == nil {
				t.Fatalf("FindTopKMatches failed: %v", actualErr)
			}

			// When we expect an error, test that we received that error.
			if tc.ExpectedErr != nil {
				if !errors.Is(actualErr, tc.ExpectedErr) {
					t.Errorf("expected error %v, got %v", tc.ExpectedErr, actualErr)
				}

				// Since this test case was about the returned error value, do
				// not make assertions about the indices and distances.
				return
			}

			if len(actualIndices) != len(tc.ExpectedIndices) {
				t.Fatalf("expected %d indices, got %d", len(tc.ExpectedIndices), len(actualIndices))
			}
			if len(actualDistances) != len(tc.ExpectedZNormalizedDistances) {
				t.Fatalf("expected %d z-normalized distances, got %d", len(tc.ExpectedZNormalizedDistances), len(actualDistances))
			}

			if tc.AllowIndicesInAnyOrder {
				// Just ensure that actual contains all the values in expected
				for _, expected := range tc.ExpectedIndices {
					if !slices.Contains(actualIndices, expected) {
						t.Errorf("expected indices to include %d, but it is missing from %v", expected, actualIndices)
					}
				}
			} else {
				// Ensure that the indices are in the expected order
				for i, expected := range tc.ExpectedIndices {
					actual := actualIndices[i]
					if actual != expected {
						t.Errorf("expected top k result at index %d to be %d, got %d", i, expected, actual)
					}
				}
			}

			for i, expected := range tc.ExpectedZNormalizedDistances {
				actual := actualDistances[i]
				if math.Abs(actual-expected) > floatToleranceForFuncsTest {
					massDistances, _ := MASSV2(tc.InputTimeSeries, tc.InputQuery)
					t.Logf("MASSV2=%v", massDistances)
					t.Errorf("expected top k distance at index %d to be %f, got %f (difference of %.10e)", i, expected, actual, math.Abs(actual-expected))
				}
			}
		})
	}
}

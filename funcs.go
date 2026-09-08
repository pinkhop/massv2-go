package massv2

import (
	"errors"
	"math"
	"slices"
)

var ErrKMustBePositive = errors.New("k must be a positive integer")

// ErrNoFiniteMatch is returned by FindBestMatch when no window has a finite
// distance from the query.
var ErrNoFiniteMatch = errors.New("no finite match")

// FindBestMatch uses MASSV2 to identify the starting index of the subsequence
// in the time-series that best matches the query. FindBestMatch returns the
// index of the best match and its z-normalized Euclidean Distance from the
// query. It never returns a non-finite distance. If no window has a finite
// distance, it returns ErrNoFiniteMatch. On any error, index and distance
// are both -1. Ties between equal finite distances select the earliest index.
//
// The distance has the accuracy target and direct-recalculation limitations
// described in the package documentation.
//
// When MASSV2 returns an error while processing timeSeries and query,
// FindBestMatch returns that error.
func FindBestMatch(timeSeries, query []float64) (idx int, dist float64, err error) {
	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		return -1, -1, err
	}

	idx = -1
	dist = -1

	for i, currDist := range distances {
		if math.IsNaN(currDist) || math.IsInf(currDist, 0) {
			continue
		}
		if idx == -1 || currDist < dist {
			dist = currDist
			idx = i
		}
	}

	if idx == -1 {
		return -1, -1, ErrNoFiniteMatch
	}

	return idx, dist, nil
}

// FindTopKMatches uses MASSV2 to identify up to k subsequences in the
// time-series that best match the query. FindTopKMatches returns the indices
// of those matches and their z-normalized Euclidean Distances from the
// query. It never returns non-finite distances. Results are ordered by
// increasing distance, with equal distances ordered by starting index.
// If fewer than k windows have finite distances, it returns fewer than k
// matches. If none have finite distances, it returns empty results with no
// error.
//
// The distances have the accuracy target and direct-recalculation limitations
// described in the package documentation.
//
// When MASSV2 returns an error while processing timeSeries and query,
// FindTopKMatches returns that error.
func FindTopKMatches(
	timeSeries,
	query []float64,
	k int,
) (indices []int, dists []float64, err error) {
	// Guard statements
	if k < 1 {
		return nil, nil, ErrKMustBePositive
	}

	allDistances, err := MASSV2(timeSeries, query)
	if err != nil {
		return nil, nil, err
	}

	type match struct {
		index    int
		distance float64
	}

	// Keep finite subsequence distances and sort them shortest first.
	matches := make([]match, 0, len(allDistances))
	for i, dist := range allDistances {
		if math.IsNaN(dist) || math.IsInf(dist, 0) {
			continue
		}
		matches = append(matches, match{i, dist})
	}
	slices.SortFunc(matches, func(a, b match) int {
		if a.distance < b.distance {
			return -1
		} else if a.distance > b.distance {
			return 1
		}

		return a.index - b.index
	})

	k = min(k, len(matches))
	indices = make([]int, k)
	dists = make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = matches[i].index
		dists[i] = matches[i].distance
	}

	return indices, dists, nil
}

package massv2

import (
	"errors"
	"slices"
)

var (
	ErrKMustBePositive = errors.New("k must be a positive integer")
)

// FindBestMatch uses MASSV2 to identify the starting index of the subsequence
// in the time-series that best matches the query. FindBestMatch returns the
// index of the best match and its z-normalized Euclidean Distance from the
// query.
//
// When MASSV2 returns an error while processing timeSeries and query,
// FindBestMatch returns that error.
func FindBestMatch(timeSeries, query []float64) (idx int, dist float64, err error) {
	distances, err := MASSV2(timeSeries, query)
	if err != nil {
		return -1, -1, err
	}

	idx = 0
	dist = distances[0]

	for i, currDist := range distances {
		if currDist < dist {
			dist = currDist
			idx = i
		}
	}

	return idx, dist, nil
}

// FindTopKMatches uses MASSV2 to identify the k subsequences in the
// time-series that best match the query. FindTopKMatches returns the indices
// of the top k matches and their z-normalized Euclidean Distances from the
// query.
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

	if k > len(allDistances) {
		k = len(allDistances)
	}

	type match struct {
		index    int
		distance float64
	}

	// Sort the subsequence distances, shortest first.
	matches := make([]match, len(allDistances))
	for i, dist := range allDistances {
		matches[i] = match{i, dist}
	}
	slices.SortFunc(matches, func(a, b match) int {
		if a.distance < b.distance {
			return -1
		} else if a.distance > b.distance {
			return 1
		}

		return a.index - b.index
	})

	indices = make([]int, k)
	dists = make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = matches[i].index
		dists[i] = matches[i].distance
	}

	return indices, dists, nil
}

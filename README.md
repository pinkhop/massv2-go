# Mueen's Algorithm for Similarity Search (MASS)

This module implements version 2 of Mueen's Algorithm for Similarity Search (MASS_V2). MASS creates the distance profile of a query against a long time series: given a query subsequence `Q` of length _m_ and a time series `T` of length _n_, MASS computes the z-normalized Euclidean distance between `Q` and every subsequence in `T` of length _m_.

The key idea underlying MASS is that z-normalized Euclidean distance reduces to a formula involving sliding dot products of `Q` against subsequences of `T`, plus mean and variance terms. Fast convolution through the fast Fourier transform (FFT) computes the sliding dot products in O(n log n) time. This implementation checks every FFT-path distance against a forward roundoff bound and directly recalculates any window the bound cannot certify within the accuracy target.

Additionally, this module provides convenience functions to find the best match or the top K matches in the time series.


## Citation

Abdullah Mueen, Sheng Zhong, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta, and Eamonn Keogh (2022), "The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance."

URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html


## Algorithm Complexity

- **Time:** O(n log n) for the FFT path, plus O(m) for each window whose FFT roundoff bound cannot certify the accuracy target
    - Exact and near-exact matches are always recalculated directly, because the FFT distance formula cancels catastrophically as the distance approaches zero
    - Nonconstant windows whose variation is tiny relative to the range of the whole series are also recalculated; a profile consisting entirely of such windows costs O(nm)
    - Constant windows are identified by linear equality scans and skip statistics rebuilding and direct recalculation; an entirely constant series skips the FFT and takes O(n + m) time
    - On ordinary data, including series with isolated spikes, linear trends, and large constant offsets, every window stays on the FFT path; the test suite asserts this on Gaussian series of up to one million observations
- **Space:** O(n), with one reusable O(m) scratch buffer allocated only when direct recalculation is needed, rather than a buffer per window
    - Requires memory for the entire time series, plus FFT working space


## Numerical Accuracy

The absolute accuracy target is `1e-7` in z-normalized distance units. Each FFT-path entry is accepted only when the combined forward bound on preparation, statistics, and convolution rounding fits inside that target; otherwise the window is recalculated directly from its original values.

Direct recalculation is expected to meet the same target for query lengths up to about one million values, based on the preparation and statistics error analysis. It does not check an error bound on the returned distance, so the target is not an unconditional guarantee for every finite result. Longer queries are accepted, but their directly recalculated distances are not guaranteed to meet the target. A nil error reports successful computation, not a separate accuracy certification. These limits also apply to distances returned by `FindBestMatch` and `FindTopKMatches`.

The series and query are each translated, scaled, and centered before any statistic or transform is calculated, so representable variation around large offsets survives, no intermediate overflows, and the FFT receives values of order one. Sliding means and standard deviations are computed by a compensated two-pass method and a replacement recurrence that tracks its own rounding error and rebuilds a window from scratch when that error grows.

A window whose values are all equal has no z-normalized form. `MASSV2` includes positive infinity (`+Inf`) for that window in the returned profile, even when the returned error is nil. Callers must check for these entries (for example, with `math.IsInf(d, 1)`) before arithmetic or serialization that requires finite values. A query whose values are all equal is rejected, as is any input containing NaN or an infinite value; the returned error identifies the offending argument.

`FindBestMatch` and `FindTopKMatches` never return non-finite distances. `FindBestMatch` returns `ErrNoFiniteMatch` when no finite candidate exists; on any error, its index and distance are both `-1`. `FindTopKMatches` returns up to `k` finite matches, so it can return fewer than `k` results when constant windows are excluded or the profile has fewer than `k` windows. If no finite candidates exist, it returns empty results with no error. Matches are ordered by increasing distance, with equal distances ordered by starting index.


## Package Layout

The root package `massv2` is the public API: `MASSV2`, `FindBestMatch`, `FindTopKMatches`, and their error values. Everything else lives under `internal/`, which the Go toolchain prevents other modules from importing: `internal/distance` holds the preparation, sliding statistics, FFT convolution, and roundoff certification that compute a profile, and `internal/testutil` holds the fixtures, exact references, and assertions shared by the test suites. Identifiers exported from `internal/` packages are not part of the supported API and may change without notice.

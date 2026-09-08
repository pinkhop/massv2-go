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

package massv2_test

import (
	"fmt"
	"testing"

	massv2 "github.com/pinkhop/massv2-go"
	"github.com/pinkhop/massv2-go/internal/testutil"
)

func BenchmarkMASSV2_withMEqual200(b *testing.B) {
	benchmarkMASSV2(b, 200, []int{50_000, 100_000, 200_000}, 42, 84)
}

func BenchmarkMASSV2_withMEqual1000(b *testing.B) {
	benchmarkMASSV2(b, 1_000, []int{50_000, 100_000, 200_000}, 42, 84)
}

func BenchmarkMASSV2_withMEqual5000(b *testing.B) {
	benchmarkMASSV2(b, 5_000, []int{50_000, 100_000, 200_000}, testutil.DefaultSeed0, testutil.DefaultSeed1)
}

func BenchmarkMASSV2_withMEqual100_LongSeries(b *testing.B) {
	benchmarkMASSV2(b, 100, []int{1_000_000}, 42, 84)
}

// BenchmarkMASSV2_Scaling varies both dimensions to expose query-length costs.
// Timings and allocations are measurements, not an asymptotic pass/fail gate.
func BenchmarkMASSV2_Scaling(b *testing.B) {
	for _, queryLength := range []int{50, 200, 800} {
		b.Run(fmt.Sprintf("m=%d", queryLength), func(b *testing.B) {
			benchmarkMASSV2(b, queryLength, []int{1_000, 2_000, 4_000, 8_000}, 12345, 54321)
		})
	}
}

// BenchmarkMASSV2_DegenerateScaling separates constant exclusion, plateau
// transitions, and nonconstant direct fallback across both input dimensions.
func BenchmarkMASSV2_DegenerateScaling(b *testing.B) {
	for _, kind := range []string{"constant", "plateaus", "ramp"} {
		for _, n := range []int{4096, 16384} {
			for _, m := range []int{32, 256} {
				b.Run(fmt.Sprintf("%s/n=%d/m=%d", kind, n, m), func(b *testing.B) {
					series := make([]float64, n)
					query := make([]float64, m)
					for i := range query {
						query[i] = float64(i)
					}
					for i := range series {
						switch kind {
						case "plateaus":
							series[i] = float64(i / (2 * m))
						case "ramp":
							series[i] = float64(i)
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_, err := massv2.MASSV2(series, query)
						if err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// benchmarkMASSV2 times the public distance-profile call for one query length
// across several series lengths and reports allocations.
func benchmarkMASSV2(b *testing.B, queryLength int, sizes []int, seriesSeed, querySeed uint64) {
	b.Helper()
	for _, n := range sizes {
		timeSeries := testutil.GenerateSyntheticData(n, seriesSeed)
		query := testutil.GenerateSyntheticData(queryLength, querySeed)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := massv2.MASSV2(timeSeries, query)
				if err != nil {
					b.Fatalf("MASSV2 failed: %v", err)
				}
			}
		})
	}
}

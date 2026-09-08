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

// Package testutil provides the fixtures, exact references, independent
// oracles, and assertions shared by the test suites of the public massv2
// package and the internal distance package. It is imported only by test
// files and is never linked into a consumer's binary.
package testutil

import (
	"math"
	"math/rand/v2"
)

// DefaultSeed0 and DefaultSeed1 seed the PCG generator when a caller supplies
// no seeds, so every fixture is reproducible across runs.
const (
	DefaultSeed0 uint64 = 0x7fa2_2276_889c_4782
	DefaultSeed1 uint64 = 0xaf4f_33b8_2757_b871
)

// NewPRNG returns a PCG generator seeded with the given values, or with the
// default seeds when fewer than two are supplied. The generator produces
// deterministic fixtures and is not a source of security material.
func NewPRNG(seed ...uint64) *rand.Rand {
	seed0 := DefaultSeed0
	seed1 := DefaultSeed1
	if len(seed) > 0 {
		seed0 = seed[0]
		if len(seed) > 1 {
			seed1 = seed[1]
		}
	}

	return rand.New(rand.NewPCG(seed0, seed1)) // #nosec G404 -- deterministic test fixtures
}

// GenerateSyntheticData generates n standard normal observations from a
// seeded generator.
func GenerateSyntheticData(n int, seed ...uint64) []float64 {
	prng := NewPRNG(seed...)
	data := make([]float64, n)
	for i := range data {
		data[i] = prng.NormFloat64()
	}
	return data
}

// GenerateSineWave generates n samples of a sinusoid without noise. frequency
// is in cycles per n samples.
func GenerateSineWave(n int, frequency, amplitude, phase float64) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)/float64(n)+phase)
	}
	return data
}

// GenerateNoisySineWave generates n samples of a sinusoid with Gaussian noise
// clipped to plus or minus 10% of the amplitude. frequency is in cycles per n
// samples.
func GenerateNoisySineWave(n int, frequency, amplitude, phase float64, seed ...uint64) []float64 {
	noiseMax := math.Abs(amplitude) * 0.1
	noiseStddev := noiseMax * 0.333

	prng := NewPRNG(seed...)
	data := make([]float64, n)
	for i := range data {
		data[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)/float64(n)+phase)

		noise := prng.NormFloat64() * noiseStddev
		if math.Abs(noise) <= noiseMax {
			data[i] += noise
		}
	}

	return data
}

// LargeOutlierSeries builds the shared fixture of a large first observation
// followed by varied ordinary observations. Tests in more than one package
// select an exact query occurrence from it well after the outlier has
// departed.
func LargeOutlierSeries() []float64 {
	timeSeries := make([]float64, 1_000)
	timeSeries[0] = 1e9
	for index := 1; index < len(timeSeries); index++ {
		timeSeries[index] = math.Sin(float64(index)*0.37) + float64(index%11)/7
	}
	return timeSeries
}

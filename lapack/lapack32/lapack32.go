// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lapack32

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/gonum32"
)

var lapack32 lapack.Float32 = gonum32.Implementation{}

// Use sets the LAPACK float32 implementation to be used by subsequent BLAS calls.
// The default implementation is native.Implementation.
func Use(l lapack.float32) {
	lapack32 = l
}

// Tridiagonal represents a tridiagonal matrix using its three diagonals.
type Tridiagonal struct {
	N  int
	DL []float32
	D  []float32
	DU []float32
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Lansy computes the specified norm of an n×n symmetric matrix. If
// norm == lapack.MaxColumnSum or norm == lapack.MaxRowSum, work must have length
// at least n and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func Lansy(norm lapack.MatrixNorm, a blas64.Symmetric, work []float32) float32 {
	return lapack32.Dlansy(norm, a.Uplo, a.N, a.Data, max(1, a.Stride), work)
}

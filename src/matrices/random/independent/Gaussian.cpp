/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/matrices.hpp>

namespace El {

template <typename F>
void MakeGaussian(AbstractMatrix<F>& A, F mean, Base<F> stddev)
{
    EL_DEBUG_CSE
    switch (A.GetDevice())
    {
    case Device::CPU:
        MakeGaussian(static_cast<Matrix<F,Device::CPU>&>(A), mean, stddev);
        break;
    case Device::GPU:
        MakeGaussian(static_cast<Matrix<F,Device::GPU>&>(A), mean, stddev);
        break;
    default:
        LogicError("MakeGaussian: Bad device.");
    }
}

// Draw each entry from a normal PDF
template<typename F, Device D>
void MakeGaussian( Matrix<F, D>& A, F mean, Base<F> stddev )
{
    EL_DEBUG_CSE
    auto sampleNormal = [=]() { return SampleNormal(mean,stddev); };
    EntrywiseFill( A, function<F()>(sampleNormal) );
}

template<typename F>
void MakeGaussian( AbstractDistMatrix<F>& A, F mean, Base<F> stddev )
{
    EL_DEBUG_CSE
    if( A.RedundantRank() == 0 )
        MakeGaussian( A.Matrix(), mean, stddev );
    Broadcast( A, A.RedundantComm(), 0 );
}

template<typename F, Device D>
void Gaussian( Matrix<F, D>& A, Int m, Int n, F mean, Base<F> stddev )
{
    EL_DEBUG_CSE
    A.Resize( m, n );
    MakeGaussian( A, mean, stddev );
}

template<typename F>
void Gaussian( AbstractMatrix<F>& A, Int m, Int n, F mean, Base<F> stddev )
{
    EL_DEBUG_CSE
    A.Resize( m, n );
    MakeGaussian( A, mean, stddev );
}

template<typename F>
void Gaussian
( AbstractDistMatrix<F>& A, Int m, Int n, F mean, Base<F> stddev )
{
    EL_DEBUG_CSE
    A.Resize( m, n );
    MakeGaussian( A, mean, stddev );
}

#define PROTO(F) \
  template void MakeGaussian \
  ( AbstractMatrix<F>& A, F mean, Base<F> stddev ); \
  template void MakeGaussian \
  ( Matrix<F, El::Device::CPU>& A, F mean, Base<F> stddev ); \
  template void MakeGaussian \
  ( Matrix<F, El::Device::GPU>& A, F mean, Base<F> stddev ); \
  template void MakeGaussian \
  ( AbstractDistMatrix<F>& A, F mean, Base<F> stddev ); \
  template void Gaussian \
  ( Matrix<F>& A, Int m, Int n, F mean, Base<F> stddev ); \
  template void Gaussian \
  ( AbstractDistMatrix<F>& A, Int m, Int n, F mean, Base<F> stddev );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

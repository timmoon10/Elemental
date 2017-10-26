/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_KRONECKER_HPP
#define EL_BLAS_KRONECKER_HPP

namespace El {

template<typename T>
void Kronecker
( const Matrix<T>& A,
  const Matrix<T>& B,
        Matrix<T>& C )
{
    EL_DEBUG_CSE
    const Int mA = A.Height();
    const Int nA = A.Width();
    const Int mB = B.Height();
    const Int nB = B.Width();
    C.Resize( mA*mB, nA*nB );

    for( Int jA=0; jA<nA; ++jA )
    {
        for( Int iA=0; iA<mA; ++iA )
        {
            auto Cij = C( IR(iA*mB,(iA+1)*mB), IR(jA*nB,(jA+1)*nB) );
            Cij = B;
            Scale( A(iA,jA), Cij );
        }
    }
}

template<typename T>
void Kronecker
( const Matrix<T>& A,
  const Matrix<T>& B,
        ElementalMatrix<T>& CPre )
{
    EL_DEBUG_CSE

    DistMatrixWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& C = CProx.Get();

    const Int mA = A.Height();
    const Int nA = A.Width();
    const Int mB = B.Height();
    const Int nB = B.Width();
    C.Resize( mA*mB, nA*nB );

    const Int localHeight = C.LocalHeight();
    const Int localWidth = C.LocalWidth();
    auto& CLoc = C.Matrix();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        const Int j = C.GlobalCol(jLoc);
        const Int jA = j / nB;
        const Int jB = j % nB;
        for( Int iLoc=0; iLoc<localHeight; ++iLoc )
        {
            const Int i = C.GlobalRow(iLoc);
            const Int iA = i / mB;
            const Int iB = i % mB;
            CLoc(iLoc,jLoc) = A(iA,jA)*B(iB,jB);
        }
    }
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Kronecker \
  ( const Matrix<T>& A, \
    const Matrix<T>& B, \
          Matrix<T>& C ); \
  EL_EXTERN template void Kronecker \
  ( const Matrix<T>& A, \
    const Matrix<T>& B, \
          ElementalMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_KRONECKER_HPP

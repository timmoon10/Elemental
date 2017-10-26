/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_CONCATENATE_HPP
#define EL_BLAS_CONCATENATE_HPP

namespace El {

template<typename T>
void HCat
( const Matrix<T>& A,
  const Matrix<T>& B,
        Matrix<T>& C )
{
    EL_DEBUG_CSE
    if( A.Height() != B.Height() )
        LogicError("Incompatible heights for HCat");
    const Int m = A.Height();
    const Int nA = A.Width();
    const Int nB = B.Width();

    C.Resize( m, nA+nB );
    Zero( C );
    auto CL = C( IR(0,m), IR(0,nA)     );
    auto CR = C( IR(0,m), IR(nA,nA+nB) );
    CL = A;
    CR = B;
}

template<typename T>
void VCat
( const Matrix<T>& A,
  const Matrix<T>& B,
        Matrix<T>& C )
{
    EL_DEBUG_CSE
    if( A.Width() != B.Width() )
        LogicError("Incompatible widths for VCat");
    const Int mA = A.Height();
    const Int mB = B.Height();
    const Int n = A.Width();

    C.Resize( mA+mB, n );
    Zero( C );
    auto CT = C( IR(0,mA),     IR(0,n) );
    auto CB = C( IR(mA,mA+mB), IR(0,n) );
    CT = A;
    CB = B;
}

template<typename T>
inline void HCat
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& CPre )
{
    EL_DEBUG_CSE
    if( A.Height() != B.Height() )
        LogicError("Incompatible heights for HCat");
    const Int m = A.Height();
    const Int nA = A.Width();
    const Int nB = B.Width();

    DistMatrixWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& C = CProx.Get();

    C.Resize( m, nA+nB );
    Zero( C );
    auto CL = C( IR(0,m), IR(0,nA)     );
    auto CR = C( IR(0,m), IR(nA,nA+nB) );
    CL = A;
    CR = B;
}

template<typename T>
void VCat
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& CPre )
{
    EL_DEBUG_CSE
    if( A.Width() != B.Width() )
        LogicError("Incompatible widths for VCat");
    const Int mA = A.Height();
    const Int mB = B.Height();
    const Int n = A.Width();

    DistMatrixWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& C = CProx.Get();

    C.Resize( mA+mB, n );
    Zero( C );
    auto CT = C( IR(0,mA),     IR(0,n) );
    auto CB = C( IR(mA,mA+mB), IR(0,n) );
    CT = A;
    CB = B;
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void HCat \
  ( const Matrix<T>& A, \
    const Matrix<T>& B, \
          Matrix<T>& C ); \
  EL_EXTERN template void VCat \
  ( const Matrix<T>& A, \
    const Matrix<T>& B, \
          Matrix<T>& C ); \
  EL_EXTERN template void HCat \
  ( const AbstractDistMatrix<T>& A, \
    const AbstractDistMatrix<T>& B, \
          AbstractDistMatrix<T>& C ); \
  EL_EXTERN template void VCat \
  ( const AbstractDistMatrix<T>& A, \
    const AbstractDistMatrix<T>& B, \
          AbstractDistMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_CONCATENATE_HPP

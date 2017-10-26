/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level3.hpp>

#include "./Syrk/LN.hpp"
#include "./Syrk/LT.hpp"
#include "./Syrk/UN.hpp"
#include "./Syrk/UT.hpp"

namespace El {

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const Matrix<T>& A, T beta, Matrix<T>& C, bool conjugate )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( orientation == NORMAL )
      {
          if( A.Height() != C.Height() || A.Height() != C.Width() )
              LogicError("Nonconformal Syrk");
      }
      else
      {
          if( A.Width() != C.Height() || A.Width() != C.Width() )
              LogicError("Nonconformal Syrk");
      }
    )
    const char uploChar = UpperOrLowerToChar( uplo );
    const char transChar = OrientationToChar( orientation );
    const Int k = ( orientation == NORMAL ? A.Width() : A.Height() );
    if( conjugate )
    {
        blas::Herk
        ( uploChar, transChar, C.Height(), k,
          RealPart(alpha), A.LockedBuffer(), A.LDim(),
          RealPart(beta),  C.Buffer(),       C.LDim() );
    }
    else
    {
        blas::Syrk
        ( uploChar, transChar, C.Height(), k,
          alpha, A.LockedBuffer(), A.LDim(),
          beta,  C.Buffer(),       C.LDim() );
    }
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const Matrix<T>& A, Matrix<T>& C, bool conjugate )
{
    EL_DEBUG_CSE
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    C.Resize( n, n );
    Zero( C );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const AbstractDistMatrix<T>& A,
  T beta,        AbstractDistMatrix<T>& C, bool conjugate )
{
    EL_DEBUG_CSE
    ScaleTrapezoid( beta, uplo, C );
    if( uplo == LOWER && orientation == NORMAL )
        syrk::LN( alpha, A, C, conjugate );
    else if( uplo == LOWER )
        syrk::LT( alpha, A, C, conjugate );
    else if( orientation == NORMAL )
        syrk::UN( alpha, A, C, conjugate );
    else
        syrk::UT( alpha, A, C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const AbstractDistMatrix<T>& A,
                 AbstractDistMatrix<T>& C, bool conjugate )
{
    EL_DEBUG_CSE
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    C.Resize( n, n );
    Zero( C );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}


#define PROTO(T) \
  template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const Matrix<T>& A, T beta, Matrix<T>& C, bool conjugate ); \
  template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const Matrix<T>& A, Matrix<T>& C, bool conjugate ); \
  template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const AbstractDistMatrix<T>& A, \
    T beta, AbstractDistMatrix<T>& C, bool conjugate ); \
  template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const AbstractDistMatrix<T>& A, \
                   AbstractDistMatrix<T>& C, bool conjugate );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_MAKESYMMETRIC_HPP
#define EL_BLAS_MAKESYMMETRIC_HPP

namespace El {

template<typename T>
void MakeSymmetric( UpperOrLower uplo, Matrix<T>& A, bool conjugate )
{
    EL_DEBUG_CSE
    const Int n = A.Width();
    if( A.Height() != n )
        LogicError("Cannot make non-square matrix symmetric");

    if( conjugate )
        MakeDiagonalReal(A);

    T* ABuf = A.Buffer();
    const Int ldim = A.LDim();
    if( uplo == LOWER )
    {
        for( Int j=0; j<n; ++j )
        {
            for( Int i=j+1; i<n; ++i )
            {
                if( conjugate )
                    ABuf[j+i*ldim] = Conj(ABuf[i+j*ldim]);
                else
                    ABuf[j+i*ldim] = ABuf[i+j*ldim];
            }
        }
    }
    else
    {
        for( Int j=0; j<n; ++j )
        {
            for( Int i=0; i<j; ++i )
            {
                if( conjugate )
                    ABuf[j+i*ldim] = Conj(ABuf[i+j*ldim]);
                else
                    ABuf[j+i*ldim] = ABuf[i+j*ldim];
            }
        }
    }
}

template<typename T>
void MakeSymmetric
( UpperOrLower uplo, ElementalMatrix<T>& A, bool conjugate )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Cannot make non-square matrix symmetric");

    MakeTrapezoidal( uplo, A );
    if( conjugate )
        MakeDiagonalReal(A);

    unique_ptr<ElementalMatrix<T>> ATrans( A.Construct(A.Grid(),A.Root()) );
    Transpose( A, *ATrans, conjugate );
    if( uplo == LOWER )
        AxpyTrapezoid( UPPER, T(1), *ATrans, A, 1 );
    else
        AxpyTrapezoid( LOWER, T(1), *ATrans, A, -1 );
}

template<typename T>
void MakeHermitian( UpperOrLower uplo, Matrix<T>& A )
{
    EL_DEBUG_CSE
    MakeSymmetric( uplo, A, true );
}

template<typename T>
void MakeHermitian( UpperOrLower uplo, ElementalMatrix<T>& A )
{
    EL_DEBUG_CSE
    MakeSymmetric( uplo, A, true );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void MakeSymmetric \
  ( UpperOrLower uplo, Matrix<T>& A, bool conjugate ); \
  EL_EXTERN template void MakeSymmetric \
  ( UpperOrLower uplo, ElementalMatrix<T>& A, bool conjugate ); \
  EL_EXTERN template void MakeHermitian \
  ( UpperOrLower uplo, Matrix<T>& A ); \
  EL_EXTERN template void MakeHermitian \
  ( UpperOrLower uplo, ElementalMatrix<T>& A );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_MAKESYMMETRIC_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_SHIFTDIAGONAL_HPP
#define EL_BLAS_SHIFTDIAGONAL_HPP

namespace El {

template<typename T,typename S>
void ShiftDiagonal( Matrix<T>& A, S alpha, Int offset )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();
    for( Int j=0; j<width; ++j )
    {
        const Int i = j-offset;
        if( i >= 0 && i < height )
            ABuf[i+j*ALDim] += alpha;
    }
}

template<typename T,typename S>
void ShiftDiagonal( AbstractDistMatrix<T>& A, S alpha, Int offset )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int localWidth = A.LocalWidth();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        const Int j = A.GlobalCol(jLoc);
        const Int i = j-offset;
        if( i >= 0 && i < height && A.IsLocalRow(i) )
        {
            const Int iLoc = A.LocalRow(i);
            ABuf[iLoc+jLoc*ALDim] += alpha;
        }
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void ShiftDiagonal \
  ( Matrix<T>& A, T alpha, Int offset ); \
  EL_EXTERN template void ShiftDiagonal \
  ( AbstractDistMatrix<T>& A, T alpha, Int offset );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_SHIFTDIAGONAL_HPP

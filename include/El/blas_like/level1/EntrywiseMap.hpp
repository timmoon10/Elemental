/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ENTRYWISEMAP_HPP
#define EL_BLAS_ENTRYWISEMAP_HPP

namespace El {

template<typename T>
void EntrywiseMap( Matrix<T>& A, function<T(const T&)> func )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    // Iterate over single loop if memory is contiguous. Otherwise
    // iterate over double loop.
    if( ALDim == m )
    {
        EL_PARALLEL_FOR
        for( Int i=0; i<m*n; ++i )
        {
            ABuf[i] = func(ABuf[i]);
        }
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<n; ++j )
        {
            EL_SIMD
            for( Int i=0; i<m; ++i )
            {
                ABuf[i+j*ALDim] = func(ABuf[i+j*ALDim]);
            }
        }
    }
}

template<typename T>
void EntrywiseMap( AbstractDistMatrix<T>& A, function<T(const T&)> func )
{ EntrywiseMap( A.Matrix(), func ); }

template<typename S,typename T>
void EntrywiseMap
( const Matrix<S>& A, Matrix<T>& B, function<T(const S&)> func )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    B.Resize( m, n );
    const S* ABuf = A.LockedBuffer();
    T* BBuf = B.Buffer();
    const Int ALDim = A.LDim();
    const Int BLDim = B.LDim();
    EL_PARALLEL_FOR
    for( Int j=0; j<n; ++j )
    {
        EL_SIMD
        for( Int i=0; i<m; ++i )
        {
            BBuf[i+j*BLDim] = func(ABuf[i+j*ALDim]);
        }
    }
}


template<typename S,typename T>
void EntrywiseMap
( const AbstractDistMatrix<S>& A,
        AbstractDistMatrix<T>& B,
        function<T(const S&)> func )
{
    if( A.DistData().colDist == B.DistData().colDist &&
        A.DistData().rowDist == B.DistData().rowDist &&
        A.Wrap() == B.Wrap() )
    {
        B.AlignWith( A.DistData() );
        B.Resize( A.Height(), A.Width() );
        EntrywiseMap( A.LockedMatrix(), B.Matrix(), func );
    }
    else
    {
        B.Resize( A.Height(), A.Width() );
        #define GUARD(CDIST,RDIST,WRAP) \
          B.DistData().colDist == CDIST && B.DistData().rowDist == RDIST && \
          B.Wrap() == WRAP
        #define PAYLOAD(CDIST,RDIST,WRAP) \
          DistMatrix<S,CDIST,RDIST,WRAP> AProx(B.Grid()); \
          AProx.AlignWith( B.DistData() ); \
          Copy( A, AProx ); \
          EntrywiseMap( AProx.Matrix(), B.Matrix(), func );
        #include <El/macros/GuardAndPayload.h>
        #undef GUARD
        #undef PAYLOAD
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void EntrywiseMap \
  ( Matrix<T>& A, \
    function<T(const T&)> func ); \
  EL_EXTERN template void EntrywiseMap \
  ( AbstractDistMatrix<T>& A, \
    function<T(const T&)> func ); \
  EL_EXTERN template void EntrywiseMap \
  ( const Matrix<T>& A, \
          Matrix<T>& B, \
          function<T(const T&)> func ); \
  EL_EXTERN template void EntrywiseMap \
  ( const AbstractDistMatrix<T>& A, \
          AbstractDistMatrix<T>& B, \
          function<T(const T&)> func );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ENTRYWISEMAP_HPP

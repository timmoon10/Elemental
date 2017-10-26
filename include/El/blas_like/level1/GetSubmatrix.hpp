/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_GETSUBMATRIX_HPP
#define EL_BLAS_GETSUBMATRIX_HPP

namespace El {

// Contiguous
// ==========
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
        Range<Int> I,
        Range<Int> J,
        Matrix<T>& ASub )
{
    EL_DEBUG_CSE
    auto ASubView = A(I,J);
    ASub = ASubView;
}

template<typename T>
void GetSubmatrix
( const ElementalMatrix<T>& A,
        Range<Int> I,
        Range<Int> J,
        ElementalMatrix<T>& ASub )
{
    EL_DEBUG_CSE
    unique_ptr<ElementalMatrix<T>>
      ASubView( A.Construct(A.Grid(),A.Root()) );
    LockedView( *ASubView, A, I, J );
    Copy( *ASubView, ASub );
}

// Non-contiguous
// ==============
template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
  const Range<Int> I,
  const vector<Int>& J,
        Matrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.end-I.beg;
    const Int nSub = J.size();
    ASub.Resize( mSub, nSub );

    T* ASubBuf = ASub.Buffer();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    const Int ASubLDim = ASub.LDim();

    for( Int jSub=0; jSub<nSub; ++jSub )
    {
        const Int j = J[jSub];
        MemCopy( &ASubBuf[jSub*ASubLDim], &ABuf[j*ALDim], mSub );
    }
}

template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
  const vector<Int>& I,
  const Range<Int> J,
        Matrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.size();
    const Int nSub = J.end-J.beg;
    ASub.Resize( mSub, nSub );

    T* ASubBuf = ASub.Buffer();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    const Int ASubLDim = ASub.LDim();

    for( Int jSub=0; jSub<nSub; ++jSub )
    {
        const Int j = J.beg + jSub;
        for( Int iSub=0; iSub<mSub; ++iSub )
        {
            const Int i = I[iSub];
            ASubBuf[iSub+jSub*ASubLDim] = ABuf[i+j*ALDim];
        }
    }
}

template<typename T>
void GetSubmatrix
( const Matrix<T>& A,
  const vector<Int>& I,
  const vector<Int>& J,
        Matrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.size();
    const Int nSub = J.size();
    ASub.Resize( mSub, nSub );

    T* ASubBuf = ASub.Buffer();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    const Int ASubLDim = ASub.LDim();

    for( Int jSub=0; jSub<nSub; ++jSub )
    {
        const Int j = J[jSub];
        for( Int iSub=0; iSub<mSub; ++iSub )
        {
            const Int i = I[iSub];
            ASubBuf[iSub+jSub*ASubLDim] = ABuf[i+j*ALDim];
        }
    }
}

template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
        Range<Int> I,
  const vector<Int>& J,
        AbstractDistMatrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.end-I.beg;
    const Int nSub = J.size();
    const Grid& g = A.Grid();
    ASub.SetGrid( g );
    ASub.Resize( mSub, nSub );
    Zero( ASub );

    // TODO(poulson): Intelligently pick the redundant rank to pack from?

    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    // Count the number of updates
    // ===========================
    Int numUpdates = 0;
    if( A.RedundantRank() == 0 )
        for( Int i=I.beg; i<I.end; ++i )
            if( A.IsLocalRow(i) )
                for( auto& j : J )
                    if( A.IsLocalCol(j) )
                        ++numUpdates;

    // Queue and process the updates
    // =============================
    ASub.Reserve( numUpdates );
    if( A.RedundantRank() == 0 )
    {
        for( Int iSub=0; iSub<mSub; ++iSub )
        {
            const Int i = I.beg + iSub;
            if( A.IsLocalRow(i) )
            {
                const Int iLoc = A.LocalRow(i);
                for( Int jSub=0; jSub<nSub; ++jSub )
                {
                    const Int j = J[jSub];
                    if( A.IsLocalCol(j) )
                    {
                        const Int jLoc = A.LocalCol(j);
                        ASub.QueueUpdate( iSub, jSub, ABuf[iLoc+jLoc*ALDim] );
                    }
                }
            }
        }
    }
    ASub.ProcessQueues();
}

template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
  const vector<Int>& I,
        Range<Int> J,
        AbstractDistMatrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.size();
    const Int nSub = J.end-J.beg;
    const Grid& g = A.Grid();
    ASub.SetGrid( g );
    ASub.Resize( mSub, nSub );
    Zero( ASub );

    // TODO(poulson): Intelligently pick the redundant rank to pack from?

    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    // Count the number of updates
    // ===========================
    Int numUpdates = 0;
    if( A.RedundantRank() == 0 )
        for( auto& i : I )
            if( A.IsLocalRow(i) )
                for( Int j=J.beg; j<J.end; ++j )
                    if( A.IsLocalCol(j) )
                        ++numUpdates;

    // Queue and process the updates
    // =============================
    ASub.Reserve( numUpdates );
    if( A.RedundantRank() == 0 )
    {
        for( Int iSub=0; iSub<mSub; ++iSub )
        {
            const Int i = I[iSub];
            if( A.IsLocalRow(i) )
            {
                const Int iLoc = A.LocalRow(i);
                for( Int jSub=0; jSub<nSub; ++jSub )
                {
                    const Int j = J.beg + jSub;
                    if( A.IsLocalCol(j) )
                    {
                        const Int jLoc = A.LocalCol(j);
                        ASub.QueueUpdate( iSub, jSub, ABuf[iLoc+jLoc*ALDim] );
                    }
                }
            }
        }
    }
    ASub.ProcessQueues();
}

template<typename T>
void GetSubmatrix
( const AbstractDistMatrix<T>& A,
  const vector<Int>& I,
  const vector<Int>& J,
        AbstractDistMatrix<T>& ASub )
{
    EL_DEBUG_CSE
    const Int mSub = I.size();
    const Int nSub = J.size();
    const Grid& g = A.Grid();
    ASub.SetGrid( g );
    ASub.Resize( mSub, nSub );
    Zero( ASub );

    // TODO(poulson): Intelligently pick the redundant rank to pack from?

    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    // Count the number of updates
    // ===========================
    Int numUpdates = 0;
    if( A.RedundantRank() == 0 )
        for( auto& i : I )
            if( A.IsLocalRow(i) )
                for( auto& j : J )
                    if( A.IsLocalCol(j) )
                        ++numUpdates;

    // Queue and process the updates
    // =============================
    ASub.Reserve( numUpdates );
    if( A.RedundantRank() == 0 )
    {
        for( Int iSub=0; iSub<mSub; ++iSub )
        {
            const Int i = I[iSub];
            if( A.IsLocalRow(i) )
            {
                const Int iLoc = A.LocalRow(i);
                for( Int jSub=0; jSub<nSub; ++jSub )
                {
                    const Int j = J[jSub];
                    if( A.IsLocalCol(j) )
                    {
                        const Int jLoc = A.LocalCol(j);
                        ASub.QueueUpdate( iSub, jSub, ABuf[iLoc+jLoc*ALDim] );
                    }
                }
            }
        }
    }
    ASub.ProcessQueues();
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void GetSubmatrix \
  ( const Matrix<T>& A, \
          Range<Int> I, \
          Range<Int> J,  \
          Matrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const ElementalMatrix<T>& A, \
          Range<Int> I, \
          Range<Int> J, \
          ElementalMatrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const Matrix<T>& A, \
    const Range<Int> I, \
    const vector<Int>& J,  \
          Matrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const Matrix<T>& A, \
    const vector<Int>& I, \
    const Range<Int> J, \
          Matrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const Matrix<T>& A, \
    const vector<Int>& I, \
    const vector<Int>& J, \
          Matrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const AbstractDistMatrix<T>& A, \
          Range<Int> I, \
    const vector<Int>& J, \
          AbstractDistMatrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const AbstractDistMatrix<T>& A, \
    const vector<Int>& I, \
          Range<Int> J, \
          AbstractDistMatrix<T>& ASub ); \
  EL_EXTERN template void GetSubmatrix \
  ( const AbstractDistMatrix<T>& A, \
    const vector<Int>& I, \
    const vector<Int>& J, \
          AbstractDistMatrix<T>& ASub );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_GETSUBMATRIX_HPP

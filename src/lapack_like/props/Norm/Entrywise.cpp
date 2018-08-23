/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename Field>
Base<Field> EntrywiseNorm( const AbstractMatrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    if(A.GetDevice() != Device::CPU)
      LogicError("EntrywiseNorm: Unsupported function for non-CPU Matrix");

    // TODO(poulson): Make this more numerically stable
    typedef Base<Field> Real;
    Real sum = 0;
    const Int width = A.Width();
    const Int height = A.Height();
    for( Int j=0; j<width; ++j )
        for( Int i=0; i<height; ++i )
            sum += Pow( Abs(A(i,j)), p );
    return Pow( sum, 1/p );
}

template<typename Field>
Base<Field> HermitianEntrywiseNorm
( UpperOrLower uplo, const Matrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square.");

    // TODO(poulson): make this more numerically stable
    typedef Base<Field> Real;
    Real sum = 0;
    const Int height = A.Height();
    const Int width = A.Width();
    if( uplo == UPPER )
    {
        for( Int j=0; j<width; ++j )
        {
            for( Int i=0; i<j; ++i )
            {
                const Real term = Pow( Abs(A(i,j)), p );
                if( i ==j )
                    sum += term;
                else
                    sum += 2*term;
            }
        }
    }
    else
    {
        for( Int j=0; j<width; ++j )
        {
            for( Int i=j+1; i<height; ++i )
            {
                const Real term = Pow( Abs(A(i,j)), p );
                if( i ==j )
                    sum += term;
                else
                    sum += 2*term;
            }
        }
    }
    return Pow( sum, 1/p );
}

template<typename Field>
Base<Field> SymmetricEntrywiseNorm
( UpperOrLower uplo, const Matrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    return HermitianEntrywiseNorm( uplo, A, p );
}

template<typename Field>
Base<Field> EntrywiseNorm( const AbstractDistMatrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    Real norm;
    if( A.Participating() )
    {
        Real localSum = 0;
        const Int localHeight = A.LocalHeight();
        const Int localWidth = A.LocalWidth();
        const Matrix<Field>& ALoc = A.LockedMatrix();
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                localSum += Pow( Abs(ALoc(iLoc,jLoc)), p );
        const Real sum = mpi::AllReduce( localSum, A.DistComm(),
                                         SyncInfo<Device::CPU>() );
        norm = Pow( sum, 1/p );
    }
    mpi::Broadcast( norm, A.Root(), A.CrossComm() );
    return norm;
}

template<typename Field>
Base<Field> HermitianEntrywiseNorm
( UpperOrLower uplo, const AbstractDistMatrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square.");

    typedef Base<Field> Real;
    Real sum;
    if( A.Participating() )
    {
        Real localSum = 0;
        const Int localWidth = A.LocalWidth();
        const Matrix<Field>& ALoc = A.LockedMatrix();
        if( uplo == UPPER )
        {
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numUpperRows = A.LocalRowOffset(j+1);
                for( Int iLoc=0; iLoc<numUpperRows; ++iLoc )
                {
                    const Int i = A.GlobalRow(iLoc);
                    const Real term = Pow( Abs(ALoc(iLoc,jLoc)), p );
                    if( i ==j )
                        localSum += term;
                    else
                        localSum += 2*term;
                }
            }
        }
        else
        {
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int numStrictlyUpperRows = A.LocalRowOffset(j);
                for( Int iLoc=numStrictlyUpperRows;
                     iLoc<A.LocalHeight(); ++iLoc )
                {
                    const Int i = A.GlobalRow(iLoc);
                    const Real term = Pow( Abs(ALoc(iLoc,jLoc)), p );
                    if( i ==j )
                        localSum += term;
                    else
                        localSum += 2*term;
                }
            }
        }
        sum = mpi::AllReduce(
            localSum, A.DistComm(), SyncInfo<Device::CPU>() );
    }
    mpi::Broadcast( sum, A.Root(), A.CrossComm() );
    return Pow( sum, 1/p );
}

template<typename Field>
Base<Field> SymmetricEntrywiseNorm
( UpperOrLower uplo, const AbstractDistMatrix<Field>& A, Base<Field> p )
{
    EL_DEBUG_CSE
    return HermitianEntrywiseNorm( uplo, A, p );
}

#define PROTO(Field) \
  template Base<Field> EntrywiseNorm( const AbstractMatrix<Field>& A, Base<Field> p ); \
  template Base<Field> \
  EntrywiseNorm( const AbstractDistMatrix<Field>& A, Base<Field> p ); \
  template Base<Field> HermitianEntrywiseNorm \
  ( UpperOrLower uplo, const Matrix<Field>& A, Base<Field> p ); \
  template Base<Field> HermitianEntrywiseNorm \
  ( UpperOrLower uplo, const AbstractDistMatrix<Field>& A, Base<Field> p ); \
  template Base<Field> SymmetricEntrywiseNorm \
  ( UpperOrLower uplo, const Matrix<Field>& A, Base<Field> p ); \
  template Base<Field> SymmetricEntrywiseNorm \
  ( UpperOrLower uplo, const AbstractDistMatrix<Field>& A, Base<Field> p );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

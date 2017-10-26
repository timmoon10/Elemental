/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include "./NormsFromScaledSquares.hpp"

namespace El {

template<typename Field>
void ColumnTwoNormsHelper
( const Matrix<Field>& ALoc, Matrix<Base<Field>>& normsLoc, mpi::Comm comm )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Int mLocal = ALoc.Height();
    const Int nLocal = ALoc.Width();

    // TODO(poulson): Ensure that NaN's propagate
    Matrix<Real> localScales( nLocal, 1 ),
                 localScaledSquares( nLocal, 1 );
    for( Int jLoc=0; jLoc<nLocal; ++jLoc )
    {
        Real localScale = 0;
        Real localScaledSquare = 1;
        for( Int iLoc=0; iLoc<mLocal; ++iLoc )
            UpdateScaledSquare
            ( ALoc(iLoc,jLoc), localScale, localScaledSquare );

        localScales(jLoc) = localScale;
        localScaledSquares(jLoc) = localScaledSquare;
    }

    NormsFromScaledSquares( localScales, localScaledSquares, normsLoc, comm );
}

template<typename Real>
void ColumnTwoNormsHelper
( const Matrix<Real>& ARealLoc,
  const Matrix<Real>& AImagLoc,
        Matrix<Real>& normsLoc, mpi::Comm comm )
{
    EL_DEBUG_CSE
    const Int mLocal = ARealLoc.Height();
    const Int nLocal = ARealLoc.Width();

    // TODO(poulson): Ensure that NaN's propagate
    Matrix<Real> localScales( nLocal, 1 ), localScaledSquares( nLocal, 1 );
    for( Int jLoc=0; jLoc<nLocal; ++jLoc )
    {
        Real localScale = 0;
        Real localScaledSquare = 1;
        for( Int iLoc=0; iLoc<mLocal; ++iLoc )
            UpdateScaledSquare
            ( ARealLoc(iLoc,jLoc), localScale, localScaledSquare );
        for( Int iLoc=0; iLoc<mLocal; ++iLoc )
            UpdateScaledSquare
            ( AImagLoc(iLoc,jLoc), localScale, localScaledSquare );

        localScales(jLoc) = localScale;
        localScaledSquares(jLoc) = localScaledSquare;
    }

    NormsFromScaledSquares( localScales, localScaledSquares, normsLoc, comm );
}

template<typename Field>
void ColumnTwoNorms( const Matrix<Field>& X, Matrix<Base<Field>>& norms )
{
    EL_DEBUG_CSE
    const Int m = X.Height();
    const Int n = X.Width();
    norms.Resize( n, 1 );
    if( m == 0 )
    {
        Zero( norms );
        return;
    }
    for( Int j=0; j<n; ++j )
        norms(j) = blas::Nrm2( m, &X(0,j), 1 );
}

template<typename Field>
void ColumnMaxNorms( const Matrix<Field>& X, Matrix<Base<Field>>& norms )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Int m = X.Height();
    const Int n = X.Width();
    norms.Resize( n, 1 );
    for( Int j=0; j<n; ++j )
    {
        // TODO(poulson): Ensure that NaN's propagate
        Real colMax = 0;
        for( Int i=0; i<m; ++i )
            colMax = Max(colMax,Abs(X(i,j)));
        norms(j) = colMax;
    }
}

template<typename Field,Dist U,Dist V,DistWrap W>
void ColumnTwoNorms
( const DistMatrix<Field,U,V,W>& A, DistMatrix<Base<Field>,V,STAR,W>& norms )
{
    EL_DEBUG_CSE
    norms.AlignWith( A );
    norms.Resize( A.Width(), 1 );
    if( A.Height() == 0 )
    {
        Zero( norms );
        return;
    }
    ColumnTwoNormsHelper( A.LockedMatrix(), norms.Matrix(), A.ColComm() );
}

template<typename Field,Dist U,Dist V,DistWrap W>
void ColumnMaxNorms
( const DistMatrix<Field,U,V,W>& A, DistMatrix<Base<Field>,V,STAR,W>& norms )
{
    EL_DEBUG_CSE
    norms.AlignWith( A );
    norms.Resize( A.Width(), 1 );
    ColumnMaxNorms( A.LockedMatrix(), norms.Matrix() );
    AllReduce( norms.Matrix(), A.ColComm(), mpi::MAX );
}

// Versions which operate on explicitly-separated complex matrices
// ===============================================================
template<typename Real,typename>
void ColumnTwoNorms
( const Matrix<Real>& XReal,
  const Matrix<Real>& XImag,
        Matrix<Real>& norms )
{
    EL_DEBUG_CSE
    const Int m = XReal.Height();
    const Int n = XReal.Width();
    norms.Resize( n, 1 );
    if( m == 0 )
    {
        Zero( norms );
        return;
    }
    for( Int j=0; j<n; ++j )
    {
        Real alpha = blas::Nrm2( m, &XReal(0,j), 1 );
        Real beta  = blas::Nrm2( m, &XImag(0,j), 1 );
        norms(j) = SafeNorm(alpha,beta);
    }
}

template<typename Real,Dist U,Dist V,typename>
void ColumnTwoNorms
( const DistMatrix<Real,U,V>& XReal,
  const DistMatrix<Real,U,V>& XImag,
        DistMatrix<Real,V,STAR>& norms )
{
    EL_DEBUG_CSE
    if( XReal.RowAlign() != norms.ColAlign() )
        LogicError("Invalid norms alignment");
    norms.Resize( XReal.Width(), 1 );
    if( XReal.Height() == 0 )
    {
        Zero( norms );
        return;
    }
    ColumnTwoNormsHelper
    ( XReal.LockedMatrix(),
      XImag.LockedMatrix(),
      norms.Matrix(),
      XReal.ColComm() );
}

#define PROTO_DIST(Field,U,V) \
  template void ColumnTwoNorms \
  ( const DistMatrix<Field,U,V,ELEMENT>& X, \
          DistMatrix<Base<Field>,V,STAR,ELEMENT>& norms ); \
  template void ColumnMaxNorms \
  ( const DistMatrix<Field,U,V,ELEMENT>& X, \
          DistMatrix<Base<Field>,V,STAR,ELEMENT>& norms ); \
  template void ColumnTwoNorms \
  ( const DistMatrix<Field,U,V,BLOCK>& X, \
          DistMatrix<Base<Field>,V,STAR,BLOCK>& norms ); \
  template void ColumnMaxNorms \
  ( const DistMatrix<Field,U,V,BLOCK>& X, \
          DistMatrix<Base<Field>,V,STAR,BLOCK>& norms );

#define PROTO(Field) \
  template void ColumnTwoNorms \
  ( const Matrix<Field>& X, \
          Matrix<Base<Field>>& norms ); \
  template void ColumnMaxNorms \
  ( const Matrix<Field>& X, \
          Matrix<Base<Field>>& norms ); \
  PROTO_DIST(Field,MC,  MR  ) \
  PROTO_DIST(Field,MC,  STAR) \
  PROTO_DIST(Field,MD,  STAR) \
  PROTO_DIST(Field,MR,  MC  ) \
  PROTO_DIST(Field,MR,  STAR) \
  PROTO_DIST(Field,STAR,MC  ) \
  PROTO_DIST(Field,STAR,MD  ) \
  PROTO_DIST(Field,STAR,MR  ) \
  PROTO_DIST(Field,STAR,STAR) \
  PROTO_DIST(Field,STAR,VC  ) \
  PROTO_DIST(Field,STAR,VR  ) \
  PROTO_DIST(Field,VC,  STAR) \
  PROTO_DIST(Field,VR,  STAR)

#define PROTO_REAL_DIST(Real,U,V) \
  template void ColumnTwoNorms \
  ( const DistMatrix<Real,U,V>& XReal, \
    const DistMatrix<Real,U,V>& XImag, \
          DistMatrix<Real,V,STAR>& norms );

#define PROTO_REAL(Real) \
  PROTO(Real) \
  template void ColumnTwoNorms \
  ( const Matrix<Real>& XReal, \
    const Matrix<Real>& XImag, \
          Matrix<Real>& norms ); \
  PROTO_REAL_DIST(Real,MC,  MR  ) \
  PROTO_REAL_DIST(Real,MC,  STAR) \
  PROTO_REAL_DIST(Real,MD,  STAR) \
  PROTO_REAL_DIST(Real,MR,  MC  ) \
  PROTO_REAL_DIST(Real,MR,  STAR) \
  PROTO_REAL_DIST(Real,STAR,MC  ) \
  PROTO_REAL_DIST(Real,STAR,MD  ) \
  PROTO_REAL_DIST(Real,STAR,MR  ) \
  PROTO_REAL_DIST(Real,STAR,STAR) \
  PROTO_REAL_DIST(Real,STAR,VC  ) \
  PROTO_REAL_DIST(Real,STAR,VR  ) \
  PROTO_REAL_DIST(Real,VC,  STAR) \
  PROTO_REAL_DIST(Real,VR,  STAR)

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

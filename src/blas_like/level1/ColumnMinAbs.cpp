/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>

namespace El {

template<typename Ring>
void ColumnMinAbs( const Matrix<Ring>& X, Matrix<Base<Ring>>& mins )
{
    EL_DEBUG_CSE
    typedef Base<Ring> RealRing;
    const Int m = X.Height();
    const Int n = X.Width();
    mins.Resize( n, 1 );
    for( Int j=0; j<n; ++j )
    {
        RealRing colMin = limits::Max<RealRing>();
        for( Int i=0; i<m; ++i )
            colMin = Min(colMin,Abs(X(i,j)));
        mins(j) = colMin;
    }
}

template<typename Ring>
void ColumnMinAbsNonzero
( const Matrix<Ring>& X,
  const Matrix<Base<Ring>>& upperBounds,
        Matrix<Base<Ring>>& mins )
{
    EL_DEBUG_CSE
    typedef Base<Ring> RealRing;
    const Int m = X.Height();
    const Int n = X.Width();
    mins.Resize( n, 1 );
    for( Int j=0; j<n; ++j )
    {
        RealRing colMin = upperBounds(j);
        for( Int i=0; i<m; ++i )
        {
            RealRing absVal = Abs(X(i,j));
            if( absVal > RealRing(0) )
                colMin = Min(colMin,absVal);
        }
        mins(j) = colMin;
    }
}

template<typename Ring,Dist U,Dist V>
void ColumnMinAbs
( const DistMatrix<Ring,U,V>& A, DistMatrix<Base<Ring>,V,STAR>& mins )
{
    EL_DEBUG_CSE
    const Int n = A.Width();
    mins.AlignWith( A );
    mins.Resize( n, 1 );
    ColumnMinAbs( A.LockedMatrix(), mins.Matrix() );
    AllReduce( mins.Matrix(), A.ColComm(), mpi::MIN );
}

template<typename Ring,Dist U,Dist V>
void ColumnMinAbsNonzero
( const DistMatrix<Ring,U,V>& A,
  const DistMatrix<Base<Ring>,V,STAR>& upperBounds,
        DistMatrix<Base<Ring>,V,STAR>& mins )
{
    EL_DEBUG_CSE
    if( upperBounds.ColAlign() != A.RowAlign() )
        LogicError("upperBounds was not properly aligned");
    const Int n = A.Width();
    mins.AlignWith( A );
    mins.Resize( n, 1 );
    ColumnMinAbsNonzero
    ( A.LockedMatrix(), upperBounds.LockedMatrix(), mins.Matrix() );
    AllReduce( mins.Matrix(), A.ColComm(), mpi::MIN );
}


#define PROTO_DIST(Ring,U,V) \
  template void ColumnMinAbs \
  ( const DistMatrix<Ring,U,V>& X, DistMatrix<Base<Ring>,V,STAR>& mins ); \
  template void ColumnMinAbsNonzero \
  ( const DistMatrix<Ring,U,V>& X, \
    const DistMatrix<Base<Ring>,V,STAR>& upperBounds, \
          DistMatrix<Base<Ring>,V,STAR>& mins );

#define PROTO(Ring) \
  template void ColumnMinAbs \
  ( const Matrix<Ring>& X, Matrix<Base<Ring>>& mins ); \
  template void ColumnMinAbsNonzero \
  ( const Matrix<Ring>& X, \
    const Matrix<Base<Ring>>& upperBounds, \
          Matrix<Base<Ring>>& mins ); \
  PROTO_DIST(Ring,MC,  MR  ) \
  PROTO_DIST(Ring,MC,  STAR) \
  PROTO_DIST(Ring,MD,  STAR) \
  PROTO_DIST(Ring,MR,  MC  ) \
  PROTO_DIST(Ring,MR,  STAR) \
  PROTO_DIST(Ring,STAR,MC  ) \
  PROTO_DIST(Ring,STAR,MD  ) \
  PROTO_DIST(Ring,STAR,MR  ) \
  PROTO_DIST(Ring,STAR,STAR) \
  PROTO_DIST(Ring,STAR,VC  ) \
  PROTO_DIST(Ring,STAR,VR  ) \
  PROTO_DIST(Ring,VC,  STAR) \
  PROTO_DIST(Ring,VR,  STAR)

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

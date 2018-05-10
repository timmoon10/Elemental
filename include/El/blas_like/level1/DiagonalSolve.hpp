/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_DIAGONALSOLVE_HPP
#define EL_BLAS_DIAGONALSOLVE_HPP

namespace El {

template<typename FDiag,typename F>
void DiagonalSolve
( LeftOrRight side,
  Orientation orientation,
  const Matrix<FDiag>& d,
        Matrix<F>& A,
  bool checkIfSingular )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const bool conj = ( orientation == ADJOINT );
    if( side == LEFT )
    {
        EL_DEBUG_ONLY(
          if( d.Height() != m )
              LogicError("Invalid left diagonal solve dimension");
        )
        for( Int i=0; i<m; ++i )
        {
            const F delta = ( conj ? Conj(d(i)) : d(i) );
            if( checkIfSingular && delta == F(0) )
                throw SingularMatrixException();
            const F deltaInv = F(1)/delta;
            for( Int j=0; j<n; ++j )
                A(i,j) *= deltaInv;
        }
    }
    else
    {
        EL_DEBUG_ONLY(
          if( d.Height() != n )
              LogicError("Invalid right diagonal solve dimension");
        )
        for( Int j=0; j<n; ++j )
        {
            const F delta = ( conj ? Conj(d(j)) : d(j) );
            if( checkIfSingular && delta == F(0) )
                throw SingularMatrixException();
            const F deltaInv = F(1)/delta;
            for( Int i=0; i<m; ++i )
                A(i,j) *= deltaInv;
        }
    }
}

template<typename F>
void SymmetricDiagonalSolve
( const Matrix<Base<F>>& d,
        Matrix<F>& A )
{
    EL_DEBUG_CSE
    const Int n = A.Width();
    EL_DEBUG_ONLY(
      if( d.Height() != n )
          LogicError("Invalid symmetric diagonal solve dimension");
    )
    for( Int j=0; j<n; ++j )
        for( Int i=0; i<n; ++i )
            A(i,j) /= d(i)*d(j);
}

template<typename FDiag,typename F,Dist U,Dist V>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FDiag>& dPre,
        DistMatrix<F,U,V>& A,
  bool checkIfSingular )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( dPre, A );
    )
    if( side == LEFT )
    {
        ElementalProxyCtrl ctrl;
        ctrl.rootConstrain = true;
        ctrl.colConstrain = true;
        ctrl.root = A.Root();
        ctrl.colAlign = A.ColAlign();

        DistMatrixReadProxy<FDiag,FDiag,U,Collect<V>()> dProx( dPre, ctrl );
        auto& d = dProx.GetLocked();

        DiagonalSolve
        ( LEFT, orientation, d.LockedMatrix(), A.Matrix(), checkIfSingular );
    }
    else
    {
        ElementalProxyCtrl ctrl;
        ctrl.rootConstrain = true;
        ctrl.colConstrain = true;
        ctrl.root = A.Root();
        ctrl.colAlign = A.RowAlign();

        DistMatrixReadProxy<FDiag,FDiag,V,Collect<U>()> dProx( dPre, ctrl );
        auto& d = dProx.GetLocked();

        DiagonalSolve
        ( RIGHT, orientation, d.LockedMatrix(), A.Matrix(), checkIfSingular );
    }
}

template<typename FDiag,typename F,Dist U,Dist V>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FDiag>& dPre,
        DistMatrix<F,U,V,BLOCK>& A,
  bool checkIfSingular )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( dPre, A );
    )
    if( side == LEFT )
    {
        ProxyCtrl ctrl;
        ctrl.rootConstrain = true;
        ctrl.colConstrain = true;
        ctrl.root = A.Root();
        ctrl.colAlign = A.ColAlign();
        ctrl.blockHeight = A.BlockHeight();
        ctrl.colCut = A.ColCut();

        DistMatrixReadProxy<FDiag,FDiag,U,Collect<V>(),BLOCK>
          dProx( dPre, ctrl );
        auto& d = dProx.GetLocked();

        DiagonalSolve
        ( LEFT, orientation, d.LockedMatrix(), A.Matrix(), checkIfSingular );
    }
    else
    {
        ProxyCtrl ctrl;
        ctrl.rootConstrain = true;
        ctrl.colConstrain = true;
        ctrl.root = A.Root();
        ctrl.colAlign = A.RowAlign();
        ctrl.blockHeight = A.BlockWidth();
        ctrl.colCut = A.RowCut();

        DistMatrixReadProxy<FDiag,FDiag,V,Collect<U>(),BLOCK>
          dProx( dPre, ctrl );
        auto& d = dProx.GetLocked();

        DiagonalSolve
        ( RIGHT, orientation, d.LockedMatrix(), A.Matrix(), checkIfSingular );
    }
}

template<typename FDiag,typename F>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FDiag>& d,
        AbstractDistMatrix<F>& A,
  bool checkIfSingular )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP,DEVICE) \
      A.ColDist() == CDIST && A.RowDist() == RDIST && A.Wrap() == WRAP && A.GetLocalDevice() == DEVICE
    #define PAYLOAD(CDIST,RDIST,WRAP,DEVICE) \
        auto& ACast = static_cast<DistMatrix<F,CDIST,RDIST,WRAP,DEVICE>&>(A); \
        DiagonalSolve( side, orientation, d, ACast, checkIfSingular );
    #include <El/macros/DeviceGuardAndPayload.h>
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(F) \
  EL_EXTERN template void DiagonalSolve \
  ( LeftOrRight side, \
    Orientation orientation, \
    const Matrix<F>& d, \
          Matrix<F>& A, \
    bool checkIfSingular ); \
  EL_EXTERN template void SymmetricDiagonalSolve \
  ( const Matrix<Base<F>>& d, \
          Matrix<F>& A ); \
  EL_EXTERN template void DiagonalSolve \
  ( LeftOrRight side, \
    Orientation orientation, \
    const AbstractDistMatrix<F>& d, \
          AbstractDistMatrix<F>& A, \
    bool checkIfSingular );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_DIAGONALSOLVE_HPP

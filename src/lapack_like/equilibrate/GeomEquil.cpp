/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
#include "./Util.hpp"

// The following routines are adaptations of the approach uses by
// Saunders et al. (originally recommended by Joseph Fourer) for iteratively
// rescaling the columns and rows by their approximate geometric means in order
// to better scale the original problem. After this iteration is finished,
// the columns or rows are rescaled so that their maximum entry has magnitude
// one (unless the row/column is identically zero).
//
// The implementation of Saunders et al. is commonly referred to as either
// gmscale.m or gmscal.m.

namespace El {

// TODO(poulson): Make this consistent with ConeGeomEquil

template<typename Field>
void GeomEquil
( Matrix<Field>& A,
  Matrix<Base<Field>>& dRow,
  Matrix<Base<Field>>& dCol,
  bool progress )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Int m = A.Height();
    const Int n = A.Width();
    Ones( dRow, m, 1 );
    Ones( dCol, n, 1 );

    // TODO(poulson): Expose these as control parameters
    const Int minIter = 3;
    const Int maxIter = 6;
    const Real damp = Real(1)/Real(1000);
    const Real relTol = Real(9)/Real(10);

    // Compute the original ratio of the maximum to minimum nonzero
    auto maxAbs = MaxAbsLoc( A );
    const Real maxAbsVal = maxAbs.value;
    if( maxAbsVal == Real(0) )
        return;
    const Real minAbsVal = MinAbsNonzero( A, maxAbsVal );
    Real ratio = maxAbsVal / minAbsVal;
    if( progress )
        Output("Original ratio is ",maxAbsVal,"/",minAbsVal,"=",ratio);

    const Real sqrtDamp = Sqrt(damp);
    const Int indent = PushIndent();
    for( Int iter=0; iter<maxIter; ++iter )
    {
        // Geometrically equilibrate the columns
        for( Int j=0; j<n; ++j )
        {
            auto aCol = A( ALL, IR(j) );
            auto maxColAbs = VectorMaxAbsLoc( aCol );
            const Real maxColAbsVal = maxColAbs.value;
            if( maxColAbsVal > Real(0) )
            {
                const Real minColAbsVal = MinAbsNonzero( aCol, maxColAbsVal );
                const Real propScale = Sqrt(minColAbsVal*maxColAbsVal);
                const Real scale = Max(propScale,sqrtDamp*maxColAbsVal);
                aCol *= 1/scale;
                dCol(j) *= scale;
            }
        }

        // Geometrically equilibrate the rows
        for( Int i=0; i<m; ++i )
        {
            auto aRow = A( IR(i), ALL );
            auto maxRowAbs = VectorMaxAbsLoc( aRow );
            const Real maxRowAbsVal = maxRowAbs.value;
            if( maxRowAbsVal > Real(0) )
            {
                const Real minRowAbsVal = MinAbsNonzero( aRow, maxRowAbsVal );
                const Real propScale = Sqrt(minRowAbsVal*maxRowAbsVal);
                const Real scale = Max(propScale,sqrtDamp*maxRowAbsVal);
                aRow *= 1/scale;
                dRow(i) *= scale;
            }
        }

        auto newMaxAbs = MaxAbsLoc( A );
        const Real newMaxAbsVal = newMaxAbs.value;
        const Real newMinAbsVal = MinAbsNonzero( A, newMaxAbsVal );
        const Real newRatio = newMaxAbsVal / newMinAbsVal;
        if( progress )
            Output("New ratio is ",newMaxAbsVal,"/",newMinAbsVal,"=",newRatio);
        if( iter >= minIter && newRatio >= ratio*relTol )
            break;
        ratio = newRatio;
    }
    SetIndent( indent );

    // Scale each column so that its maximum entry is 1 or 0
    for( Int j=0; j<n; ++j )
    {
        auto aCol = A( ALL, IR(j) );
        auto maxColAbs = VectorMaxAbsLoc( aCol );
        const Real maxColAbsVal = maxColAbs.value;
        if( maxColAbsVal > Real(0) )
        {
            aCol *= 1/maxColAbsVal;
            dCol(j) *= maxColAbsVal;
        }
    }
}

template<typename Field>
void StackedGeomEquil
( Matrix<Field>& A,
  Matrix<Field>& B,
  Matrix<Base<Field>>& dRowA,
  Matrix<Base<Field>>& dRowB,
  Matrix<Base<Field>>& dCol,
  bool progress )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Int mA = A.Height();
    const Int mB = B.Height();
    const Int n = A.Width();
    Ones( dRowA, mA, 1 );
    Ones( dRowB, mB, 1 );
    Ones( dCol, n, 1 );

    // TODO(poulson): Expose these as control parameters
    const Int minIter = 3;
    const Int maxIter = 6;
    const Real damp = Real(1)/Real(1000);
    const Real relTol = Real(9)/Real(10);

    // Compute the original ratio of the maximum to minimum nonzero
    auto maxAbsA = MaxAbsLoc( A );
    auto maxAbsB = MaxAbsLoc( B );
    const Real maxAbsVal = Max(maxAbsA.value,maxAbsB.value);
    if( maxAbsVal == Real(0) )
        return;
    const Real minAbsValA = MinAbsNonzero( A, maxAbsVal );
    const Real minAbsValB = MinAbsNonzero( B, maxAbsVal );
    const Real minAbsVal = Min(minAbsValA,minAbsValB);
    Real ratio = maxAbsVal / minAbsVal;
    if( progress )
        Output("Original ratio is ",maxAbsVal,"/",minAbsVal,"=",ratio);

    const Real sqrtDamp = Sqrt(damp);
    const Int indent = PushIndent();
    for( Int iter=0; iter<maxIter; ++iter )
    {
        // Geometrically equilibrate the columns
        for( Int j=0; j<n; ++j )
        {
            auto aCol = A( ALL, IR(j) );
            auto bCol = B( ALL, IR(j) );
            auto maxColAbsA = VectorMaxAbsLoc( aCol );
            auto maxColAbsB = VectorMaxAbsLoc( bCol );
            const Real maxColAbsVal = Max(maxColAbsA.value,maxColAbsB.value);
            if( maxColAbsVal > Real(0) )
            {
                const Real minColAbsAVal = MinAbsNonzero( aCol, maxColAbsVal );
                const Real minColAbsBVal = MinAbsNonzero( bCol, maxColAbsVal );
                const Real minColAbsVal = Min(minColAbsAVal,minColAbsBVal);
                const Real propScale = Sqrt(minColAbsVal*maxColAbsVal);
                const Real scale = Max(propScale,sqrtDamp*maxColAbsVal);
                aCol *= 1/scale;
                bCol *= 1/scale;
                dCol(j) *= scale;
            }
        }

        // Geometrically equilibrate the rows
        for( Int i=0; i<mA; ++i )
        {
            auto aRow = A( IR(i), ALL );
            auto maxRowAbs = VectorMaxAbsLoc( aRow );
            const Real maxRowAbsVal = maxRowAbs.value;
            if( maxRowAbsVal > Real(0) )
            {
                const Real minRowAbsVal = MinAbsNonzero( aRow, maxRowAbsVal );
                const Real propScale = Sqrt(minRowAbsVal*maxRowAbsVal);
                const Real scale = Max(propScale,sqrtDamp*maxRowAbsVal);
                aRow *= 1/scale;
                dRowA(i) *= scale;
            }
        }
        for( Int i=0; i<mB; ++i )
        {
            auto bRow = B( IR(i), ALL );
            auto maxRowAbs = VectorMaxAbsLoc( bRow );
            const Real maxRowAbsVal = maxRowAbs.value;
            if( maxRowAbsVal > Real(0) )
            {
                const Real minRowAbsVal = MinAbsNonzero( bRow, maxRowAbsVal );
                const Real propScale = Sqrt(minRowAbsVal*maxRowAbsVal);
                const Real scale = Max(propScale,sqrtDamp*maxRowAbsVal);
                bRow *= 1/scale;
                dRowB(i) *= scale;
            }
        }

        auto newMaxAbsA = MaxAbsLoc( A );
        auto newMaxAbsB = MaxAbsLoc( B );
        const Real newMaxAbsVal = Max(newMaxAbsA.value,newMaxAbsB.value);
        const Real newMinAbsValA = MinAbsNonzero( A, newMaxAbsVal );
        const Real newMinAbsValB = MinAbsNonzero( B, newMaxAbsVal );
        const Real newMinAbsVal = Min(newMinAbsValA,newMinAbsValB);
        const Real newRatio = newMaxAbsVal / newMinAbsVal;
        if( progress )
            Output("New ratio is ",newMaxAbsVal,"/",newMinAbsVal,"=",newRatio);
        if( iter >= minIter && newRatio >= ratio*relTol )
            break;
        ratio = newRatio;
    }
    SetIndent( indent );

    // Scale each column so that its maximum entry is 1 or 0
    for( Int j=0; j<n; ++j )
    {
        auto aCol = A( ALL, IR(j) );
        auto bCol = B( ALL, IR(j) );
        auto maxColAbsA = VectorMaxAbsLoc( aCol );
        auto maxColAbsB = VectorMaxAbsLoc( bCol );
        const Real maxColAbsVal = Max(maxColAbsA.value,maxColAbsB.value);
        if( maxColAbsVal > Real(0) )
        {
            aCol *= 1/maxColAbsVal;
            bCol *= 1/maxColAbsVal;
            dCol(j) *= maxColAbsVal;
        }
    }
}

template<typename Field>
void GeomEquil
( AbstractDistMatrix<Field>& APre,
  AbstractDistMatrix<Base<Field>>& dRowPre,
  AbstractDistMatrix<Base<Field>>& dColPre,
  bool progress )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;

    ElementalProxyCtrl control;
    control.colConstrain = true;
    control.rowConstrain = true;
    control.colAlign = 0;
    control.rowAlign = 0;

    DistMatrixReadWriteProxy<Field,Field,MC,MR> AProx( APre, control );
    DistMatrixWriteProxy<Real,Real,MC,STAR> dRowProx( dRowPre, control );
    DistMatrixWriteProxy<Real,Real,MR,STAR> dColProx( dColPre, control );
    auto& A = AProx.Get();
    auto& dRow = dRowProx.Get();
    auto& dCol = dColProx.Get();

    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocal = A.LocalHeight();
    const Int nLocal = A.LocalWidth();
    Ones( dRow, m, 1 );
    Ones( dCol, n, 1 );

    // TODO(poulson): Expose these as control parameters
    const Int minIter = 3;
    const Int maxIter = 6;
    const Real relTol = Real(9)/Real(10);

    // TODO(poulson): Incorporate damping
    //const Real damp = Real(1)/Real(1000);
    //const Real sqrtDamp = Sqrt(damp);

    // Compute the original ratio of the maximum to minimum nonzero
    auto maxAbs = MaxAbsLoc( A );
    const Real maxAbsVal = maxAbs.value;
    if( maxAbsVal == Real(0) )
        return;
    const Real minAbsVal = MinAbsNonzero( A, maxAbsVal );
    Real ratio = maxAbsVal / minAbsVal;
    if( progress && A.Grid().Rank() == 0 )
        Output("Original ratio is ",maxAbsVal,"/",minAbsVal,"=",ratio);

    DistMatrix<Real,MC,STAR> rowScale(A.Grid());
    DistMatrix<Real,MR,STAR> colScale(A.Grid());
    auto& colScaleLoc = colScale.Matrix();
    auto& rowScaleLoc = rowScale.Matrix();
    const Int indent = PushIndent();
    for( Int iter=0; iter<maxIter; ++iter )
    {
        // Geometrically equilibrate the columns
        // -------------------------------------
        // TODO(poulson): Remove GeometricColumnScaling
        GeometricColumnScaling( A, colScale );
        for( Int jLoc=0; jLoc<nLocal; ++jLoc )
            if( colScaleLoc(jLoc) == Real(0) )
                colScaleLoc(jLoc) = Real(1);
        DiagonalScale( LEFT, NORMAL, colScale, dCol );
        DiagonalSolve( RIGHT, NORMAL, colScale, A );

        // Geometrically equilibrate the rows
        // ----------------------------------
        // TODO(poulson): Remove GeometricRowScaling
        GeometricRowScaling( A, rowScale );
        for( Int iLoc=0; iLoc<mLocal; ++iLoc )
            if( rowScaleLoc(iLoc) == Real(0) )
                rowScaleLoc(iLoc) = Real(1);
        DiagonalScale( LEFT, NORMAL, rowScale, dRow );
        DiagonalSolve( LEFT, NORMAL, rowScale, A );

        auto newMaxAbs = MaxAbsLoc( A );
        const Real newMaxAbsVal = newMaxAbs.value;
        const Real newMinAbsVal = MinAbsNonzero( A, newMaxAbsVal );
        const Real newRatio = newMaxAbsVal / newMinAbsVal;
        if( progress && A.Grid().Rank() == 0 )
            Output("New ratio is ",newMaxAbsVal,"/",newMinAbsVal,"=",newRatio);
        if( iter >= minIter && newRatio >= ratio*relTol )
            break;
        ratio = newRatio;
    }
    SetIndent( indent );

    // Scale each column so that its maximum entry is 1 or 0
    ColumnMaxNorms( A, colScale );
    for( Int jLoc=0; jLoc<nLocal; ++jLoc )
        if( colScaleLoc(jLoc) == Real(0) )
            colScaleLoc(jLoc) = Real(1);
    DiagonalScale( LEFT, NORMAL, colScale, dCol );
    DiagonalSolve( RIGHT, NORMAL, colScale, A );
}

template<typename Field>
void StackedGeomEquil
( AbstractDistMatrix<Field>& APre,
  AbstractDistMatrix<Field>& BPre,
  AbstractDistMatrix<Base<Field>>& dRowAPre,
  AbstractDistMatrix<Base<Field>>& dRowBPre,
  AbstractDistMatrix<Base<Field>>& dColPre,
  bool progress )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;

    ElementalProxyCtrl control;
    control.colConstrain = true;
    control.rowConstrain = true;
    control.colAlign = 0;
    control.rowAlign = 0;

    DistMatrixReadWriteProxy<Field,Field,MC,MR> AProx( APre, control );
    DistMatrixReadWriteProxy<Field,Field,MC,MR> BProx( BPre, control );
    DistMatrixWriteProxy<Real,Real,MC,STAR> dRowAProx( dRowAPre, control );
    DistMatrixWriteProxy<Real,Real,MC,STAR> dRowBProx( dRowBPre, control );
    DistMatrixWriteProxy<Real,Real,MR,STAR> dColProx( dColPre, control );
    auto& A = AProx.Get();
    auto& B = BProx.Get();
    auto& dRowA = dRowAProx.Get();
    auto& dRowB = dRowBProx.Get();
    auto& dCol = dColProx.Get();

    const Int mA = A.Height();
    const Int mB = B.Height();
    const Int n = A.Width();
    const Int mLocalA = A.LocalHeight();
    const Int mLocalB = B.LocalHeight();
    const Int nLocal = A.LocalWidth();
    Ones( dRowA, mA, 1 );
    Ones( dRowB, mB, 1 );
    Ones( dCol, n, 1 );

    // TODO(poulson): Expose these as control parameters
    const Int minIter = 3;
    const Int maxIter = 6;
    const Real relTol = Real(9)/Real(10);

    // TODO(poulson): Incorporate damping
    //const Real damp = Real(1)/Real(1000);
    //const Real sqrtDamp = Sqrt(damp);

    // Compute the original ratio of the maximum to minimum nonzero
    auto maxAbsA = MaxAbsLoc( A );
    auto maxAbsB = MaxAbsLoc( B );
    const Real maxAbsVal = Max(maxAbsA.value,maxAbsB.value);
    if( maxAbsVal == Real(0) )
        return;
    const Real minAbsValA = MinAbsNonzero( A, maxAbsVal );
    const Real minAbsValB = MinAbsNonzero( B, maxAbsVal );
    const Real minAbsVal = Min(minAbsValA,minAbsValB);
    Real ratio = maxAbsVal / minAbsVal;
    if( progress && A.Grid().Rank() == 0 )
        Output("Original ratio is ",maxAbsVal,"/",minAbsVal,"=",ratio);

    DistMatrix<Real,MC,STAR> rowScaleA(A.Grid()),
                             rowScaleB(A.Grid());
    DistMatrix<Real,MR,STAR> colScale(A.Grid()), colScaleB(B.Grid());
    auto& rowScaleALoc = rowScaleA.Matrix();
    auto& rowScaleBLoc = rowScaleB.Matrix();
    auto& colScaleLoc = colScale.Matrix();
    auto& colScaleBLoc = colScaleB.Matrix();
    const Int indent = PushIndent();
    for( Int iter=0; iter<maxIter; ++iter )
    {
        // Geometrically equilibrate the columns
        // -------------------------------------
        // TODO(poulson): Remove StackedGeometricColumnScaling
        StackedGeometricColumnScaling( A, B, colScale );
        for( Int jLoc=0; jLoc<nLocal; ++jLoc )
            if( colScaleLoc(jLoc) == Real(0) )
                colScaleLoc(jLoc) = Real(1);
        DiagonalScale( LEFT, NORMAL, colScale, dCol );
        DiagonalSolve( RIGHT, NORMAL, colScale, A );
        DiagonalSolve( RIGHT, NORMAL, colScale, B );

        // Geometrically equilibrate the rows
        // ----------------------------------
        // TODO(poulson): Remove GeometricRowScaling
        GeometricRowScaling( A, rowScaleA );
        for( Int iLoc=0; iLoc<mLocalA; ++iLoc )
            if( rowScaleALoc(iLoc) == Real(0) )
                rowScaleALoc(iLoc) = Real(1);
        DiagonalScale( LEFT, NORMAL, rowScaleA, dRowA );
        DiagonalSolve( LEFT, NORMAL, rowScaleA, A );

        // TODO(poulson): Remove GeometricRowScaling
        GeometricRowScaling( B, rowScaleB );
        for( Int iLoc=0; iLoc<mLocalB; ++iLoc )
            if( rowScaleBLoc(iLoc) == Real(0) )
                rowScaleBLoc(iLoc) = Real(1);
        DiagonalScale( LEFT, NORMAL, rowScaleB, dRowB );
        DiagonalSolve( LEFT, NORMAL, rowScaleB, B );

        auto newMaxAbsA = MaxAbsLoc( A );
        auto newMaxAbsB = MaxAbsLoc( B );
        const Real newMaxAbsVal = Max(newMaxAbsA.value,newMaxAbsB.value);
        const Real newMinAbsValA = MinAbsNonzero( A, newMaxAbsVal );
        const Real newMinAbsValB = MinAbsNonzero( B, newMaxAbsVal );
        const Real newMinAbsVal = Min(newMinAbsValA,newMinAbsValB);
        const Real newRatio = newMaxAbsVal / newMinAbsVal;
        if( progress && A.Grid().Rank() == 0 )
            Output("New ratio is ",newMaxAbsVal,"/",newMinAbsVal,"=",newRatio);
        if( iter >= minIter && newRatio >= ratio*relTol )
            break;
        ratio = newRatio;
    }
    SetIndent( indent );

    // Scale each column so that its maximum entry is 1 or 0
    // =====================================================
    colScaleB.AlignWith( colScale );
    ColumnMaxNorms( A, colScale );
    ColumnMaxNorms( B, colScaleB );
    for( Int jLoc=0; jLoc<nLocal; ++jLoc )
    {
        Real maxScale = Max(colScaleLoc(jLoc),colScaleBLoc(jLoc));
        if( maxScale == Real(0) )
            maxScale = 1;
        colScaleLoc(jLoc) = maxScale;
    }
    DiagonalScale( LEFT, NORMAL, colScale, dCol );
    DiagonalSolve( RIGHT, NORMAL, colScale, A );
    DiagonalSolve( RIGHT, NORMAL, colScale, B );
}


#define PROTO(Field) \
  template void GeomEquil \
  ( Matrix<Field>& A, \
    Matrix<Base<Field>>& dRow, \
    Matrix<Base<Field>>& dCol, \
    bool progress ); \
  template void GeomEquil \
  ( AbstractDistMatrix<Field>& A, \
    AbstractDistMatrix<Base<Field>>& dRow, \
    AbstractDistMatrix<Base<Field>>& dCol, \
    bool progress ); \
  template void StackedGeomEquil \
  ( Matrix<Field>& A, \
    Matrix<Field>& B, \
    Matrix<Base<Field>>& dRowA, \
    Matrix<Base<Field>>& dRowB, \
    Matrix<Base<Field>>& dCol, \
    bool progress ); \
  template void StackedGeomEquil \
  ( AbstractDistMatrix<Field>& A, \
    AbstractDistMatrix<Field>& B, \
    AbstractDistMatrix<Base<Field>>& dRowA, \
    AbstractDistMatrix<Base<Field>>& dRowB, \
    AbstractDistMatrix<Base<Field>>& dCol, \
    bool progress );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

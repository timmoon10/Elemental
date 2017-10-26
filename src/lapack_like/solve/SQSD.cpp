/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

// Solvers for Symmetric Quasi Semi-Definite matrices,
//
//   J = | F,    A |,
//       | A^T, -G |
//
// where F and G are Symmetric Positive Semi-Definite (and F is n0 x n0).

template<typename Field>
void SQSDSolve
( Int n0, UpperOrLower uplo, const Matrix<Field>& A, Matrix<Field>& B )
{
    EL_DEBUG_CSE
    const Orientation orient = NORMAL;
    const bool conjugate = true;
    // TODO(poulson): LDLPivotCtrl control structure
    return SymmetricSolve( uplo, orient, A, B, conjugate );
}

template<typename Field>
void SQSDSolve
( Int n0,
  UpperOrLower uplo,
  const AbstractDistMatrix<Field>& A,
        AbstractDistMatrix<Field>& B )
{
    EL_DEBUG_CSE
    const Orientation orient = NORMAL;
    const bool conjugate = true;
    // TODO(poulson): LDLPivotCtrl control structure
    return SymmetricSolve( uplo, orient, A, B, conjugate );
}


#define PROTO(Field) \
  template void SQSDSolve \
  ( Int n0, \
    UpperOrLower uplo, \
    const Matrix<Field>& A, \
          Matrix<Field>& B ); \
  template void SQSDSolve \
  ( Int n0, \
    UpperOrLower uplo, \
    const AbstractDistMatrix<Field>& A, \
          AbstractDistMatrix<Field>& B );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

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
void SymmetricDiagonalEquil
( Matrix<Field>& A,
  Matrix<Base<Field>>& d,
  bool progress )
{
    EL_DEBUG_CSE
    // TODO(poulson): Ensure A is square
    const Int n = A.Height();
    Ones( d, n, 1 );
    if( progress )
        Output("Diagonal equilibration not yet enabled for dense matrices");
}

template<typename Field>
void SymmetricDiagonalEquil
( AbstractDistMatrix<Field>& A,
  AbstractDistMatrix<Base<Field>>& d,
  bool progress )
{
    EL_DEBUG_CSE
    // TODO(poulson): Ensure A is square
    const Int n = A.Height();
    Ones( d, n, 1 );
    if( progress )
        Output("Diagonal equilibration not yet enabled for dense matrices");
}

#define PROTO(Field) \
  template void SymmetricDiagonalEquil \
  ( Matrix<Field>& A, \
    Matrix<Base<Field>>& d, \
    bool progress ); \
  template void SymmetricDiagonalEquil \
  ( AbstractDistMatrix<Field>& A, \
    AbstractDistMatrix<Base<Field>>& d, \
    bool progress );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

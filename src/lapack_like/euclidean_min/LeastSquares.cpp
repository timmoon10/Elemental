/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

namespace ls {

template<typename F>
void Overwrite
( Orientation orientation,
        Matrix<F>& A,
  const Matrix<F>& B,
        Matrix<F>& X )
{
    EL_DEBUG_CSE

    Matrix<F> phase;
    Matrix<Base<F>> signature;

    const Int m = A.Height();
    const Int n = A.Width();
    if( m >= n )
    {
        QR( A, phase, signature );
        qr::SolveAfter( orientation, A, phase, signature, B, X );
    }
    else
    {
        LQ( A, phase, signature );
        lq::SolveAfter( orientation, A, phase, signature, B, X );
    }
}

template<typename F>
void Overwrite
( Orientation orientation,
        AbstractDistMatrix<F>& APre,
  const AbstractDistMatrix<F>& B,
        AbstractDistMatrix<F>& X )
{
    EL_DEBUG_CSE

    DistMatrixReadProxy<F,F,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    DistMatrix<F,MD,STAR> phase(A.Grid());
    DistMatrix<Base<F>,MD,STAR> signature(A.Grid());

    const Int m = A.Height();
    const Int n = A.Width();
    if( m >= n )
    {
        QR( A, phase, signature );
        qr::SolveAfter( orientation, A, phase, signature, B, X );
    }
    else
    {
        LQ( A, phase, signature );
        lq::SolveAfter( orientation, A, phase, signature, B, X );
    }
}

} // namespace ls

template<typename F>
void LeastSquares
( Orientation orientation,
  const Matrix<F>& A,
  const Matrix<F>& B,
        Matrix<F>& X )
{
    EL_DEBUG_CSE
    Matrix<F> ACopy( A );
    ls::Overwrite( orientation, ACopy, B, X );
}

template<typename F>
void LeastSquares
( Orientation orientation,
  const AbstractDistMatrix<F>& A,
  const AbstractDistMatrix<F>& B,
        AbstractDistMatrix<F>& X )
{
    EL_DEBUG_CSE
    DistMatrix<F> ACopy( A );
    ls::Overwrite( orientation, ACopy, B, X );
}


#define PROTO(F) \
  template void ls::Overwrite \
  ( Orientation orientation, \
          Matrix<F>& A, \
    const Matrix<F>& B, \
          Matrix<F>& X ); \
  template void ls::Overwrite \
  ( Orientation orientation, \
          AbstractDistMatrix<F>& A, \
    const AbstractDistMatrix<F>& B, \
          AbstractDistMatrix<F>& X ); \
  template void LeastSquares \
  ( Orientation orientation, \
    const Matrix<F>& A, \
    const Matrix<F>& B, \
          Matrix<F>& X ); \
  template void LeastSquares \
  ( Orientation orientation, \
    const AbstractDistMatrix<F>& A, \
    const AbstractDistMatrix<F>& B, \
          AbstractDistMatrix<F>& X );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

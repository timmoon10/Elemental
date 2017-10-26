/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename F>
void Tikhonov
( Orientation orientation,
  const Matrix<F>& A,
  const Matrix<F>& B,
  const Matrix<F>& G,
        Matrix<F>& X,
  TikhonovAlg alg )
{
    EL_DEBUG_CSE
    const bool normal = ( orientation==NORMAL );
    const Int m = ( normal ? A.Height() : A.Width()  );
    const Int n = ( normal ? A.Width()  : A.Height() );
    if( G.Width() != n )
        LogicError("Tikhonov matrix was the wrong width");
    if( orientation == TRANSPOSE && IsComplex<F>::value )
        LogicError("Transpose version of complex Tikhonov not yet supported");

    if( m >= n )
    {
        Matrix<F> Z;
        if( alg == TIKHONOV_CHOLESKY )
        {
            if( orientation == NORMAL )
                Herk( LOWER, ADJOINT, Base<F>(1), A, Z );
            else
                Herk( LOWER, NORMAL, Base<F>(1), A, Z );
            Herk( LOWER, ADJOINT, Base<F>(1), G, Base<F>(1), Z );
            Cholesky( LOWER, Z );
        }
        else
        {
            const Int mG = G.Height();
            Zeros( Z, m+mG, n );
            auto ZT = Z( IR(0,m),    IR(0,n) );
            auto ZB = Z( IR(m,m+mG), IR(0,n) );
            if( orientation == NORMAL )
                ZT = A;
            else
                Adjoint( A, ZT );
            ZB = G;
            qr::ExplicitTriang( Z );
        }
        if( orientation == NORMAL )
            Gemm( ADJOINT, NORMAL, F(1), A, B, X );
        else
            Gemm( NORMAL, NORMAL, F(1), A, B, X );
        cholesky::SolveAfter( LOWER, NORMAL, Z, X );
    }
    else
    {
        LogicError("This case not yet supported");
    }
}

template<typename F>
void Tikhonov
( Orientation orientation,
  const AbstractDistMatrix<F>& APre,
  const AbstractDistMatrix<F>& BPre,
  const AbstractDistMatrix<F>& G,
        AbstractDistMatrix<F>& XPre,
  TikhonovAlg alg )
{
    EL_DEBUG_CSE

    DistMatrixReadProxy<F,F,MC,MR>
      AProx( APre ),
      BProx( BPre );
    DistMatrixWriteProxy<F,F,MC,MR>
      XProx( XPre );
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& X = XProx.Get();

    const bool normal = ( orientation==NORMAL );
    const Int m = ( normal ? A.Height() : A.Width()  );
    const Int n = ( normal ? A.Width()  : A.Height() );
    if( G.Width() != n )
        LogicError("Tikhonov matrix was the wrong width");
    if( orientation == TRANSPOSE && IsComplex<F>::value )
        LogicError("Transpose version of complex Tikhonov not yet supported");

    if( m >= n )
    {
        DistMatrix<F> Z(A.Grid());
        if( alg == TIKHONOV_CHOLESKY )
        {
            if( orientation == NORMAL )
                Herk( LOWER, ADJOINT, Base<F>(1), A, Z );
            else
                Herk( LOWER, NORMAL, Base<F>(1), A, Z );
            Herk( LOWER, ADJOINT, Base<F>(1), G, Base<F>(1), Z );
            Cholesky( LOWER, Z );
        }
        else
        {
            const Int mG = G.Height();
            Zeros( Z, m+mG, n );
            auto ZT = Z( IR(0,m),    IR(0,n) );
            auto ZB = Z( IR(m,m+mG), IR(0,n) );
            if( orientation == NORMAL )
                ZT = A;
            else
                Adjoint( A, ZT );
            ZB = G;
            qr::ExplicitTriang( Z );
        }
        if( orientation == NORMAL )
            Gemm( ADJOINT, NORMAL, F(1), A, B, X );
        else
            Gemm( NORMAL, NORMAL, F(1), A, B, X );
        cholesky::SolveAfter( LOWER, NORMAL, Z, X );
    }
    else
    {
        LogicError("This case not yet supported");
    }
}


#define PROTO(F) \
  template void Tikhonov \
  ( Orientation orientation, \
    const Matrix<F>& A, \
    const Matrix<F>& B, \
    const Matrix<F>& G, \
          Matrix<F>& X, \
          TikhonovAlg alg ); \
  template void Tikhonov \
  ( Orientation orientation, \
    const AbstractDistMatrix<F>& A, \
    const AbstractDistMatrix<F>& B, \
    const AbstractDistMatrix<F>& G, \
          AbstractDistMatrix<F>& X, \
          TikhonovAlg alg );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/matrices.hpp>

namespace El {

template<typename T>
void DynamicRegCounter( Matrix<T>& A, Int n )
{
    EL_DEBUG_CSE
    Zeros( A, 2*n, 2*n );
    auto ATL = A( IR(0,n),   IR(0,n)   );
    auto ATR = A( IR(0,n),   IR(n,2*n) );
    auto ABL = A( IR(n,2*n), IR(0,n)   );
    auto ABR = A( IR(n,2*n), IR(n,2*n) );

    JordanCholesky( ATL, n );
    Identity( ATR, n, n );
    Identity( ABL, n, n );
    Identity( ABR, n, n ); ABR *= -1;
}

template<typename T>
void DynamicRegCounter( ElementalMatrix<T>& APre, Int n )
{
    EL_DEBUG_CSE
    DistMatrixWriteProxy<T,T,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    Zeros( A, 2*n, 2*n );
    auto ATL = A( IR(0,n),   IR(0,n)   );
    auto ATR = A( IR(0,n),   IR(n,2*n) );
    auto ABL = A( IR(n,2*n), IR(0,n)   );
    auto ABR = A( IR(n,2*n), IR(n,2*n) );

    JordanCholesky( ATL, n );
    Identity( ATR, n, n );
    Identity( ABL, n, n );
    Identity( ABR, n, n ); ABR *= -1;
}


#define PROTO(T) \
  template void DynamicRegCounter( Matrix<T>& A, Int n ); \
  template void DynamicRegCounter( ElementalMatrix<T>& A, Int n );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

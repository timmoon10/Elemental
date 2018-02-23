/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename T>
void ConfigurePrecision( ostream& os )
{
    // Force the full precision to be reported
    const Int numDecimals =
      BinaryToDecimalPrecision(NumMantissaBits(Base<T>()))+1;
    os.precision( numDecimals );
}

// Dense
// =====
template <typename T>
void Print(AbstractMatrix<T> const& A, string title, ostream& os)
{
    switch (A.GetDevice())
    {
    case Device::CPU:
        Print(static_cast<Matrix<T,Device::CPU> const&>(A), title, os);
        break;
    case Device::GPU:
    {
        // Copy to host
        Matrix<T,Device::CPU> A_CPU =
            static_cast<Matrix<T,Device::CPU> const&>(A);
        Print(A_CPU, title, os);
    }
    break;
    default:
        LogicError("Print: Bad device.");
    }
}

template<typename T>
void Print( const Matrix<T>& A, string title, ostream& os )
{
    EL_DEBUG_CSE
    if( title != "" )
        os << title << endl;

    ConfigurePrecision<T>( os );

    const Int height = A.Height();
    const Int width = A.Width();
    for( Int i=0; i<height; ++i )
    {
        for( Int j=0; j<width; ++j )
            os << A.Get(i,j) << " ";
        os << endl;
    }
    os << endl;
}

template<typename T>
void Print
( const AbstractDistMatrix<T>& A, string title, ostream& os )
{
    EL_DEBUG_CSE
    if( A.ColStride() == 1 && A.RowStride() == 1 )
    {
        if( A.CrossRank() == A.Root() && A.RedundantRank() == 0 )
            Print( A.LockedMatrix(), title, os );
    }
    else
    {
        DistMatrix<T,CIRC,CIRC> A_CIRC_CIRC( A );
        if( A_CIRC_CIRC.CrossRank() == A_CIRC_CIRC.Root() )
            Print( A_CIRC_CIRC.LockedMatrix(), title, os );
    }
}

// Utilities
// =========

template<typename T>
void Print( const vector<T>& x, string title, ostream& os )
{
    EL_DEBUG_CSE
    if( title != "" )
        os << title << endl;

    ConfigurePrecision<T>( os );

    const Int length = x.size();
    for( Int i=0; i<length; ++i )
        os << x[i] << " ";
    os << endl;
}

#define PROTO(T) \
  template void Print \
  ( const vector<T>& x, string title, ostream& os ); \
  template void Print \
  ( const Matrix<T>& A, string title, ostream& os ); \
  template void Print \
  ( const AbstractDistMatrix<T>& A, string title, ostream& os );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

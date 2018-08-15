/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_FILTER_HPP
#define EL_BLAS_COPY_FILTER_HPP

namespace El {
namespace copy {

template<typename T,Dist U,Dist V,Device D,typename>
void Filter
( DistMatrix<T,Collect<U>(),Collect<V>(),ELEMENT,D> const& A,
  DistMatrix<T,U,V,ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    B.Resize( A.Height(), A.Width() );
    if( !B.Participating() )
        return;

    SyncInfo<D> syncInfoA(A.LockedMatrix());

    const Int colShift = B.ColShift();
    const Int rowShift = B.RowShift();
    util::InterleaveMatrix(
        B.LocalHeight(), B.LocalWidth(),
        A.LockedBuffer(colShift,rowShift), B.ColStride(), B.RowStride()*A.LDim(),
        B.Buffer(),                        1,             B.LDim(),
        syncInfoA);
    // FIXME: Need to sync A and B here
}

template<typename T,Dist U,Dist V,Device D,typename,typename>
void Filter
( DistMatrix<T,Collect<U>(),Collect<V>(),ELEMENT,D> const& A,
  DistMatrix<T,U,V,ELEMENT,D>& B )
{
    LogicError("Filter: Bad device/type combination.");
}

template<typename T,Dist U,Dist V>
void Filter
( const DistMatrix<T,Collect<U>(),Collect<V>(),BLOCK>& A,
        DistMatrix<T,        U,           V   ,BLOCK>& B )
{
    EL_DEBUG_CSE
    // TODO(poulson): More efficient implementation
    GeneralPurpose( A, B );
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_FILTER_HPP

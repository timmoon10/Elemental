/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_COLALLTOALLPROMOTE_HPP
#define EL_BLAS_COPY_COLALLTOALLPROMOTE_HPP

namespace El {
namespace copy {

// FIXME (trb 03/06/18) -- Need to do the GPU impl
template<typename T,Dist U,Dist V,Device D>
void ColAllToAllPromote
( const DistMatrix<T,U,V,ELEMENT,D>& A,
  DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),ELEMENT,D>& B)
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    const Int height = A.Height();
    const Int width = A.Width();
    B.AlignColsAndResize
    ( Mod(A.ColAlign(),B.ColStride()), height, width, false, false );
    if( !B.Participating() )
        return;

    const Int colStride = A.ColStride();
    const Int colStridePart = A.PartialColStride();
    const Int colStrideUnion = A.PartialUnionColStride();
    const Int colRankPart = A.PartialColRank();
    const Int colDiff = B.ColAlign() - Mod(A.ColAlign(),colStridePart);

    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int maxLocalWidth = MaxLength(width,colStrideUnion);
    const Int portionSize = mpi::Pad( maxLocalHeight*maxLocalWidth );

    SyncInfo<D> syncInfoA(A.LockedMatrix()), syncInfoB(A.LockedMatrix());
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    if( colDiff == 0 )
    {
        if( A.PartialUnionColStride() == 1 )
        {
            Copy( A.LockedMatrix(), B.Matrix() );
        }
        else
        {
            simple_buffer<T,D> buffer(2*colStrideUnion*portionSize, syncInfoB);
            T* firstBuf  = buffer.data();
            T* secondBuf = buffer.data() + colStrideUnion*portionSize;

            // Pack
            util::RowStridedPack(
                A.LocalHeight(), width,
                B.RowAlign(), colStrideUnion,
                A.LockedBuffer(), A.LDim(),
                firstBuf,         portionSize, syncInfoB);

            // Simultaneously Gather in columns and Scatter in rows
            mpi::AllToAll(
                firstBuf,  portionSize,
                secondBuf, portionSize, A.PartialUnionColComm(),
                syncInfoB);

            // Unpack
            util::PartialColStridedUnpack(
                height, B.LocalWidth(),
                A.ColAlign(), colStride,
                colStrideUnion, colStridePart, colRankPart,
                B.ColShift(),
                secondBuf,  portionSize,
                B.Buffer(), B.LDim(), syncInfoB);
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( A.Grid().Rank() == 0 )
            cerr << "Unaligned PartialColAllToAllPromote" << endl;
#endif
        const Int sendColRankPart = Mod( colRankPart+colDiff, colStridePart );
        const Int recvColRankPart = Mod( colRankPart-colDiff, colStridePart );

        simple_buffer<T,D> buffer(2*colStrideUnion*portionSize, syncInfoB);
        T* firstBuf  = buffer.data();
        T* secondBuf = buffer.data() + colStrideUnion*portionSize;

        // Pack
        util::RowStridedPack(
            A.LocalHeight(), width,
            B.RowAlign(), colStrideUnion,
            A.LockedBuffer(), A.LDim(),
            secondBuf,        portionSize, syncInfoB);

        Synchronize(syncInfoB);

        // Realign the input
        mpi::SendRecv(
            secondBuf, colStrideUnion*portionSize, sendColRankPart,
            firstBuf,  colStrideUnion*portionSize, recvColRankPart,
            A.PartialColComm());

        // Simultaneously Scatter in columns and Gather in rows
        mpi::AllToAll(
            firstBuf,  portionSize,
            secondBuf, portionSize, A.PartialUnionColComm(),
            syncInfoB);

        // Unpack
        util::PartialColStridedUnpack(
            height, B.LocalWidth(),
            A.ColAlign(), colStride,
            colStrideUnion, colStridePart, recvColRankPart,
            B.ColShift(),
            secondBuf,  portionSize,
            B.Buffer(), B.LDim(), syncInfoB);
    }
}

template<typename T,Dist U,Dist V>
void ColAllToAllPromote
( const DistMatrix<T,        U,                     V   ,BLOCK>& A,
        DistMatrix<T,Partial<U>(),PartialUnionRow<U,V>(),BLOCK>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    // TODO: More efficient implementation
    GeneralPurpose( A, B );
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_COLALLTOALLPROMOTE_HPP

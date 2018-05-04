/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_UTIL_HPP
#define EL_BLAS_COPY_UTIL_HPP

#ifdef HYDROGEN_HAVE_CUDA
#include "../GPU/Copy.hpp"
#endif

namespace El
{
namespace copy
{
namespace util
{
namespace details
{

template <typename T, Device D, bool=IsDeviceValidType_v<T,D>()>
struct Impl
{
    template <typename... Ts>
    static void InterleaveMatrix(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }

    template <typename... Ts>
    static void RowStridedPack(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }

    template <typename... Ts>
    static void RowStridedUnpack(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }

    template <typename... Ts>
    static void PartialRowStridedPack(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }

    template <typename... Ts>
    static void PartialRowStridedUnpack(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }

    template <typename... Ts>
    static void PartialColStridedColumnPack(Ts&&...)
    {
        LogicError("copy::util::Bad device/type combination.");
    }
};

template <typename T>
struct Impl<T, Device::CPU, true>
{
    static void InterleaveMatrix(
        Int height, Int width,
        T const* A, Int colStrideA, Int rowStrideA,
        T* B, Int colStrideB, Int rowStrideB)
    {
        if (colStrideA == 1 && colStrideB == 1)
        {
            lapack::Copy('F', height, width, A, rowStrideA, B, rowStrideB);
        }
        else
        {
#ifdef HYDROGEN_HAVE_MKL
            mkl::omatcopy
                (NORMAL, height, width, T(1),
                  A, rowStrideA, colStrideA,
                  B, rowStrideB, colStrideB);
#else
            for(Int j=0; j<width; ++j)
                StridedMemCopy
                    (&B[j*rowStrideB], colStrideB,
                      &A[j*rowStrideA], colStrideA, height);
#endif
        }
    }

    static void RowStridedPack(Int height, Int width,
                               Int rowAlign, Int rowStride,
                               T const* A,Int ALDim,
                               T* BPortions, Int portionSize)
    {
        for (Int k=0; k<rowStride; ++k)
        {
            const Int rowShift = Shift_(k, rowAlign, rowStride);
            const Int localWidth = Length_(width, rowShift, rowStride);
            lapack::Copy
                ('F', height, localWidth,
                 &A[rowShift*ALDim],        rowStride*ALDim,
                 &BPortions[k*portionSize], height);
        }
    }

    static void RowStridedUnpack
    (Int height, Int width,
      Int rowAlign, Int rowStride,
      const T* APortions, Int portionSize,
      T* B,         Int BLDim)
    {
        for (Int k=0; k<rowStride; ++k)
        {
            const Int rowShift = Shift_(k, rowAlign, rowStride);
            const Int localWidth = Length_(width, rowShift, rowStride);
            lapack::Copy
                ('F', height, localWidth,
                 &APortions[k*portionSize], height,
                 &B[rowShift*BLDim],        rowStride*BLDim);
        }
    }

    static void PartialRowStridedPack
    (Int height, Int width,
     Int rowAlign, Int rowStride,
     Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
     Int rowShiftA,
     const T* A,         Int ALDim,
     T* BPortions, Int portionSize)
    {
        for (Int k=0; k<rowStrideUnion; ++k)
        {
            const Int rowShift =
                Shift_(rowRankPart+k*rowStridePart, rowAlign, rowStride);
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_(width, rowShift, rowStride);
            lapack::Copy
                ('F', height, localWidth,
                 &A[rowOffset*ALDim],       rowStrideUnion*ALDim,
                 &BPortions[k*portionSize], height);
        }
    }

    static void PartialRowStridedUnpack
    (Int height, Int width,
     Int rowAlign, Int rowStride,
     Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
     Int rowShiftB,
     const T* APortions, Int portionSize,
     T* B, Int BLDim)
    {
        for (Int k=0; k<rowStrideUnion; ++k)
        {
            const Int rowShift =
                Shift_(rowRankPart+k*rowStridePart, rowAlign, rowStride);
            const Int rowOffset = (rowShift-rowShiftB) / rowStridePart;
            const Int localWidth = Length_(width, rowShift, rowStride);
            lapack::Copy
                ('F', height, localWidth,
                 &APortions[k*portionSize], height,
                 &B[rowOffset*BLDim],       rowStrideUnion*BLDim);
        }
    }

    static void PartialColStridedColumnPack
    (Int height,
     Int colAlign, Int colStride,
     Int colStrideUnion, Int colStridePart, Int colRankPart,
     Int colShiftA,
     const T* A,
     T* BPortions, Int portionSize)
    {
        for (Int k=0; k<colStrideUnion; ++k)
        {
            const Int colShift =
                Shift_(colRankPart+k*colStridePart, colAlign, colStride);
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_(height, colShift, colStride);
            StridedMemCopy
                (&BPortions[k*portionSize], 1,
                 &A[colOffset],             colStrideUnion, localHeight);
        }
    }

};

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
struct Impl<T, Device::GPU, true>
{
    static void InterleaveMatrix(Int height, Int width,
                     T const* A, Int colStrideA, Int rowStrideA,
                     T* B, Int colStrideB, Int rowStrideB)
    {
        if (colStrideA == 1 && colStrideB == 1)
        {
            EL_CHECK_CUDA(cudaMemcpy2DAsync(B, rowStrideB*sizeof(T),
                                            A, rowStrideA*sizeof(T),
                                            height*sizeof(T), width,
                                            cudaMemcpyDeviceToDevice,
                                            GPUManager::Stream()));
        }
        else
        {
            Copy_GPU_impl(height, width,
                          A, colStrideA, rowStrideA,
                          B, colStrideB, rowStrideB);
        }
    }

    static void RowStridedPack(Int height, Int width,
                               Int rowAlign, Int rowStride,
                               T const* A,Int ALDim,
                               T* BPortions, Int portionSize)
    {
        for (Int k=0; k<rowStride; ++k)
        {
            const Int rowShift = Shift_(k, rowAlign, rowStride);
            const Int localWidth = Length_(width, rowShift, rowStride);
            EL_CHECK_CUDA(cudaMemcpy2DAsync(BPortions + k*portionSize, height*sizeof(T),
                                            A+rowShift*ALDim, rowStride*ALDim*sizeof(T),
                                            height*sizeof(T), localWidth,
                                            cudaMemcpyDeviceToDevice,
                                            GPUManager::Stream()));
        }
    }

    static void RowStridedUnpack(Int height, Int width,
                                 Int rowAlign, Int rowStride,
                                 T const* APortions, Int portionSize,
                                 T* B, Int BLDim)
    {
        for (Int k=0; k<rowStride; ++k)
        {
            const Int rowShift = Shift_(k, rowAlign, rowStride);
            const Int localWidth = Length_(width, rowShift, rowStride);
            EL_CHECK_CUDA(cudaMemcpy2DAsync(B+rowShift*BLDim, rowStride*BLDim*sizeof(T),
                                            APortions+k*portionSize, height*sizeof(T),
                                            height*sizeof(T), localWidth,
                                            cudaMemcpyDeviceToDevice,
                                            GPUManager::Stream()));
        }
    }

    static void PartialRowStridedPack
    (Int height, Int width,
     Int rowAlign, Int rowStride,
     Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
     Int rowShiftA,
     const T* A,         Int ALDim,
     T* BPortions, Int portionSize)
    {
        for (Int k=0; k<rowStrideUnion; ++k)
        {
            const Int rowShift = Shift_(rowRankPart+k*rowStridePart,
                                        rowAlign, rowStride);
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_(width, rowShift, rowStride);
            EL_CHECK_CUDA(cudaMemcpy2DAsync(
                BPortions + k*portionSize, height*sizeof(T),
                A + rowOffset*ALDim, rowStrideUnion*ALDim*sizeof(T),
                height*sizeof(T), localWidth,
                cudaMemcpyDeviceToDevice,
                GPUManager::Stream()));
        }
    }

    static void PartialRowStridedUnpack
    (Int height, Int width,
     Int rowAlign, Int rowStride,
     Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
     Int rowShiftB,
     const T* APortions, Int portionSize,
     T* B, Int BLDim)
    {
        for (Int k=0; k<rowStrideUnion; ++k)
        {
            const Int rowShift = Shift_(rowRankPart+k*rowStridePart,
                                        rowAlign, rowStride);
            const Int rowOffset = (rowShift-rowShiftB) / rowStridePart;
            const Int localWidth = Length_(width, rowShift, rowStride);
            EL_CHECK_CUDA(cudaMemcpy2DAsync(
                B + rowOffset*BLDim, rowStrideUnion*BLDim*sizeof(T),
                APortions + k*portionSize, height*sizeof(T),
                height*sizeof(T), localWidth,
                cudaMemcpyDeviceToDevice,
                GPUManager::Stream()));
        }
    }

    static void PartialColStridedColumnPack
    (Int height,
     Int colAlign, Int colStride,
     Int colStrideUnion, Int colStridePart, Int colRankPart,
     Int colShiftA,
     const T* A,
     T* BPortions, Int portionSize)
    {
        for (Int k=0; k<colStrideUnion; ++k)
        {
            const Int colShift =
                Shift_(colRankPart+k*colStridePart, colAlign, colStride);
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_(height, colShift, colStride);

            (void)colOffset;
            LogicError("PartialColStridedColumnPack<T,GPU>: Not implemented.");
#if 0
            StridedMemCopy
                (&BPortions[k*portionSize], 1,
                 &A[colOffset],             colStrideUnion, localHeight);
#endif
        }
        cudaDeviceSynchronize();
    }

};
#endif // HYDROGEN_HAVE_CUDA

template <Device SrcD, Device DestD> struct InterDevice;

#ifdef HYDROGEN_HAVE_CUDA
template <>
struct InterDevice<Device::CPU,Device::GPU>
{
    template <typename T>
    static void MemCopy2D(T * EL_RESTRICT const dest, Int const dest_ldim,
                   T const* EL_RESTRICT const src, Int const src_ldim,
                   Int const height, Int const width)
    {
        auto stream = GPUManager::Stream();
        EL_CHECK_CUDA(cudaMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            cudaMemcpyHostToDevice,
            stream));
        EL_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
};// InterDevice<CPU,GPU>

template <>
struct InterDevice<Device::GPU,Device::CPU>
{
    template <typename T>
    static void MemCopy2D(T * EL_RESTRICT const dest, Int const dest_ldim,
                   T const* EL_RESTRICT const src, Int const src_ldim,
                   Int const height, Int const width)
    {
        auto stream = GPUManager::Stream();
        EL_CHECK_CUDA(cudaMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            cudaMemcpyDeviceToHost,
            stream));
        EL_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
};// InterDevice<CPU,GPU>
#endif // HYDROGEN_HAVE_CUDA

}// namespace details


template <Device SrcD, Device DestD, typename T>
void InterDeviceMemCopy2D(
    T * EL_RESTRICT const dest, Int const dest_ldim,
    T const* EL_RESTRICT const src, Int const src_ldim,
    Int const height, Int const width)
{
#ifndef EL_RELEASE
    if ((dest_ldim < height) || (src_ldim < height))
        LogicError("InterDeviceMemCopy2D: Bad ldim/height.");
#endif // !EL_RELEASE
    details::InterDevice<SrcD,DestD>::MemCopy2D(
        dest, dest_ldim, src, src_ldim, height, width);
}

template<typename T, Device D>
void InterleaveMatrix
(Int height, Int width,
  const T* A, Int colStrideA, Int rowStrideA,
        T* B, Int colStrideB, Int rowStrideB)
{
    details::Impl<T,D>::InterleaveMatrix(height, width,
                                         A, colStrideA, rowStrideA,
                                         B, colStrideB, rowStrideB);
}

template<typename T,Device D>
void ColStridedPack
(Int height, Int width,
  Int colAlign, Int colStride,
  const T* A,         Int ALDim,
        T* BPortions, Int portionSize)
{
    for (Int k=0; k<colStride; ++k)
    {
        const Int colShift = Shift_(k, colAlign, colStride);
        const Int localHeight = Length_(height, colShift, colStride);
        InterleaveMatrix<T,D>
        (localHeight, width,
          &A[colShift],              colStride, ALDim,
          &BPortions[k*portionSize], 1,         localHeight);
    }
}

// FIXME: GPU IMPL
// TODO(poulson): Use this routine
template<typename T>
void ColStridedColumnPack
(Int height,
  Int colAlign, Int colStride,
  const T* A,
        T* BPortions, Int portionSize)
{
    for (Int k=0; k<colStride; ++k)
    {
        const Int colShift = Shift_(k, colAlign, colStride);
        const Int localHeight = Length_(height, colShift, colStride);
        StridedMemCopy
        (&BPortions[k*portionSize], 1,
          &A[colShift],              colStride, localHeight);
    }
}

template<typename T,Device D>
void ColStridedUnpack
(Int height, Int width,
  Int colAlign, Int colStride,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    for (Int k=0; k<colStride; ++k)
    {
        const Int colShift = Shift_(k, colAlign, colStride);
        const Int localHeight = Length_(height, colShift, colStride);
        InterleaveMatrix<T,D>
        (localHeight, width,
          &APortions[k*portionSize], 1,         localHeight,
          &B[colShift],              colStride, BLDim);
    }
}

template<typename T>
void BlockedColStridedUnpack
(Int height, Int width,
  Int colAlign, Int colStride,
  Int blockHeight, Int colCut,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    const Int firstBlockHeight = blockHeight - colCut;
    for (Int portion=0; portion<colStride; ++portion)
    {
        const T* APortion = &APortions[portion*portionSize];
        const Int colShift = Shift_(portion, colAlign, colStride);
        const Int localHeight =
          BlockedLength_(height, colShift, blockHeight, colCut, colStride);

        // Loop over the block rows from this portion
        Int blockRow = colShift;
        Int rowIndex =
          (colShift==0 ? 0 : firstBlockHeight + (colShift-1)*blockHeight);
        Int packedRowIndex = 0;
        while(rowIndex < height)
        {
            const Int thisBlockHeight =
              (blockRow == 0 ?
                firstBlockHeight :
                Min(blockHeight,height-rowIndex));

            lapack::Copy
            ('F', thisBlockHeight, width,
              &APortion[packedRowIndex], localHeight,
              &B[rowIndex],              BLDim);

            blockRow += colStride;
            rowIndex += thisBlockHeight + (colStride-1)*blockHeight;
            packedRowIndex += thisBlockHeight;
        }
    }
}

template <typename T,Device D>
void PartialColStridedPack
(Int height, Int width,
  Int colAlign, Int colStride,
  Int colStrideUnion, Int colStridePart, Int colRankPart,
  Int colShiftA,
  const T* A,         Int ALDim,
        T* BPortions, Int portionSize)
{
    for (Int k=0; k<colStrideUnion; ++k)
    {
        const Int colShift =
            Shift_(colRankPart+k*colStridePart, colAlign, colStride);
        const Int colOffset = (colShift-colShiftA) / colStridePart;
        const Int localHeight = Length_(height, colShift, colStride);
        InterleaveMatrix<T,D>
        (localHeight, width,
          A+colOffset,             colStrideUnion, ALDim,
          BPortions+k*portionSize, 1,              localHeight);
    }
}

template<typename T>
void PartialColStridedColumnPack
(Int height,
  Int colAlign, Int colStride,
  Int colStrideUnion, Int colStridePart, Int colRankPart,
  Int colShiftA,
  const T* A,
        T* BPortions, Int portionSize)
{
    for (Int k=0; k<colStrideUnion; ++k)
    {
        const Int colShift =
            Shift_(colRankPart+k*colStridePart, colAlign, colStride);
        const Int colOffset = (colShift-colShiftA) / colStridePart;
        const Int localHeight = Length_(height, colShift, colStride);
        StridedMemCopy
        (&BPortions[k*portionSize], 1,
          &A[colOffset],             colStrideUnion, localHeight);
    }
}

template<typename T,Device D>
void PartialColStridedUnpack
(Int height, Int width,
  Int colAlign, Int colStride,
  Int colStrideUnion, Int colStridePart, Int colRankPart,
  Int colShiftB,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    for (Int k=0; k<colStrideUnion; ++k)
    {
        const Int colShift =
            Shift_(colRankPart+k*colStridePart, colAlign, colStride);
        const Int colOffset = (colShift-colShiftB) / colStridePart;
        const Int localHeight = Length_(height, colShift, colStride);
        InterleaveMatrix<T,D>
        (localHeight, width,
          &APortions[k*portionSize], 1,              localHeight,
          &B[colOffset],             colStrideUnion, BLDim);
    }
}

template<typename T>
void PartialColStridedColumnUnpack
(Int height,
  Int colAlign, Int colStride,
  Int colStrideUnion, Int colStridePart, Int colRankPart,
  Int colShiftB,
  const T* APortions, Int portionSize,
        T* B)
{
    for (Int k=0; k<colStrideUnion; ++k)
    {
        const Int colShift =
            Shift_(colRankPart+k*colStridePart, colAlign, colStride);
        const Int colOffset = (colShift-colShiftB) / colStridePart;
        const Int localHeight = Length_(height, colShift, colStride);
        StridedMemCopy
        (&B[colOffset],             colStrideUnion,
          &APortions[k*portionSize], 1,              localHeight);
    }
}

template<typename T,Device D>
void RowStridedPack
(Int height, Int width,
  Int rowAlign, Int rowStride,
  const T* A,         Int ALDim,
        T* BPortions, Int portionSize)
{
    details::Impl<T,D>::RowStridedPack(height, width, rowAlign, rowStride,
                                       A, ALDim, BPortions, portionSize);
}

template<typename T,Device D>
void RowStridedUnpack
(Int height, Int width,
  Int rowAlign, Int rowStride,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    details::Impl<T,D>::RowStridedUnpack(height, width, rowAlign, rowStride,
                                         APortions, portionSize, B, BLDim);
}



template<typename T>
void BlockedRowStridedUnpack
(Int height, Int width,
  Int rowAlign, Int rowStride,
  Int blockWidth, Int rowCut,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    const Int firstBlockWidth = blockWidth - rowCut;
    for (Int portion=0; portion<rowStride; ++portion)
    {
        const T* APortion = &APortions[portion*portionSize];
        const Int rowShift = Shift_(portion, rowAlign, rowStride);
        // Loop over the block columns from this portion
        Int blockCol = rowShift;
        Int colIndex =
          (rowShift==0 ? 0 : firstBlockWidth + (rowShift-1)*blockWidth);
        Int packedColIndex = 0;

        while(colIndex < width)
        {
            const Int thisBlockWidth =
              (blockCol == 0 ?
                firstBlockWidth :
                Min(blockWidth,width-colIndex));

            lapack::Copy
            ('F', height, thisBlockWidth,
              &APortion[packedColIndex*height], height,
              &B[colIndex*BLDim],               BLDim);

            blockCol += rowStride;
            colIndex += thisBlockWidth + (rowStride-1)*blockWidth;
            packedColIndex += thisBlockWidth;
        }
    }
}

template<typename T>
void BlockedRowFilter
(Int height, Int width,
  Int rowShift, Int rowStride,
  Int blockWidth, Int rowCut,
  const T* A, Int ALDim,
        T* B, Int BLDim)
{
    EL_DEBUG_CSE
    const Int firstBlockWidth = blockWidth - rowCut;

    // Loop over the block columns from this portion
    Int blockCol = rowShift;
    Int colIndex =
      (rowShift==0 ? 0 : firstBlockWidth + (rowShift-1)*blockWidth);
    Int packedColIndex = 0;

    while(colIndex < width)
    {
        const Int thisBlockWidth =
          (blockCol == 0 ?
            firstBlockWidth :
            Min(blockWidth,width-colIndex));

        lapack::Copy
        ('F', height, thisBlockWidth,
          &A[colIndex      *ALDim], ALDim,
          &B[packedColIndex*BLDim], BLDim);

        blockCol += rowStride;
        colIndex += thisBlockWidth + (rowStride-1)*blockWidth;
        packedColIndex += thisBlockWidth;
    }
}

template<typename T>
void BlockedColFilter
(Int height, Int width,
  Int colShift, Int colStride,
  Int blockHeight, Int colCut,
  const T* A, Int ALDim,
        T* B, Int BLDim)
{
    EL_DEBUG_CSE
    const Int firstBlockHeight = blockHeight - colCut;

    // Loop over the block rows from this portion
    Int blockRow = colShift;
    Int rowIndex =
      (colShift==0 ? 0 : firstBlockHeight + (colShift-1)*blockHeight);
    Int packedRowIndex = 0;

    while(rowIndex < height)
    {
        const Int thisBlockHeight =
          (blockRow == 0 ?
            firstBlockHeight :
            Min(blockHeight,height-rowIndex));

        lapack::Copy
        ('F', thisBlockHeight, width,
          &A[rowIndex],       ALDim,
          &B[packedRowIndex], BLDim);

        blockRow += colStride;
        rowIndex += thisBlockHeight + (colStride-1)*blockHeight;
        packedRowIndex += thisBlockHeight;
    }
}

template<typename T,Device D>
void PartialRowStridedPack
(Int height, Int width,
  Int rowAlign, Int rowStride,
  Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
  Int rowShiftA,
  const T* A,         Int ALDim,
        T* BPortions, Int portionSize)
{
    details::Impl<T,D>::PartialRowStridedPack(
        height, width, rowAlign, rowStride,
        rowStrideUnion, rowStridePart, rowRankPart, rowShiftA,
        A, ALDim, BPortions, portionSize);
}

template<typename T,Device D>
void PartialRowStridedUnpack
(Int height, Int width,
  Int rowAlign, Int rowStride,
  Int rowStrideUnion, Int rowStridePart, Int rowRankPart,
  Int rowShiftB,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    details::Impl<T,D>::PartialRowStridedUnpack(
        height, width, rowAlign, rowStride,
        rowStrideUnion, rowStridePart, rowRankPart, rowShiftB,
        APortions, portionSize, B, BLDim);
}

// NOTE: This is implicitly column-major
template<typename T,Device D>
void StridedPack
(Int height, Int width,
  Int colAlign, Int colStride,
  Int rowAlign, Int rowStride,
  const T* A,         Int ALDim,
        T* BPortions, Int portionSize)
{
    for (Int l=0; l<rowStride; ++l)
    {
        const Int rowShift = Shift_(l, rowAlign, rowStride);
        const Int localWidth = Length_(width, rowShift, rowStride);
        for (Int k=0; k<colStride; ++k)
        {
            const Int colShift = Shift_(k, colAlign, colStride);
            const Int localHeight = Length_(height, colShift, colStride);
            InterleaveMatrix<T,D>
            (localHeight, localWidth,
              &A[colShift+rowShift*ALDim], colStride, rowStride*ALDim,
              &BPortions[(k+l*colStride)*portionSize], 1, localHeight);
        }
    }
}

// NOTE: This is implicitly column-major
template<typename T,Device D>
void StridedUnpack
(Int height, Int width,
  Int colAlign, Int colStride,
  Int rowAlign, Int rowStride,
  const T* APortions, Int portionSize,
        T* B,         Int BLDim)
{
    for (Int l=0; l<rowStride; ++l)
    {
        const Int rowShift = Shift_(l, rowAlign, rowStride);
        const Int localWidth = Length_(width, rowShift, rowStride);
        for (Int k=0; k<colStride; ++k)
        {
            const Int colShift = Shift_(k, colAlign, colStride);
            const Int localHeight = Length_(height, colShift, colStride);
            InterleaveMatrix<T,D>
            (localHeight, localWidth,
              &APortions[(k+l*colStride)*portionSize], 1, localHeight,
              &B[colShift+rowShift*BLDim], colStride, rowStride*BLDim);
        }
    }
}

} // namespace util
} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_UTIL_HPP

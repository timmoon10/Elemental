/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_AXPY_UTIL_HPP
#define EL_BLAS_AXPY_UTIL_HPP

namespace El {
namespace axpy {
namespace util {

// Serial axpy on a contiguous array
template<typename T> inline
void SerialArrayAxpy
( T a, Int n, const T* EL_RESTRICT x, T* EL_RESTRICT y )
{
    EL_SIMD
    for( Int i=0; i<n; ++i ) { y[i] += a * x[i]; }
}

// Axpy on a contiguous array
template<typename T>
void ArrayAxpy( T a, Int n, const T* EL_RESTRICT x, T* EL_RESTRICT y )
{
#ifdef _OPENMP
    const Int lineSize = Max( 64 / sizeof(T), 1 ); // Assuming 64B cache lines
    if( n > 4 * lineSize )
    {

        // Find cache lines
        std::size_t alignedSize = n * sizeof(T);
        void* alignedPtr = reinterpret_cast<void*>(y);
        std::align( lineSize * sizeof(T), sizeof(T), alignedPtr, alignedSize );
        const Int offset = ( alignedPtr != nullptr ?
                             reinterpret_cast<T*>(alignedPtr) - y :
                             0 );
        const Int firstLine = (offset > 0) ? offset - lineSize : 0;
        const Int numLines = (n - firstLine + lineSize - 1) / lineSize;

        // Distribute cache lines amongst threads
        const Int maxThreads = omp_get_max_threads();
        const Int linesPerThread = (numLines + maxThreads - 1) / maxThreads;
        const Int sizePerThread = linesPerThread * lineSize;
            
        // Apply axpy to cache lines in parallel
        #pragma omp parallel
        {
            const Int thread = omp_get_thread_num();
            const Int front = Max(firstLine + thread * sizePerThread, Int(0));
            const Int back = Min(firstLine + (thread+1) * sizePerThread, n);
            SerialArrayAxpy( a, back - front, &x[front], &y[front] );
        }

    }
    else
#endif // _OPENMP
    {
        SerialArrayAxpy( a, n, x, y );
    }
}

template<typename T>
void MatrixAxpy
( T alpha, Int height, Int width,
  const T* EL_RESTRICT A, Int lda,
        T* EL_RESTRICT B, Int ldb )
{

    if( width == 1 || ( lda == height && ldb == height ) )
    {
        ArrayAxpy( alpha, height * width, A, B );
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            SerialArrayAxpy( alpha, height, &A[j*lda], &B[j*ldb] );
        }
    }
  
}

template<typename T>
void InterleaveMatrixUpdate
( T alpha, Int height, Int width,
  const T* A, Int colStrideA, Int rowStrideA,
        T* B, Int colStrideB, Int rowStrideB )
{
    // TODO: Add OpenMP parallelization and/or optimize
    for( Int j=0; j<width; ++j )
        blas::Axpy
        ( height, alpha,
          &A[rowStrideA*j], colStrideA,
          &B[rowStrideB*j], colStrideB );
}

template<typename T>
void UpdateWithLocalData
( T alpha, const ElementalMatrix<T>& A, DistMatrix<T,STAR,STAR>& B )
{
    EL_DEBUG_CSE
    axpy::util::InterleaveMatrixUpdate
    ( alpha, A.LocalHeight(), A.LocalWidth(),
      A.LockedBuffer(),
      1,             A.LDim(),
      B.Buffer(A.ColShift(),A.RowShift()),
      A.ColStride(), A.RowStride()*B.LDim() );
}

} // namespace util
} // namespace axpy
} // namespace El

#endif // ifndef EL_BLAS_AXPY_UTIL_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ZERO_HPP
#define EL_BLAS_ZERO_HPP

#ifdef HYDROGEN_HAVE_CUDA
#include "GPU/Fill.hpp"
#endif

namespace El {

template<typename T>
void Zero( AbstractMatrix<T>& A )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    const Int ALDim = A.LDim();
    T* ABuf = A.Buffer();

    switch (A.GetDevice())
    {
    case Device::CPU:
        if( width == 1 || ALDim == height )
        {
#ifdef _OPENMP
            #pragma omp parallel
            {
                const Int numThreads = omp_get_num_threads();
                const Int thread = omp_get_thread_num();
                const Int chunk = (size + numThreads - 1) / numThreads;
                const Int start = Min(chunk * thread, size);
                const Int end = Min(chunk * (thread + 1), size);
                MemZero( &ABuf[start], end - start );
            }
#else
            MemZero( ABuf, size );
#endif
        }
        else
        {
            EL_PARALLEL_FOR
            for( Int j=0; j<width; ++j )
            {
                MemZero( &ABuf[j*ALDim], height );
            }
        }
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        Fill_GPU_impl(height, width, T(0), ABuf, ALDim);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Bad device type in Zero");
    }

}

template<typename T>
void Zero( AbstractDistMatrix<T>& A )
{
    EL_DEBUG_CSE
    Zero( A.Matrix() );
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Zero( AbstractMatrix<T>& A ); \
  EL_EXTERN template void Zero( AbstractDistMatrix<T>& A );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ZERO_HPP

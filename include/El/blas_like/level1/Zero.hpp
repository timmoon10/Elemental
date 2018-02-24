/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ZERO_HPP
#define EL_BLAS_ZERO_HPP

namespace El {

template<typename T>
void Zero( AbstractMatrix<T>& A )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int ALDim = A.LDim();
    T* ABuf = A.Buffer();

    // Zero out all entries if memory is contiguous. Otherwise zero
    // out each column.
    if( ALDim == height )
    {
        switch (A.GetDevice())
        {
        case Device::CPU:
            MemZero( ABuf, height*width );
            break;
        case Device::GPU:
#ifdef HYDROGEN_HAVE_CUDA
            if (cudaMemset(ABuf,0x0,height*width*sizeof(T)) != cudaSuccess)
                RuntimeError("Something wrong with cudaMemset");
            break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("Bad device type for Zero");
        }
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            switch (A.GetDevice())
            {
            case Device::CPU:
                MemZero( &ABuf[j*ALDim], height );
                break;
            case Device::GPU:
#ifdef HYDROGEN_HAVE_CUDA
                if (cudaMemset(ABuf+j*ALDim,0x0,height*sizeof(T)) != cudaSuccess)
                    RuntimeError("Something wrong with cudaMemset");
                break;
#endif // HYDROGEN_HAVE_CUDA
            default:
                LogicError("Bad device type for Zero");
            }
        }
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

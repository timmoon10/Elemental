/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_FILL_HPP
#define EL_BLAS_FILL_HPP

#ifdef HYDROGEN_HAVE_CUDA
#include "GPU/Fill.hpp"
#include <thrust/device_ptr.h>
#endif

namespace El
{
struct FillDispatch
{
    template <typename... Ts>
    static void Call(Ts&&... args)
    {
        Fill_GPU_impl(std::forward<Ts>(args)...);
    }
};// FillDispatch

template<typename T>
void Fill( AbstractMatrix<T>& A, T alpha )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    // Iterate over single loop if memory is contiguous. Otherwise
    // iterate over double loop.
    if( ALDim == m )
    {
        switch (A.GetDevice())
        {
        case Device::CPU:
        {
            for( Int i=0; i<m*n; ++i )
            {
                ABuf[i] = alpha;
            }
        }
        break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
        {
            constexpr bool valid_type = IsDeviceValidType_v<T,Device::GPU>();
            using Dispatcher =
                typename std::conditional<valid_type,
                                          FillDispatch,
                                          BadDeviceDispatch>::type;
            Dispatcher::Call(ABuf, m*n, alpha);
        }
        break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("Bad device type in Fill");
        }
    }
    else
    {
        for( Int j=0; j<n; ++j )
        {
            switch(A.GetDevice())
            {
            case Device::CPU:
            {
                for( Int i=0; i<m; ++i )
                {
                    ABuf[i+j*ALDim] = alpha;
                }
            }
            break;
#ifdef HYDROGEN_HAVE_CUDA
            case Device::GPU:
            {
                // FIXME: probably faster to do both loops on GPU!
                constexpr bool valid_type = IsDeviceValidType_v<T,Device::GPU>();
                using Dispatcher =
                    typename std::conditional<valid_type,
                                              FillDispatch,
                                              BadDeviceDispatch>::type;
                Dispatcher::Call(ABuf + j*ALDim, m, alpha);
            }
            break;
#endif // HYDROGEN_HAVE_CUDA
            default:
                LogicError("Bad device type in Fill");
            }
        }
    }
}

template<typename T>
void Fill( AbstractDistMatrix<T>& A, T alpha )
{
    EL_DEBUG_CSE
    Fill( A.Matrix(), alpha );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Fill( AbstractMatrix<T>& A, T alpha ); \
  EL_EXTERN template void Fill( AbstractDistMatrix<T>& A, T alpha );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_FILL_HPP

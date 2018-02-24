/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ALLREDUCE_HPP
#define EL_BLAS_ALLREDUCE_HPP

#ifdef HYDROGEN_HAVE_CUDA
#include "GPU/AllReduce.hpp"
#endif

namespace El
{

struct AllReduceDispatch
{
    template <typename... Ts>
    static void Call(Ts&&... args)
    {
        AllReduce_GPU_impl(std::forward<Ts>(args)...);
    }
};// AllReduceDispatch

template<typename T>
void AllReduce( AbstractMatrix<T>& A, mpi::Comm comm, mpi::Op op )
{
    EL_DEBUG_CSE
    if( mpi::Size(comm) == 1 )
        return;
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height*width;
    if( height == A.LDim() )
    {
        mpi::AllReduce( A.Buffer(), size, op, comm );
    }
    else
    {
        switch (A.GetDevice())
        {
        case Device::CPU:
        {
            vector<T> buf;
            FastResize( buf, size );

            // Pack
            copy::util::InterleaveMatrix
                ( height, width,
                  A.LockedBuffer(), 1, A.LDim(),
                  buf.data(),       1, height );

            mpi::AllReduce( buf.data(), size, op, comm );

            // Unpack
            copy::util::InterleaveMatrix
                ( height,        width,
                  buf.data(), 1, height,
                  A.Buffer(), 1, A.LDim() );
        }
        break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
        {
            constexpr bool valid_type = IsDeviceValidType_v<T,Device::GPU>();
            using Dispatcher =
                typename std::conditional<valid_type,
                                          AllReduceDispatch,
                                          BadDeviceDispatch>::type;
            Dispatcher::Call(static_cast<Matrix<T,Device::GPU>&>(A), comm, op);
        }
        break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("Bad device in AllReduce");
        }
    }
}

template<typename T>
void AllReduce( AbstractDistMatrix<T>& A, mpi::Comm comm, mpi::Op op )
{
    EL_DEBUG_CSE
    if( mpi::Size(comm) == 1 )
        return;
    if( !A.Participating() )
        return;

    AllReduce(A.Matrix(), comm, op);
#if 0
    const Int localHeight = A.LocalHeight();
    const Int localWidth = A.LocalWidth();
    const Int localSize = localHeight*localWidth;
    if( localHeight == A.LDim() )
    {
        mpi::AllReduce( A.Buffer(), localSize, op, comm );
    }
    else
    {
        vector<T> buf;
        FastResize( buf, localSize );

        // Pack
        copy::util::InterleaveMatrix
        ( localHeight, localWidth,
          A.LockedBuffer(), 1, A.LDim(),
          buf.data(),       1, localHeight );

        mpi::AllReduce( buf.data(), localSize, op, comm );

        // Unpack
        copy::util::InterleaveMatrix
        ( localHeight, localWidth,
          buf.data(), 1, localHeight,
          A.Buffer(), 1, A.LDim() );
    }
#endif // 0
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void AllReduce \
  ( AbstractMatrix<T>& A, mpi::Comm comm, mpi::Op op ); \
  EL_EXTERN template void AllReduce \
  ( AbstractDistMatrix<T>& A, mpi::Comm comm, mpi::Op op );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_ALLREDUCE_HPP

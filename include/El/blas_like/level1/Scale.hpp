/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_SCALE_HPP
#define EL_BLAS_SCALE_HPP

#ifdef HYDROGEN_ENABLE_CUDA
#include "GPU/Scale.hpp"
#endif

namespace El
{

template<typename T,typename S>
void Scale( S alphaS, AbstractMatrix<T>& A )
{
    EL_DEBUG_CSE
    const T alpha = T(alphaS);

    const Int ALDim = A.LDim();
    const Int height = A.Height();
    const Int width = A.Width();
    T* ABuf = A.Buffer();

    // TODO(poulson): Use imatcopy if MKL or OpenBLAS is detected

    if( alpha == T(0) )
    {
        Zero( A );
    }
    else if( alpha != T(1) )
    {
        if( ALDim == height )
        {
            switch (A.GetDevice())
            {
            case Device::CPU:
            {
                EL_PARALLEL_FOR
                for( Int i=0; i<height*width; ++i )
                    ABuf[i] *= alpha;
            }
            break;
#ifdef HYDROGEN_ENABLE_CUDA
            case Device::GPU:
                Scale_GPU_impl(ABuf, ABuf, height*width, alphaS);
                break;
#endif // HYDROGEN_ENABLE_CUDA
            default:
                LogicError("Bad device for scale!");
            }
        }
        else
        {
            switch (A.GetDevice())
            {
            case Device::CPU:
            {
                EL_PARALLEL_FOR
                for( Int j=0; j<width; ++j )
                {
                    EL_SIMD
                    for( Int i=0; i<height; ++i )
                    {
                        ABuf[i+j*ALDim] *= alpha;
                    }
                }
            }
            break;
#ifdef HYDROGEN_ENABLE_CUDA
            case Device::GPU:
            {
                for( Int j=0; j<width; ++j )
                {
                    // FIXME: Probably faster to do both loops on GPU!
                    Scale_GPU_impl(
                        ABuf + j*ALDim, ABuf + j*ALDim, height, alphaS);
                }
            }
            break;
#endif // HYDROGEN_ENABLE_CUDA
            default:
                LogicError("Bad device for scale");
            }
        }
    }
}

template<typename Real,typename S,typename>
void Scale( S alphaS, AbstractMatrix<Real>& AReal, AbstractMatrix<Real>& AImag )
{
    EL_DEBUG_CSE
    typedef Complex<Real> C;
    const C alpha = C(alphaS);
    if( alpha != C(1) )
    {
        if( alpha == C(0) )
        {
            Zero( AReal );
            Zero( AImag );
        }
        else
        {
            const Real alphaReal=alpha.real(), alphaImag=alpha.imag();
            Matrix<Real> ARealCopy;
            Copy( AReal, ARealCopy );
            Scale( alphaReal, AReal );
            Axpy( -alphaImag, AImag, AReal );
            Scale( alphaReal, AImag );
            Axpy( alphaImag, ARealCopy, AImag );
        }
    }
}

template<typename T,typename S>
void Scale( S alpha, AbstractDistMatrix<T>& A )
{
    EL_DEBUG_CSE
    Scale( alpha, A.Matrix() );
}

template<typename Real,typename S,typename>
void Scale( S alpha, AbstractDistMatrix<Real>& AReal,
                     AbstractDistMatrix<Real>& AImag )
{
    EL_DEBUG_CSE
    Scale( alpha, AReal.Matrix(), AImag.Matrix() );
}


#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void Scale \
  ( T alpha, AbstractMatrix<T>& A ); \
  EL_EXTERN template void Scale \
  ( T alpha, AbstractDistMatrix<T>& A );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_SCALE_HPP

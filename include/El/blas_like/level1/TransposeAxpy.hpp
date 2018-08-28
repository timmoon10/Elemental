/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_TRANSPOSEAXPY_HPP
#define EL_BLAS_TRANSPOSEAXPY_HPP

namespace El {

template <typename T, typename S>
void TransposeAxpy(
    S alphaS, AbstractMatrix<T> const& X, AbstractMatrix<T>& Y, bool conjugate)
{
    EL_DEBUG_CSE
    if (X.GetDevice() != Y.GetDevice())
        LogicError("X and Y must have same device for TransposeAxpy.");

    switch (X.GetDevice())
    {
    case Device::CPU:
        TransposeAxpy(alphaS,
                      static_cast<Matrix<T,Device::CPU> const&>(X),
                      static_cast<Matrix<T,Device::CPU>&>(Y),
                      conjugate);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        TransposeAxpy(alphaS,
                      static_cast<Matrix<T,Device::GPU> const&>(X),
                      static_cast<Matrix<T,Device::GPU>&>(Y),
                      conjugate);

        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Bad device for TransposeAxpy");
    }
}

template<typename T,typename S>
void TransposeAxpy
(       S alphaS,
  const Matrix<T>& X,
        Matrix<T>& Y,
        bool conjugate )
{
    EL_DEBUG_CSE
    const T alpha = T(alphaS);
    const Int mX = X.Height();
    const Int nX = X.Width();
    const Int nY = Y.Width();
    const Int ldX = X.LDim();
    const Int ldY = Y.LDim();
    const T* XBuf = X.LockedBuffer();
          T* YBuf = Y.Buffer();
    // If X and Y are vectors, we can allow one to be a column and the other
    // to be a row. Otherwise we force X and Y to be the same dimension.
    if( mX == 1 || nX == 1 )
    {
        const Int lengthX = ( nX==1 ? mX : nX );
        const Int incX = ( nX==1 ? 1  : ldX );
        const Int incY = ( nY==1 ? 1  : ldY );
        EL_DEBUG_ONLY(
          const Int mY = Y.Height();
          const Int lengthY = ( nY==1 ? mY : nY );
          if( lengthX != lengthY )
              LogicError("Nonconformal TransposeAxpy");
        )
        if( conjugate )
            for( Int j=0; j<lengthX; ++j )
                YBuf[j*incY] += alpha*Conj(XBuf[j*incX]);
        else
            blas::Axpy( lengthX, alpha, XBuf, incX, YBuf, incY );
    }
    else
    {
        EL_DEBUG_ONLY(
          const Int mY = Y.Height();
          if( mX != nY || nX != mY )
              LogicError("Nonconformal TransposeAxpy");
        )
        if( nX <= mX )
        {
            if( conjugate )
                for( Int j=0; j<nX; ++j )
                    for( Int i=0; i<mX; ++i )
                        YBuf[j+i*ldY] += alpha*Conj(XBuf[i+j*ldX]);
            else
                for( Int j=0; j<nX; ++j )
                    blas::Axpy( mX, alpha, &XBuf[j*ldX], 1, &YBuf[j], ldY );
        }
        else
        {
            if( conjugate )
                for( Int i=0; i<mX; ++i )
                    for( Int j=0; j<nX; ++j )
                        YBuf[j+i*ldY] += alpha*Conj(XBuf[i+j*ldX]);
            else
                for( Int i=0; i<mX; ++i )
                    blas::Axpy( nX, alpha, &XBuf[i], ldX, &YBuf[i*ldY], 1 );
        }
    }
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T, typename S,
          typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void TransposeAxpy(S alphaS,
                   Matrix<T,Device::GPU> const& X,
                   Matrix<T,Device::GPU>& Y,
                   bool conjugate)
{
    EL_DEBUG_CSE
    const T alpha = T(alphaS);
    const Int mX = X.Height();
    const Int nX = X.Width();
    const Int nY = Y.Width();
    const Int ldX = X.LDim();
    const Int ldY = Y.LDim();
    const T* XBuf = X.LockedBuffer();
    T* YBuf = Y.Buffer();

#ifndef EL_RELEASE
    if (conjugate)
        std::cerr << "TransposeAxpy: Conjugate not supported on GPU.\n"
                  << "  However, the type should be real anyway." << std::endl;
#endif // !EL_RELEASE

    SyncInfo<Device::GPU> syncInfoA(X), syncInfoB(Y);
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    // Keep the old stream so we can restore it. I don't know if this
    // is necessary, but it might be good to keep the cuBLAS handle
    // "looking const" outside this function.
    cudaStream_t old_stream;
    EL_CHECK_CUBLAS(
        cublasGetStream(GPUManager::cuBLASHandle(), &old_stream));
    EL_CHECK_CUBLAS(
        cublasSetStream(GPUManager::cuBLASHandle(), syncInfoB.stream_));


    // If X and Y are vectors, we can allow one to be a column and the other
    // to be a row. Otherwise we force X and Y to be the same dimension.
    if( mX == 1 || nX == 1 )
    {
        const Int lengthX = ( nX==1 ? mX : nX );
        const Int incX = ( nX==1 ? 1  : ldX );
        const Int incY = ( nY==1 ? 1  : ldY );
#ifndef EL_RELEASE
        const Int mY = Y.Height();
        const Int lengthY = ( nY==1 ? mY : nY );
        if( lengthX != lengthY )
            LogicError("Nonconformal TransposeAxpy");
#endif // !EL_RELEASE

        cublas::Axpy( lengthX, alpha, XBuf, incX, YBuf, incY );

    }
    else
    {
        EL_DEBUG_ONLY(
          const Int mY = Y.Height();
          if( mX != nY || nX != mY )
              LogicError("Nonconformal TransposeAxpy");
        )
        cublas::Geam(conjugate ? 'C' : 'T', 'N', nX, mX,
                     alpha, XBuf, ldX,
                     T(1), YBuf, ldY, YBuf, ldY);
    }
    // Restore the "default" stream
    EL_CHECK_CUBLAS(
        cublasSetStream(GPUManager::cuBLASHandle(), old_stream));
}

template <typename T, typename S,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void TransposeAxpy (S alphaS,
                    Matrix<T,Device::GPU> const& X,
                    Matrix<T,Device::GPU>& Y,
                    bool conjugate )
{
    LogicError("TransposeAxpy: Bad type/device combo.");
}
#endif // HYDROGEN_HAVE_CUDA

template<typename T,typename S>
void TransposeAxpy
(       S alphaS,
  const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
        bool conjugate )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( A, B );
      if( A.Height() != B.Width() || A.Width() != B.Height() )
          LogicError("A and B must have transposed dimensions");
    )
    const T alpha = T(alphaS);

    const DistData& ADistData = A.DistData();
    const DistData& BDistData = B.DistData();
    if( ADistData.colDist == BDistData.rowDist &&
        ADistData.rowDist == BDistData.colDist &&
        ADistData.colAlign==BDistData.rowAlign &&
        ADistData.rowAlign==BDistData.colAlign )
    {
        TransposeAxpy( alpha, A.LockedMatrix(), B.Matrix(), conjugate );
    }
    else
    {
        unique_ptr<ElementalMatrix<T>>
            C( B.ConstructTranspose(A.Grid(),A.Root()) );
        C->AlignRowsWith( B.DistData() );
        C->AlignColsWith( B.DistData() );
        Copy( A, *C );
        TransposeAxpy( alpha, C->LockedMatrix(), B.Matrix(), conjugate );
    }
}

template<typename T,typename S>
void AdjointAxpy( S alphaS, const Matrix<T>& X, Matrix<T>& Y )
{
    EL_DEBUG_CSE
    TransposeAxpy( alphaS, X, Y, true );
}

template<typename T,typename S>
void AdjointAxpy
( S alphaS, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y )
{
    EL_DEBUG_CSE
    TransposeAxpy( alphaS, X, Y, true );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO_TYPES(T,S) \
  EL_EXTERN template void TransposeAxpy \
  (       S alpha, \
    const AbstractMatrix<T>& A, \
          AbstractMatrix<T>& B, \
          bool conjugate ); \
  EL_EXTERN template void TransposeAxpy \
  (       S alpha, \
    const Matrix<T>& A, \
          Matrix<T>& B, \
          bool conjugate ); \
  EL_EXTERN template void TransposeAxpy \
  (       S alpha, \
    const ElementalMatrix<T>& A, \
          ElementalMatrix<T>& B, \
          bool conjugate ); \
  EL_EXTERN template void AdjointAxpy \
  (       S alpha, \
    const Matrix<T>& A, \
          Matrix<T>& B ); \
  EL_EXTERN template void AdjointAxpy \
  (       S alpha, \
    const ElementalMatrix<T>& A, \
          ElementalMatrix<T>& B );

#define PROTO_INT(T) PROTO_TYPES(T,T)

#define PROTO_REAL(T) \
  PROTO_TYPES(T,Int) \
  PROTO_TYPES(T,T)

#define PROTO_COMPLEX(T) \
  PROTO_TYPES(T,Int) \
  PROTO_TYPES(T,Base<T>) \
  PROTO_TYPES(T,T)

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef PROTO_TYPES
#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_TRANSPOSEAXPY_HPP

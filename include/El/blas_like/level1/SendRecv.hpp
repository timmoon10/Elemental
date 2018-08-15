/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_SENDRECV_HPP
#define EL_BLAS_SENDRECV_HPP

namespace El
{
template <typename T>
void SendRecv(
    AbstractMatrix<T> const& A, AbstractMatrix<T>& B,
    mpi::Comm comm, int sendRank, int recvRank)
{
    if (A.GetDevice() != B.GetDevice())
        LogicError("SendRecv: Matrices must be on the same device.");

    switch (A.GetDevice())
    {
    case Device::CPU:
        SendRecv(
            static_cast<Matrix<T,Device::CPU> const&>(A),
            static_cast<Matrix<T,Device::CPU>&>(B),
            comm, sendRank, recvRank);
        break;
    case Device::GPU:
        SendRecv(
            static_cast<Matrix<T,Device::GPU> const&>(A),
            static_cast<Matrix<T,Device::GPU>&>(B),
            comm, sendRank, recvRank);
        break;
    default:
        LogicError("SendRecv: Unsupported device.");
    }
}

template <typename T, Device D>
void SendRecv
( Matrix<T,D> const& A, Matrix<T,D>& B,
  mpi::Comm comm, int sendRank, int recvRank )
{
    EL_DEBUG_CSE
    const Int heightA = A.Height();
    const Int heightB = B.Height();
    const Int widthA = A.Width();
    const Int widthB = B.Width();
    const Int sizeA = heightA*widthA;
    const Int sizeB = heightB*widthB;

    SyncInfo<D> syncInfoA(A), syncInfoB(B);

    if (heightA == A.LDim() && heightB == B.LDim())
    {
        mpi::SendRecv(
            A.LockedBuffer(), sizeA, sendRank,
            B.Buffer(),       sizeB, recvRank, comm);
    }
    else if( heightA == A.LDim() )
    {
        simple_buffer<T,D> recvBuf(sizeB);

        mpi::SendRecv(
             A.LockedBuffer(), sizeA, sendRank,
             recvBuf.data(),   sizeB, recvRank, comm);

        copy::util::InterleaveMatrix(
            heightB, widthB,
            recvBuf.data(), 1, heightB,
            B.Buffer(),     1, B.LDim(), syncInfoB);
    }
    else
    {
        simple_buffer<T,D> sendBuf(sizeA);

        copy::util::InterleaveMatrix(
            heightA, widthA,
            A.LockedBuffer(), 1, A.LDim(),
            sendBuf.data(),   1, heightA, syncInfoA);

        simple_buffer<T,D> recvBuf(sizeB);

        mpi::SendRecv(
            sendBuf.data(), sizeA, sendRank,
            recvBuf.data(), sizeB, recvRank, comm);
        copy::util::InterleaveMatrix(
            heightB, widthB,
            recvBuf.data(), 1, heightB,
            B.Buffer(),     1, B.LDim(), syncInfoB);
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) \
  EL_EXTERN template void SendRecv \
  ( const Matrix<T,Device::CPU>& A, Matrix<T,Device::CPU>& B, mpi::Comm comm, \
    int sendRank, int recvRank );

EL_EXTERN template void SendRecv(
    Matrix<float,Device::GPU> const&, Matrix<float,Device::GPU>&,
    mpi::Comm, int, int);
EL_EXTERN template void SendRecv(
    Matrix<double,Device::GPU> const&, Matrix<double,Device::GPU>&,
    mpi::Comm, int, int);
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_BLAS_SENDRECV_HPP

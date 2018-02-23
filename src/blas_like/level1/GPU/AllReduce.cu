#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/AllReduce.hpp>
#include <El/blas_like/level1/GPU/BasicInterleaveMatrix.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace El
{

template <typename T>
void AllReduce_GPU_impl(
    Matrix<T,Device::GPU>& A, mpi::Comm comm, mpi::Op op)
{
    auto const height = A.Height();
    auto const width = A.Width();
    auto const size = height * width;
    auto const ldim = A.LDim();

    thrust::device_vector<T> buf(size);

    // Copy into the buffer
    BasicInterleaveMatrix_GPU_Gather(
        thrust::device_ptr<T const>(A.LockedBuffer()), buf,
        height, width, ldim);

    mpi::AllReduce(buf.data().get(), size, op, comm);

    // Copy into the matrix
    BasicInterleaveMatrix_GPU_Scatter(
        buf, thrust::device_ptr<T>(A.Buffer()), height, width, ldim);
}

template void AllReduce_GPU_impl(
    Matrix<float,Device::GPU>& A, mpi::Comm comm, mpi::Op op);
template void AllReduce_GPU_impl(
    Matrix<double,Device::GPU>& A, mpi::Comm comm, mpi::Op op);

}// namespace El

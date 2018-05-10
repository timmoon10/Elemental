#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/Hadamard.hpp>
#include <El/core/imports/cuda.hpp>

namespace
{

template <typename T>
__global__ void Hadamard1D_kernel( size_t size,
                                   T const* __restrict__ X,
                                   T const* __restrict__ Y,
                                   T* __restrict__ Z)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads) {
        Z[pos] = X[pos] * Y[pos];
    }
}

template <typename T>
__global__ void MultAssign_kernel( size_t size, T const* X, T* Y)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads) {
        Y[pos] *= X[pos];
    }
}

template <typename T>
__global__ void Hadamard2D_kernel( size_t height, size_t width,
                                   T const* X, size_t colStrideX, size_t rowStrideX,
                                   T const* Y, size_t colStrideY, size_t rowStrideY,
                                   T* Z, size_t colStrideZ, size_t rowStrideZ )
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads) {
        const size_t i = pos % height;
        const size_t j = pos / height;
        const auto& x_local = X[i*colStrideX+j*rowStrideX];
        const auto& y_local = Y[i*colStrideY+j*rowStrideY];
        Z[i*colStrideZ+j*rowStrideZ] = x_local * y_local;
    }
}

}// namespace <anon>

namespace El
{

template <typename T, typename>
void Hadamard_GPU_impl(
    size_t height, size_t width,
    T const* X, size_t colStrideX, size_t rowStrideX,
    T const* Y, size_t colStrideY, size_t rowStrideY,
    T* Z, size_t colStrideZ, size_t rowStrideZ )
{
    if( height <= 0 || width <= 0 ) { return; }
    const size_t size = height * width;
    const size_t blockDim = 256;
    const size_t gridDim = (size + blockDim - 1) / blockDim;
    auto stream = GPUManager::Stream();
    if( colStrideX == 1 && rowStrideX == height
        && colStrideY == 1 && rowStrideY == height
        && colStrideZ == 1 && rowStrideZ == height )
    {
        if( X == Z )
        {
            EL_CHECK_CUDA_KERNEL( MultAssign_kernel<T>,
                                  gridDim, blockDim, 0, stream,
                                  ( size, Y, Z ) );
        }
        else if( Y == Z )
        {
            EL_CHECK_CUDA_KERNEL( MultAssign_kernel<T>,
                                  gridDim, blockDim, 0, stream,
                                  ( size, X, Z ) );
        }
        else
        {
            EL_CHECK_CUDA_KERNEL( Hadamard1D_kernel<T>,
                                  gridDim, blockDim, 0, stream,
                                  ( size, X, Y, Z ) );
        }
    }
    else
    {
        EL_CHECK_CUDA_KERNEL( Hadamard2D_kernel<T>,
                              gridDim, blockDim, 0, stream,
                              ( height, width,
                                X, colStrideX, rowStrideX,
                                Y, colStrideY, rowStrideY,
                                Z, colStrideZ, rowStrideZ ) );
    }

}

template void Hadamard_GPU_impl(
    size_t, size_t,
    float const*, size_t, size_t,
    float const*, size_t, size_t,
    float*, size_t, size_t);
template void Hadamard_GPU_impl(
    size_t, size_t,
    double const*, size_t, size_t,
    double const*, size_t, size_t,
    double*, size_t, size_t);

}// namespace El

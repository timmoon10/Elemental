#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/Axpy.hpp>
#include <El/core/imports/cuda.hpp>

namespace
{

template <typename T>
__global__ void Axpy_kernel( size_t height, size_t width,
                             T alpha,
                             T const* X, size_t colStrideX, size_t rowStrideX,
                             T* Y, size_t colStrideY, size_t rowStrideY )
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads) {
        const size_t i = pos % height;
        const size_t j = pos / height;
        Y[i*colStrideY+j*rowStrideY] += alpha * X[i*colStrideX+j*rowStrideX];
    }
}

}// namespace <anon>

namespace El
{

template <typename T, typename>
void Axpy_GPU_impl(
    size_t height, size_t width,
    T const& alpha,
    T const* X, size_t colStrideX, size_t rowStrideX,
    T* Y, size_t colStrideY, size_t rowStrideY )
{
    if( height <= 0 || width <= 0 ) { return; }
    const size_t size = height * width;
    const size_t blockDim = 256;
    const size_t gridDim = (size + blockDim - 1) / blockDim;
    cudaStream_t stream = GPUManager::Stream();
    EL_CHECK_CUDA_KERNEL( Axpy_kernel<T>,
                          gridDim, blockDim, 0, stream,
                          ( height, width, alpha,
                            X, colStrideX, rowStrideX,
                            Y, colStrideY, rowStrideY ) );
}

template void Axpy_GPU_impl(
    size_t, size_t, float const&,
    float const*, size_t, size_t, float*, size_t, size_t);
template void Axpy_GPU_impl(
    size_t, size_t, double const&,
    double const*, size_t, size_t, double*, size_t, size_t);

}// namespace El

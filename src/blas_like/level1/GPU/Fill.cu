#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/Fill.hpp>
#include <El/core/imports/cuda.hpp>

namespace
{
  
template <typename T>
__global__ void Fill1D_kernel( size_t size, T value, T* buffer )
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads) {
        buffer[pos] = value;
    }
}

template <typename T>
__global__ void Fill2D_kernel( size_t height, size_t width, T value,
                               T* buffer, size_t ldim )
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads) {
        const size_t i = pos % height;
        const size_t j = pos / height;
        buffer[i+j*ldim] = value;
    }
}

}// namespace <anon>

namespace El
{

template <typename T, typename>
void Fill_GPU_impl(
    size_t height, size_t width, T const& value,
    T* buffer, size_t ldim )
{
    if( height <= 0 || width <= 0 ) { return; }
    
    const size_t size = height * width;
    const size_t blockDim = 256;
    const size_t gridDim = (size + blockDim - 1) / blockDim;
    cudaStream_t stream = GPUManager::Stream();
    if( value == T(0) )
    {
        if( width == 1 || ldim == height )
        {
            EL_CHECK_CUDA(cudaMemsetAsync( buffer, 0x0, size*sizeof(T),
                                           stream ));
        }
        else
        {
            EL_CHECK_CUDA(cudaMemset2DAsync( buffer, ldim*sizeof(T), 0x0,
                                             height*sizeof(T), width,
                                             stream ));
        }
    }
    else
    {
        if( width == 1 || ldim == height )
        {
            EL_CHECK_CUDA_KERNEL( Fill1D_kernel<T>,
                                  gridDim, blockDim, 0, stream,
                                  ( size, value, buffer ) );
        }
        else
        {
            EL_CHECK_CUDA_KERNEL( Fill2D_kernel<T>,
                                  gridDim, blockDim, 0, stream,
                                  ( height, width, value, buffer, ldim ) );
        }
    }

}

template void Fill_GPU_impl(
    size_t, size_t, float const&, float*, size_t);
template void Fill_GPU_impl(
    size_t, size_t, double const&, double*, size_t);

}// namespace El

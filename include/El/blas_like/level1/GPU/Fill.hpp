#ifndef EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

namespace El
{

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Fill_GPU_impl(size_t, size_t, T const&, T*, size_t,
                   cudaStream_t = GPUManager::Stream());

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Fill_GPU_impl(size_t, size_t, T const&, T*, size_t,
                   cudaStream_t = GPUManager::Stream())
{
    LogicError("Fill: Type not valid on GPU.");
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

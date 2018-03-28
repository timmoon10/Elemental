#ifndef EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_

namespace El
{

template <typename T, typename S,
          typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Scale_GPU_impl(
    T* Input, T* Output, size_t size, S const& alpha);

template <typename T, typename S,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Scale_GPU_impl(
    T* Input, T* Output, size_t size, S const& alpha)
{
    LogicError("Scale: Type not valid on GPU.");
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_

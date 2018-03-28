#ifndef EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

namespace El
{

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Fill_GPU_impl(T*, size_t, T const&);

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Fill_GPU_impl(T*, size_t, T const&)
{
    LogicError("Fill: Type not valid on GPU.");
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

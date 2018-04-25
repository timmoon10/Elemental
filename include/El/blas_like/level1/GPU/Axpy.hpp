#ifndef EL_BLAS_LIKE_LEVEL1_GPU_AXPY_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_AXPY_HPP_

namespace El
{

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Axpy_GPU_impl(
    size_t, size_t, T const&,
    T const*, size_t, size_t, T*, size_t, size_t);

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Axpy_GPU_impl(size_t, size_t, T const&, T*, size_t)
{
    LogicError("Axpy: Type not valid on GPU.");
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_AXPY_HPP_

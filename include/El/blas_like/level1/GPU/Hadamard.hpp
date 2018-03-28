#ifndef EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

namespace El
{

template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void Hadamard_GPU_impl(
    T const* Input0, T const* Input1, T* Output, size_t size);

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
void Hadamard_GPU_impl(
    T const* Input0, T const* Input1, T* Output, size_t size)
{
    LogicError("Hadamard: Type not valid on GPU.");
}


}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

#ifndef EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

#include <thrust/device_ptr.h>

namespace El
{

template <typename T>
void Hadamard_GPU_impl(
    T const* Input0, T const* Input1, T* Output, size_t size);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

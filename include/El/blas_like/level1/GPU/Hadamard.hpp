#ifndef EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

#include <thrust/device_ptr.h>

namespace El
{

template <typename T>
void Hadamard_GPU_impl(
    thrust::device_ptr<T> Input0, thrust::device_ptr<T> Input1,
    thrust::device_ptr<T> Output, size_t size);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_HADAMARD_HPP_

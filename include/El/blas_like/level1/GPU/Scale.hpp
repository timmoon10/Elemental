#ifndef EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_

#include <thrust/device_ptr.h>

namespace El
{

template <typename T, typename S>
void Scale_GPU_impl(
    T* Input, T* Output, size_t size, S const& alpha);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_SCALE_HPP_

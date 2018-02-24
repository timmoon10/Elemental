#ifndef EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

#include <thrust/device_ptr.h>

namespace El
{

template <typename T>
void Fill_GPU_impl(T*, size_t, T const&);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_FILL_HPP_

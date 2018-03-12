#ifndef EL_BLAS_LIKE_LEVEL1_GPU_TRANSPOSE_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_TRANSPOSE_HPP_

namespace El
{

template <typename T>
void Transpose_GPU_impl(
    T* Output, unsigned Output_LDim,
    T const* Input, unsigned Height, unsigned Width, unsigned LDim);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_TRANSPOSE_HPP_

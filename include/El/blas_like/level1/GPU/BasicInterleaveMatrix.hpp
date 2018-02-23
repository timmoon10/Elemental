#ifndef EL_BLAS_LIKE_LEVEL1_GPU_BASIC_INTERLEAVE_MATRIX_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_BASIC_INTERLEAVE_MATRIX_HPP_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace El
{

template <typename T>
void BasicInterleaveMatrix_GPU_Gather(
    thrust::device_ptr<T const>, thrust::device_vector<T>&,
    size_t const&, size_t const&, size_t const&);

template <typename T>
void BasicInterleaveMatrix_GPU_Scatter(
    thrust::device_vector<T> const&, thrust::device_ptr<T>,
    size_t const&, size_t const&, size_t const&);

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_BASIC_INTERLEAVE_MATRIX_HPP_

#include <El/blas_like/level1/GPU/Hadamard.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace El
{

template <typename T>
void Hadamard_GPU_impl(
    thrust::device_ptr<T> Input0, thrust::device_ptr<T> Input1,
    thrust::device_ptr<T> Output, size_t size)
{
    thrust::transform(thrust::device,
                      Input0, Input0 + size, Input1, Output,
                      thrust::multiplies<T>{});
}

template void Hadamard_GPU_impl(
    thrust::device_ptr<float>, thrust::device_ptr<float>,
    thrust::device_ptr<float>, size_t);
template void Hadamard_GPU_impl(
    thrust::device_ptr<double>, thrust::device_ptr<double>,
    thrust::device_ptr<double>, size_t);
}
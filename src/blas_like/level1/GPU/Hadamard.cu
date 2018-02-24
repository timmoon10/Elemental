#include <El/blas_like/level1/GPU/Hadamard.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace El
{

template <typename T>
void Hadamard_GPU_impl(
    T const* Input0, T const* Input1, T* Output, size_t size)
{
    thrust::transform(thrust::device,
                      Input0, Input0 + size, Input1, Output,
                      thrust::multiplies<T>{});
}

template void Hadamard_GPU_impl(
    float const*, float const*, float*, size_t);
template void Hadamard_GPU_impl(
    double const*, double const*, double*, size_t);
}
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/Fill.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace El
{

template <typename T>
void Fill_GPU_impl(
    thrust::device_ptr<T> buffer, size_t size, T const& value)
{
    thrust::fill_n(thrust::device, buffer, size, value);
}

template void Fill_GPU_impl(
    thrust::device_ptr<float>, size_t, float const&);
template void Fill_GPU_impl(
    thrust::device_ptr<double>, size_t, double const&);

}// namespace El

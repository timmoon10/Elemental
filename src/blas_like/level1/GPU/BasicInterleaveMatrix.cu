#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace El
{

template <typename T>
void BasicInterleaveMatrix_GPU_Gather(
    thrust::device_ptr<T const> src, thrust::device_vector<T>& dest,
    size_t const& srcHeight, size_t const& srcWidth, size_t const& srcLDim)
{
    dest.resize(srcHeight*srcWidth);

    auto iter = dest.begin();
    for (auto col = size_t{0}; col < srcWidth; ++col)
        iter = thrust::copy_n(
            thrust::device, src + col*srcLDim, srcHeight, iter);
}

template <typename T>
void BasicInterleaveMatrix_GPU_Scatter(
    thrust::device_vector<T> const& src, thrust::device_ptr<T> dest,
    size_t const& destHeight, size_t const& destWidth, size_t const& destLDim)
{
    auto iter = src.begin();
    for (auto col = size_t{0}; col < destWidth; ++col)
    {
        thrust::copy_n(
            thrust::device, iter, destHeight, dest + col*destLDim);
        iter += destHeight;
    }

}

template void BasicInterleaveMatrix_GPU_Gather(
    thrust::device_ptr<float const>, thrust::device_vector<float>&,
    size_t const&, size_t const&, size_t const&);
template void BasicInterleaveMatrix_GPU_Gather(
    thrust::device_ptr<double const>, thrust::device_vector<double>&,
    size_t const&, size_t const&, size_t const&);

template void BasicInterleaveMatrix_GPU_Scatter(
    thrust::device_vector<float> const&, thrust::device_ptr<float>,
    size_t const&, size_t const&, size_t const&);
template void BasicInterleaveMatrix_GPU_Scatter(
    thrust::device_vector<double> const&, thrust::device_ptr<double>,
    size_t const&, size_t const&, size_t const&);

}// namespace El
#include <El/blas_like/level1/GPU/Scale.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace El
{

namespace
{

// Quick functor for the a*x operation
template <typename S, typename T>
class ScaleBy
{
    const S alpha_;
public:
    ScaleBy(S alpha) : alpha_{alpha} {}

    __host__ __device__ T operator()(T const& x) { return x*alpha_; }
};// class ScaleBy

}// namespace <anon>

template <typename T, typename S>
void Scale_GPU_impl(
    T* Input, T* Output, size_t size, S const& alpha)
{
    thrust::transform(thrust::device,
                      Input, Input + size, Output, ScaleBy<S,T>{alpha});
}

template void Scale_GPU_impl(
    float*, float*, size_t, float const&);

template void Scale_GPU_impl(
    double*, double*, size_t, double const&);

}// namespace El
#ifndef EL_CORE_DEVICE_HPP_
#define EL_CORE_DEVICE_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

namespace El
{

// Special typedef to help distinguish host/device memory
template <typename T> using DevicePtr = T*;

enum class Device : unsigned char { CPU, GPU };

// A trait to determine if the given (scalar) type is valid for a
// given device type.
template <typename T, Device D>
struct IsDeviceValidType : std::false_type {};

template <typename T>
struct IsDeviceValidType<T,Device::CPU> : std::true_type {};

template <> struct IsDeviceValidType< float,Device::GPU> : std::true_type {};
template <> struct IsDeviceValidType<double,Device::GPU> : std::true_type {};

// Constexpr function wrapping the value above
template <typename T, Device D>
constexpr bool IsDeviceValidType_v() { return IsDeviceValidType<T,D>::value; }

struct BadDeviceDispatch
{
    template <typename... Ts>
    static void Call(Ts&&...)
    {
        LogicError("Bad device type!");
    }
};// struct BadDeviceDispatch

template <typename T, Device D> class simple_buffer;

template <typename T>
class simple_buffer<T,Device::CPU>
{
public:
    simple_buffer(size_t size)
    {
        vec_.reserve(size);
    }

    simple_buffer(size_t size, T const& value)
        : vec_(size, value)
    {}

    T* data() noexcept { return vec_.data(); }
private:
    std::vector<T> vec_;
};// class simple_buffer<T,Device::CPU>

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
class simple_buffer<T,Device::GPU>
{
public:
    simple_buffer(size_t size)
    {
        T* ptr;
        auto error = cudaMalloc(&ptr, size*sizeof(T));
        if (error != cudaSuccess)
            RuntimeError("simple_buffer: cudaMalloc failed with message: \"",
                         cudaGetErrorString(error), "\"");
        else
            data_ = ptr;
    }

    simple_buffer(size_t size, T const& value)
        : simple_buffer(size)
    {
        // FIXME
        if (value != T(0))
            LogicError("Cannot value-initialize to nonzero value on GPU.");

        auto error = cudaMemset(data_, value, size*sizeof(T));
        if (error != cudaSuccess)
            RuntimeError("simple_buffer: cudaMemset failed with message: \"",
                         cudaGetErrorString(error), "\"");
    }

    ~simple_buffer()
    {
        if (data_)
        {
            auto error = cudaFree(data_);
            if (error != cudaSuccess)
            {
                std::cerr << "Error in destructor. About to terminate.\n\n"
                          << "cudaError = " << cudaGetErrorString(error)
                          << std::endl;
                std::terminate();
            }
            data_ = nullptr;
        }
    }

    T* data() noexcept { return data_; }
private:
    T* data_ = nullptr;
};// class simple_buffer<T,Device::GPU>
#endif // HYDROGEN_HAVE_CUDA

}// namespace El
#endif // EL_CORE_DEVICE_HPP_

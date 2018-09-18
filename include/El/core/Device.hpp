#ifndef EL_CORE_DEVICE_HPP_
#define EL_CORE_DEVICE_HPP_

namespace El
{

// Special typedef to help distinguish host/device memory
template <typename T> using DevicePtr = T*;

enum class Device : unsigned char
{
    CPU
#ifdef HYDROGEN_HAVE_CUDA
    , GPU
#endif
};

template <Device D>
std::string DeviceName();

template <> inline std::string DeviceName<Device::CPU>()
{ return "CPU"; }

#ifdef HYDROGEN_HAVE_CUDA
template <> inline std::string DeviceName<Device::GPU>()
{ return "GPU"; }
#endif

// A trait to determine if the given (scalar) type is valid for a
// given device type.
template <typename T, Device D>
struct IsDeviceValidType : std::false_type {};

template <typename T>
struct IsDeviceValidType<T,Device::CPU> : std::true_type {};

#ifdef HYDROGEN_HAVE_CUDA
template <> struct IsDeviceValidType< float,Device::GPU> : std::true_type {};
template <> struct IsDeviceValidType<double,Device::GPU> : std::true_type {};
#endif

// Constexpr function wrapping the value above
template <typename T, Device D>
constexpr bool IsDeviceValidType_v() { return IsDeviceValidType<T,D>::value; }

// Predicate to test if two devices are the same
template <Device D1, Device D2>
using SameDevice = EnumSame<Device,D1,D2>;

// Basic inter-device memory operations
template <Device SrcD, Device DestD> struct InterDeviceCopy;

#ifdef HYDROGEN_HAVE_CUDA

template <Device D1, Device D2>
constexpr cudaMemcpyKind CUDAMemcpyKind();

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::CPU,Device::GPU>()
{
    return cudaMemcpyHostToDevice;
}

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::GPU,Device::CPU>()
{
    return cudaMemcpyDeviceToHost;
}

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::GPU,Device::GPU>()
{
    return cudaMemcpyDeviceToDevice;
}

template <>
struct InterDeviceCopy<Device::CPU,Device::GPU>
{
    template <typename T>
    static void MemCopy1DAsync(T * EL_RESTRICT const dest,
                               T const* EL_RESTRICT const src,
                               Int const size,
                               cudaStream_t stream = GPUManager::Stream())
    {
        EL_CHECK_CUDA(cudaMemcpyAsync(
                          dest, src, size*sizeof(T),
                          CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                          stream));
    }

    template <typename T>
    static void MemCopy2DAsync(T * EL_RESTRICT const dest, Int const dest_ldim,
                               T const* EL_RESTRICT const src,
                               Int const src_ldim,
                               Int const height, Int const width,
                               cudaStream_t stream = GPUManager::Stream())
    {
        EL_CHECK_CUDA(cudaMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            CUDAMemcpyKind<Device::CPU,Device::GPU>(),
            stream));
    }
};// InterDevice<CPU,GPU>

template <>
struct InterDeviceCopy<Device::GPU,Device::CPU>
{
    template <typename T>
    static void MemCopy1DAsync(T * EL_RESTRICT const dest,
                               T const* EL_RESTRICT const src, Int const size,
                               cudaStream_t stream = GPUManager::Stream())
    {
        EL_CHECK_CUDA(cudaMemcpyAsync(
                          dest, src, size*sizeof(T),
                          CUDAMemcpyKind<Device::GPU,Device::CPU>(),
                          stream));
    }

    template <typename T>
    static void MemCopy2DAsync(T * EL_RESTRICT const dest, Int const dest_ldim,
                               T const* EL_RESTRICT const src, Int const src_ldim,
                               Int const height, Int const width,
                               cudaStream_t stream = GPUManager::Stream())
    {
        EL_CHECK_CUDA(cudaMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            CUDAMemcpyKind<Device::GPU,Device::CPU>(),
            stream));
    }
};// InterDevice<CPU,GPU>
#endif // HYDROGEN_HAVE_CUDA

}// namespace El
#endif // EL_CORE_DEVICE_HPP_

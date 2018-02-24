#ifndef EL_CORE_DEVICE_HPP_
#define EL_CORE_DEVICE_HPP_

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

}// namespace El
#endif // EL_CORE_DEVICE_HPP_

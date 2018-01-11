#ifndef EL_CORE_DEVICE_HPP_
#define EL_CORE_DEVICE_HPP_

namespace El
{

// Special typedef to help distinguish host/device memory
template <typename T> using DevicePtr = T*;

enum class Device : unsigned char { CPU, GPU };

}// namespace El
#endif // EL_CORE_DEVICE_HPP_

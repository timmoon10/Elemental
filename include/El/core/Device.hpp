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

// A simple data management class for temporary contiguous memory blocks
template <typename T, Device D> class simple_buffer;

template <typename T>
class simple_buffer<T,Device::CPU>
{
public:
    simple_buffer() = default;

    simple_buffer(size_t size)
    {
        this->allocate(size);
    }

    simple_buffer(size_t size, T const& value)
        : vec_(size, value)
    {
        data_ = vec_.data();
    }

    void allocate(size_t size)
    {
        vec_.reserve(size);
        size_ = size;
        data_ = vec_.data();
    }

    size_t size() const noexcept
    {
        return size_;
    }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }

#ifdef HYDROGEN_HAVE_CUDA
    void shallowCopyIfPossible(simple_buffer<T,Device::GPU>& A)
    {
        // Shallow copy not possible
        this->allocate(A.size());
        auto stream = GPUManager::Stream();
        EL_CHECK_CUDA(cudaMemcpyAsync(data_, A.data(), size_*sizeof(T),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        EL_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
#endif // HYDROGEN_HAVE_CUDA

    void shallowCopyIfPossible(simple_buffer<T,Device::CPU>& A)
    {
        data_ = A.data();
        size_ = A.size();
    }

private:
    T* data_ = nullptr;// To be used as VIEW ONLY.
    std::vector<T> vec_;
    size_t size_ = 0;
};// class simple_buffer<T,Device::CPU>

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
class simple_buffer<T,Device::GPU>
{
public:
    simple_buffer() = default;

    simple_buffer(size_t size)
    {
        this->allocate(size);
    }

    simple_buffer(size_t size, T const& value)
        : simple_buffer(size)
    {
        if( value == T(0) )
        {
            EL_CHECK_CUDA(cudaMemsetAsync(data_, 0x0, size*sizeof(T),
                                          GPUManager::Stream()));
        }
        else
        {
            simple_buffer<T,Device::CPU> cpuBuffer(size, value);
            shallowCopyIfPossible(cpuBuffer);
        }
    }

    ~simple_buffer()
    {
        if (data_ && own_data_)
        {
            EL_CHECK_CUDA(cudaFree(data_));
            data_ = nullptr;
        }
    }

    void allocate(size_t size)
    {
        if (size < size_ && own_data_)
        {
            size_ = size;
            return;
        }

        T* ptr;
        EL_CHECK_CUDA(cudaMalloc(&ptr, size*sizeof(T)));
        std::swap(data_,ptr);
        const bool own_ptr = own_data_;
        own_data_ = true;
        size_ = size;
        if (ptr && own_ptr)
        {
            EL_CHECK_CUDA(cudaFree(ptr));
            ptr = nullptr;
        }
    }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }

    size_t size() const noexcept { return size_; }

    void shallowCopyIfPossible(simple_buffer<T,Device::CPU>& A)
    {
        // Shallow copy not possible
        this->allocate(A.size());
        auto stream = GPUManager::Stream();
        EL_CHECK_CUDA(cudaMemcpyAsync(data_, A.data(), size_*sizeof(T),
                                      cudaMemcpyHostToDevice,
                                      stream));
        EL_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    void shallowCopyIfPossible(simple_buffer<T,Device::GPU>& A)
    {
        if (own_data_)
            LogicError("Can't shallowCopy into buffer that owns data.");
        data_ = A.data();
        size_ = A.size();
    }
private:
    T* data_ = nullptr;
    size_t size_ = 0;
    bool own_data_ = false;
};// class simple_buffer<T,Device::GPU>


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

#endif // HYDROGEN_HAVE_CUDA

}// namespace El
#endif // EL_CORE_DEVICE_HPP_

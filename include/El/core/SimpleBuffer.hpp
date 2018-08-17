#ifndef EL_CORE_SIMPLEBUFFER_HPP_
#define EL_CORE_SIMPLEBUFFER_HPP_

namespace El
{

// A simple data management class for temporary contiguous memory blocks
template <typename T, Device D>
class simple_buffer
{
public:
    simple_buffer() = default;
    explicit simple_buffer(size_t size,
                           unsigned int mode = DefaultMemoryMode<D>(),
                           SyncInfo<D> const& = SyncInfo<D>{});
    explicit simple_buffer(size_t size, T const& value,
                           unsigned int mode = DefaultMemoryMode<D>(),
                           SyncInfo<D> const& = SyncInfo<D>{});

    void allocate(size_t size);

    size_t size() const noexcept;
    T* data() noexcept;
    T const* data() const noexcept;

private:
    Memory<T,D> mem_;
    T* data_;
    size_t size_;
}; // class simple_buffer


namespace details
{

template <typename T>
void setBufferToValue(T* buffer, size_t size, T const& value,
                      SyncInfo<Device::CPU> const& = SyncInfo<Device::CPU>{})
{
    std::fill_n(buffer, size, value);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
void setBufferToValue(T* buffer, size_t size, T const& value,
                      SyncInfo<Device::GPU> syncInfo = SyncInfo<Device::GPU>{})
{
    if( value == T(0) )
    {
        EL_CHECK_CUDA(cudaMemsetAsync(buffer, 0x0, size*sizeof(T),
                                      syncInfo.stream_));
    }
    else
    {
        std::vector<T> tmp(size, value);
        EL_CHECK_CUDA(
            cudaMemcpyAsync(
                buffer, tmp.data(), size*sizeof(T),
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                syncInfo.stream_));
    }
    AddSynchronizationPoint(syncInfo);
}
#endif // HYDROGEN_HAVE_CUDA
}// namespace details


template <typename T, Device D>
simple_buffer<T,D>::simple_buffer(
    size_t size, unsigned int mode, SyncInfo<D> const& syncInfo)
    : mem_{size, mode, syncInfo},
      data_{mem_.Buffer()},
      size_{mem_.Size()}
{}

template <typename T, Device D>
simple_buffer<T,D>::simple_buffer(
    size_t size, T const& value, unsigned mode, SyncInfo<D> const& syncInfo)
    : simple_buffer{size, mode, syncInfo}
{
    details::setBufferToValue(this->data(), size, value, syncInfo);
}

template <typename T, Device D>
void simple_buffer<T,D>::allocate(size_t size)
{
    data_ = mem_.Require(size);
    size_ = size;
}

template <typename T, Device D>
size_t simple_buffer<T,D>::size() const noexcept
{
    return size_;
}

template <typename T, Device D>
T* simple_buffer<T,D>::data() noexcept
{
    return data_;
}

template <typename T, Device D>
T const* simple_buffer<T,D>::data() const noexcept
{
    return data_;
}
}// namespace El
#endif // EL_CORE_SIMPLEBUFFER_HPP_

#ifndef EL_CORE_SYNCINFO_HPP_
#define EL_CORE_SYNCINFO_HPP_

namespace El
{

// Device-specific synchronization information. For CPUs, this is
// empty since all CPU operations are synchronous with respect to the
// host. For GPUs, this will be a stream and an associated event. It
// might be sufficient to just be a stream. Unclear.
//
// The use-case for this is to cope with the matrix-free part of the
// interface. Many of the copy routines have the paradigm that they
// take Matrix<T,D>s as arguments and then the host will organize and
// dispatch subkernels that operate on data buffers, i.e., T[]
// data. This provides a lightweight way to pass the CUDA stream
// through the T* interface without an entire matrix (which,
// semantically, may not make sense).
//
// This also might be useful for interacting with
// Aluminum/MPI/NCCL/whatever.
template <Device D> struct SyncInfo
{
    SyncInfo() {}

    template <typename T>
    SyncInfo(Matrix<T,D> const&) {}

};// struct SyncInfo<D>

template <Device D>
void AddSynchronizationPoint(SyncInfo<D> syncInfo)
{}

template <Device D>
void AddSynchronizationPoint(SyncInfo<D> A, SyncInfo<D> B)
{}

#ifdef HYDROGEN_HAVE_CUDA

template <>
struct SyncInfo<Device::GPU>
{
    SyncInfo() : SyncInfo{GPUManager::Stream(), GPUManager::Event()} {}

    SyncInfo(cudaStream_t stream, cudaEvent_t event)
        : stream_{stream}, event_{event} {}

    template <typename T>
    SyncInfo(Matrix<T,Device::GPU> const& A)
        : stream_{A.Stream()}, event_{A.Event()} {}

    cudaStream_t stream_;
    cudaEvent_t event_;
};// struct SyncInfo<Device::GPU>

inline void AddSynchronizationPoint(SyncInfo<Device::GPU> syncInfo)
{
    EL_CHECK_CUDA(cudaEventRecord(syncInfo.event_, syncInfo.stream_));
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::CPU> A, SyncInfo<Device::GPU> B)
{
    LogicError("I don't know what should happen here.");
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> A, SyncInfo<Device::CPU> B)
{
    LogicError("I don't know what should happen here.");
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> A, SyncInfo<Device::GPU> B)
{
    EL_CHECK_CUDA(cudaEventRecord(A.event_, A.stream_));
    EL_CHECK_CUDA(cudaStreamWaitEvent(B.stream_, A.event_, 0));
}
#endif // HYDROGEN_HAVE_CUDA

}// namespace El
#endif // EL_CORE_SYNCINFO_HPP_

#ifndef EL_CORE_SYNCINFO_HPP_
#define EL_CORE_SYNCINFO_HPP_

#include "IndexSequence.hpp"

namespace El
{

/** \class SyncInfo
 *  \brief Manage device-specific synchronization information.
 *
 *  Device-specific synchronization information. For CPUs, this is
 *  empty since all CPU operations are synchronous with respect to the
 *  host. For GPUs, this will be a stream and an associated event.
 *
 *  The use-case for this is to cope with the matrix-free part of the
 *  interface. Many of the copy routines have the paradigm that they
 *  take Matrix<T,D>s as arguments and then the host will organize and
 *  dispatch subkernels that operate on data buffers, i.e., T[]
 *  data. In the GPU case, for example, this provides a lightweight
 *  way to pass the CUDA stream through the T* interface without an
 *  entire matrix (which, semantically, may not make sense).
 *
 *  This also might be useful for interacting with
 *  Aluminum/MPI/NCCL/whatever. It essentially enables tagged
 *  dispatch, where the tags possibly contain some extra
 *  device-specific helpers.
 */
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

template <Device D>
void Synchronize(SyncInfo<D> const&)
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

inline void AddSynchronizationPoint(SyncInfo<Device::GPU> const& syncInfo)
{
    EL_CHECK_CUDA(cudaEventRecord(syncInfo.event_, syncInfo.stream_));
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::CPU> const& A, SyncInfo<Device::GPU> const& B)
{
    LogicError("I don't know what should happen here.");
}

inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> const& A, SyncInfo<Device::CPU> const& B)
{
    LogicError("I don't know what should happen here.");
}

// This captures the work done on A and forces B to wait for completion
inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> const& A, SyncInfo<Device::GPU> const& B)
{
    if (A.stream_ != B.stream_)
    {
        AddSynchronizationPoint(A);
        EL_CHECK_CUDA(cudaStreamWaitEvent(B.stream_, A.event_, 0));
    }
}

// This captures the work done on A and forces B and C to wait for completion
inline void AddSynchronizationPoint(
    SyncInfo<Device::GPU> const& A,
    SyncInfo<Device::GPU> const& B, SyncInfo<Device::GPU> const& C)
{
    bool const ABdiff = (A.stream_ == B.stream_);
    bool const ACdiff = (A.stream_ == C.stream_);

    if (ABdiff || ACdiff)
        AddSynchronizationPoint(A);

    if (ABdiff)
        EL_CHECK_CUDA(cudaStreamWaitEvent(B.stream_, A.event_, 0));

    if (ACdiff)
        EL_CHECK_CUDA(cudaStreamWaitEvent(C.stream_, A.event_, 0));
}

inline void Synchronize(SyncInfo<Device::GPU> const& syncInfo)
{
    EL_CHECK_CUDA(cudaStreamSynchronize(syncInfo.stream_));
}

#endif // HYDROGEN_HAVE_CUDA

template <Device D, Device... Ds>
void AllWaitOnMaster(
    SyncInfo<D> const& Master, SyncInfo<Ds> const&... Others)
{
    AddSynchronizationPoint(Master, Others...);
}

template <Device D>
void MasterWaitOnAll(SyncInfo<D> const& Master)
{}

template <Device D, Device D1, Device... Ds>
void MasterWaitOnAll(
    SyncInfo<D> const& Master, SyncInfo<D1> const& Other,
    SyncInfo<Ds> const&... others)
{
    AddSynchronizationPoint(Other, Master);
    MasterWaitOnAll(Master, others...);
}

/** \class MultiSync
 *  \brief RAII class to wrap a bunch of SyncInfo objects.
 *
 *  Provides basic synchronization for the common case in which an
 *  operation may act upon objects that exist on multiple distinct
 *  synchronous processing elements (e.g., cudaStreams) but actual
 *  computation can only occur on one of them.
 *
 *  Constructing an object of this class will cause the master
 *  processing element to wait on the others, asynchronously with
 *  respect to the CPU, if possible. Symmetrically, destruction of
 *  this object will cause the other processing elements to wait on
 *  the master processing element, asynchronously with respect to the
 *  CPU, if possible.
 *
 *  The master processing element is assumed to be the first SyncInfo
 *  passed into the constructor.
 */
template <Device... Ds>
class MultiSync
{
public:
    MultiSync(SyncInfo<Ds> const&... syncInfos)
        : syncInfos_{syncInfos...}
    {
        SyncMasterToAll_(MakeIndexSequence<sizeof...(Ds)>());
    }

    ~MultiSync()
    {
        SyncAllToMaster_(MakeIndexSequence<sizeof...(Ds)>());
    }
private:

    template <size_t... Is>
    void SyncMasterToAll_(IndexSequence<Is...>)
    {
        MasterWaitOnAll(std::get<Is>(syncInfos_)...);
    }

    template <size_t... Is>
    void SyncAllToMaster_(IndexSequence<Is...>)
    {
        AllWaitOnMaster(std::get<Is>(syncInfos_)...);
    }

    std::tuple<SyncInfo<Ds>...> syncInfos_;
};// class MultiSync

template <Device... Ds>
auto MakeMultiSync(SyncInfo<Ds> const&... syncInfos) -> MultiSync<Ds...>
{
    return MultiSync<Ds...>(syncInfos...);
}

}// namespace El
#endif // EL_CORE_SYNCINFO_HPP_

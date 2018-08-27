// Broadcast

#include <El/core/imports/aluminum.hpp>

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Broadcast(T* buffer, int count, int root, Comm comm, SyncInfo<D> const&)
{
    EL_DEBUG_CSE

    // FIXME What kind of synchronization needs to happen here??
    Al::Bcast<BestBackend<T,D>>(buffer, count, root, *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto multisync = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Bcast<BestBackend<T,Device::GPU>>(
        buffer, count, root, *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

    Synchronize(syncInfo);// NOOP on CPU,
                          // cudaStreamSynchronize on GPU.
    CheckMpi(MPI_Bcast(buffer, count, TypeMap<T>(), root, comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void Broadcast(Complex<T>* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    CheckMpi(MPI_Bcast(buffer, 2*count, TypeMap<T>(), root, comm.comm));
#else
    CheckMpi(MPI_Bcast(buffer, count, TypeMap<Complex<T>>(), root, comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;
    std::vector<byte> packedBuf;
    Serialize(count, buffer, packedBuf);
    CheckMpi(MPI_Bcast(packedBuf.data(), count, TypeMap<T>(), root, comm.comm));
    Deserialize(count, packedBuf, buffer);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void Broadcast(T*, int, int, Comm, SyncInfo<D> const&)
{
    LogicError("Broadcast: Bad device/type combination.");
}

template <typename T, Device D>
void Broadcast( T& b, int root, Comm comm, SyncInfo<D> const& syncInfo )
{ Broadcast( &b, 1, root, std::move(comm), syncInfo ); }

#define MPI_BROADCAST_PROTO_DEV(T,D)                                    \
    template void Broadcast(T*, int, int, Comm, SyncInfo<D> const&);       \
    template void Broadcast(T&, int, Comm, SyncInfo<D> const&);

#define MPI_BROADCAST_PROTO(T)                  \
    MPI_BROADCAST_PROTO_DEV(T,Device::CPU)      \
    MPI_BROADCAST_PROTO_DEV(T,Device::GPU)

MPI_BROADCAST_PROTO(byte)
MPI_BROADCAST_PROTO(int)
MPI_BROADCAST_PROTO(unsigned)
MPI_BROADCAST_PROTO(long int)
MPI_BROADCAST_PROTO(unsigned long)
MPI_BROADCAST_PROTO(float)
MPI_BROADCAST_PROTO(double)
#ifdef EL_HAVE_MPI_LONG_LONG
MPI_BROADCAST_PROTO(long long int)
MPI_BROADCAST_PROTO(unsigned long long)
#endif
MPI_BROADCAST_PROTO(ValueInt<Int>)
MPI_BROADCAST_PROTO(Entry<Int>)
MPI_BROADCAST_PROTO(Complex<float>)
MPI_BROADCAST_PROTO(ValueInt<float>)
MPI_BROADCAST_PROTO(ValueInt<Complex<float>>)
MPI_BROADCAST_PROTO(Entry<float>)
MPI_BROADCAST_PROTO(Entry<Complex<float>>)
MPI_BROADCAST_PROTO(Complex<double>)
MPI_BROADCAST_PROTO(ValueInt<double>)
MPI_BROADCAST_PROTO(ValueInt<Complex<double>>)
MPI_BROADCAST_PROTO(Entry<double>)
MPI_BROADCAST_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_BROADCAST_PROTO(DoubleDouble)
MPI_BROADCAST_PROTO(QuadDouble)
MPI_BROADCAST_PROTO(Complex<DoubleDouble>)
MPI_BROADCAST_PROTO(Complex<QuadDouble>)
MPI_BROADCAST_PROTO(ValueInt<DoubleDouble>)
MPI_BROADCAST_PROTO(ValueInt<QuadDouble>)
MPI_BROADCAST_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_BROADCAST_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_BROADCAST_PROTO(Entry<DoubleDouble>)
MPI_BROADCAST_PROTO(Entry<QuadDouble>)
MPI_BROADCAST_PROTO(Entry<Complex<DoubleDouble>>)
MPI_BROADCAST_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_BROADCAST_PROTO(Quad)
MPI_BROADCAST_PROTO(Complex<Quad>)
MPI_BROADCAST_PROTO(ValueInt<Quad>)
MPI_BROADCAST_PROTO(ValueInt<Complex<Quad>>)
MPI_BROADCAST_PROTO(Entry<Quad>)
MPI_BROADCAST_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_BROADCAST_PROTO(BigInt)
MPI_BROADCAST_PROTO(BigFloat)
MPI_BROADCAST_PROTO(Complex<BigFloat>)
MPI_BROADCAST_PROTO(ValueInt<BigInt>)
MPI_BROADCAST_PROTO(ValueInt<BigFloat>)
MPI_BROADCAST_PROTO(ValueInt<Complex<BigFloat>>)
MPI_BROADCAST_PROTO(Entry<BigInt>)
MPI_BROADCAST_PROTO(Entry<BigFloat>)
MPI_BROADCAST_PROTO(Entry<Complex<BigFloat>>)
#endif

}// namespace mpi
}// namespace El

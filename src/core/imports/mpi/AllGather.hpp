// AllGather

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& /*syncInfo*/)
{
    EL_DEBUG_CSE

    // FIXME: Synchronization here??
    Al::Allgather<BestBackend<T,D>>(sbuf, rbuf, sc, *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto multisync = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Allgather<BestBackend<T,Device::GPU>>(
        sbuf, rbuf, sc, *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    Synchronize(syncInfo);

#ifdef EL_USE_BYTE_ALLGATHERS
    LogicError("AllGather: Let Tom know if you go down this code path.");
    CheckMpi(
        MPI_Allgather(
            reinterpret_cast<UCP>(const_cast<T*>(sbuf)),
            sizeof(T)*sc, MPI_UNSIGNED_CHAR,
            reinterpret_cast<UCP>(rbuf),
            sizeof(T)*rc, MPI_UNSIGNED_CHAR,
            comm.comm));
#else
    CheckMpi(
        MPI_Allgather(
            sbuf, sc, TypeMap<T>(), rbuf, rc, TypeMap<T>(), comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllGather(
    const Complex<T>* sbuf, int sc,
    Complex<T>* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    Synchronize(syncInfo);

#ifdef EL_USE_BYTE_ALLGATHERS
    LogicError("AllGather: Let Tom know if you go down this code path.");
    CheckMpi(
        MPI_Allgather(
            reinterpret_cast<UCP>(const_cast<Complex<T>*>(sbuf)),
            2*sizeof(T)*sc, MPI_UNSIGNED_CHAR,
            reinterpret_cast<UCP>(rbuf),
            2*sizeof(T)*rc, MPI_UNSIGNED_CHAR,
            comm.comm));
#else
#ifdef EL_AVOID_COMPLEX_MPI
    CheckMpi(
        MPI_Allgather(
            sbuf, 2*sc, TypeMap<T>(), rbuf, 2*rc, TypeMap<T>(), comm.comm));
#else
    CheckMpi(
        MPI_Allgather(
            sbuf, sc, TypeMap<Complex<T>>(),
            rbuf, rc, TypeMap<Complex<T>>(), comm.comm));
#endif
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void AllGather(
    T const* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    const int commSize = mpi::Size(comm);
    const int totalRecv = rc*commSize;

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize(sc, sbuf, packedSend);

    ReserveSerialized(totalRecv, rbuf, packedRecv);
    CheckMpi(
        MPI_Allgather(
            packedSend.data(), sc, TypeMap<T>(),
            packedRecv.data(), rc, TypeMap<T>(), comm.comm));
    Deserialize(totalRecv, packedRecv, rbuf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void AllGather(
    T const*, int, T*, int, Comm, SyncInfo<D> const&)
{
    LogicError("AllGather: Bad device/type combination.");
}

#define MPI_ALLGATHER_PROTO_DEV(T,D) \
    template void AllGather(const T* sbuf, int sc, T* rbuf, int rc, Comm comm, \
                            SyncInfo<D> const&);

#define MPI_ALLGATHER_PROTO(T) \
    MPI_ALLGATHER_PROTO_DEV(T,Device::CPU) \
    MPI_ALLGATHER_PROTO_DEV(T,Device::GPU)

MPI_ALLGATHER_PROTO(byte)
MPI_ALLGATHER_PROTO(int)
MPI_ALLGATHER_PROTO(unsigned)
MPI_ALLGATHER_PROTO(long int)
MPI_ALLGATHER_PROTO(unsigned long)
MPI_ALLGATHER_PROTO(float)
MPI_ALLGATHER_PROTO(double)
#ifdef EL_HAVE_MPI_LONG_LONG
MPI_ALLGATHER_PROTO(long long int)
MPI_ALLGATHER_PROTO(unsigned long long)
#endif
MPI_ALLGATHER_PROTO(ValueInt<Int>)
MPI_ALLGATHER_PROTO(Entry<Int>)
MPI_ALLGATHER_PROTO(Complex<float>)
MPI_ALLGATHER_PROTO(ValueInt<float>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<float>>)
MPI_ALLGATHER_PROTO(Entry<float>)
MPI_ALLGATHER_PROTO(Entry<Complex<float>>)
MPI_ALLGATHER_PROTO(Complex<double>)
MPI_ALLGATHER_PROTO(ValueInt<double>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<double>>)
MPI_ALLGATHER_PROTO(Entry<double>)
MPI_ALLGATHER_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_ALLGATHER_PROTO(DoubleDouble)
MPI_ALLGATHER_PROTO(QuadDouble)
MPI_ALLGATHER_PROTO(Complex<DoubleDouble>)
MPI_ALLGATHER_PROTO(Complex<QuadDouble>)
MPI_ALLGATHER_PROTO(ValueInt<DoubleDouble>)
MPI_ALLGATHER_PROTO(ValueInt<QuadDouble>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_ALLGATHER_PROTO(Entry<DoubleDouble>)
MPI_ALLGATHER_PROTO(Entry<QuadDouble>)
MPI_ALLGATHER_PROTO(Entry<Complex<DoubleDouble>>)
MPI_ALLGATHER_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_ALLGATHER_PROTO(Quad)
MPI_ALLGATHER_PROTO(Complex<Quad>)
MPI_ALLGATHER_PROTO(ValueInt<Quad>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<Quad>>)
MPI_ALLGATHER_PROTO(Entry<Quad>)
MPI_ALLGATHER_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_ALLGATHER_PROTO(BigInt)
MPI_ALLGATHER_PROTO(BigFloat)
MPI_ALLGATHER_PROTO(Complex<BigFloat>)
MPI_ALLGATHER_PROTO(ValueInt<BigInt>)
MPI_ALLGATHER_PROTO(ValueInt<BigFloat>)
MPI_ALLGATHER_PROTO(ValueInt<Complex<BigFloat>>)
MPI_ALLGATHER_PROTO(Entry<BigInt>)
MPI_ALLGATHER_PROTO(Entry<BigFloat>)
MPI_ALLGATHER_PROTO(Entry<Complex<BigFloat>>)
#endif

} // namespace mpi
} // namespace El

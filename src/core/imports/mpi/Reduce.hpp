// Reduce

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Reduce(T const* sbuf, T* rbuf, int count, Op op,
            int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    Al::Reduce<BestBackend<T,D>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        root, *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Reduce(T const* sbuf, T* rbuf, int count, Op op,
            int root, Comm comm, SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Reduce<BestBackend<T,Device::GPU>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        root, *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void Reduce(T const* sbuf, T* rbuf, int count, Op op,
            int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);
    CheckMpi(
        MPI_Reduce(
            sbuf, rbuf, count, TypeMap<T>(), opC, root, comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void Reduce(const Complex<T>* sbuf, Complex<T>* rbuf, int count, Op op,
            int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        MPI_Op opC = NativeOp<T>(op);
        CheckMpi(
            MPI_Reduce(
                sbuf, rbuf, 2*count, TypeMap<T>(), opC, root, comm.comm));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<T>>(op);
        CheckMpi(
            MPI_Reduce(
                sbuf, rbuf, count, TypeMap<Complex<T>>(), opC, root, comm.comm));
    }
#else
    MPI_Op opC = NativeOp<Complex<T>>(op);
    CheckMpi(
        MPI_Reduce(
            sbuf, rbuf, count, TypeMap<Complex<T>>(), opC, root, comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void Reduce(T const* sbuf, T* rbuf, int count, Op op,
            int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);

    const int commRank = mpi::Rank(comm);
    std::vector<byte> packedSend, packedRecv;
    Serialize(count, sbuf, packedSend);

    if (commRank == root)
        ReserveSerialized(count, rbuf, packedRecv);
    CheckMpi(
        MPI_Reduce(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            opC, root, comm.comm));
    if (commRank == root)
        Deserialize(count, packedRecv, rbuf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void Reduce(T const*, T*, int, Op, int, Comm, SyncInfo<D> const&)
{
    LogicError("Reduce: Bad device/type combination.");
}

template <typename T, Device D>
void Reduce(T const* sbuf, T* rbuf, int count, int root,
            Comm comm, SyncInfo<D> const& syncInfo)
{
    Reduce(sbuf, rbuf, count, SUM, root, std::move(comm), syncInfo);
}

template <typename T, Device D>
T Reduce(T sb, Op op, int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    T rb;
    Reduce(&sb, &rb, 1, op, root, std::move(comm), syncInfo);
    return rb;
}

template <typename T, Device D>
T Reduce(T sb, int root, Comm comm, SyncInfo<D> const& syncInfo)
{
    T rb;
    Reduce(&sb, &rb, 1, SUM, root, std::move(comm), syncInfo);
    return rb;
}

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void Reduce(T* buf, int count, Op op,
            int root, Comm comm, SyncInfo<D> const& /*syncInfo*/)
{
    EL_DEBUG_CSE

    Al::Reduce<BestBackend<T,D>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        root, *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void Reduce(T* buf, int count, Op op,
            int root, Comm comm, SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Reduce<BestBackend<T,Device::GPU>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        root, *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                    Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=EnableIf<IsPacked<T>>>*/>
void Reduce(T* buf, int count, Op op, int root, Comm comm,
            SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);

    const int commRank = Rank(comm);
    if (commRank == root)
    {
        CheckMpi(
            MPI_Reduce(
                MPI_IN_PLACE, buf, count, TypeMap<T>(), opC, root, comm.comm));
    }
    else
        CheckMpi(
            MPI_Reduce(
                buf, 0, count, TypeMap<T>(), opC, root, comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                    Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=EnableIf<IsPacked<T>>>*/>
void Reduce(Complex<T>* buf, int count, Op op, int root, Comm comm,
            SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (Size(comm) == 1 || count == 0)
        return;

    Synchronize(syncInfo);

    const int commRank = Rank(comm);
#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        MPI_Op opC = NativeOp<T>(op);
        if (commRank == root)
        {
            CheckMpi(
                MPI_Reduce(
                    MPI_IN_PLACE, buf, 2*count, TypeMap<T>(), opC,
                    root, comm.comm));
        }
        else
            CheckMpi(
                MPI_Reduce(
                    buf, 0, 2*count, TypeMap<T>(), opC, root, comm.comm));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<T>>(op);
        if (commRank == root)
        {
            CheckMpi(
                MPI_Reduce(
                    MPI_IN_PLACE, buf, count, TypeMap<Complex<T>>(), opC,
                    root, comm.comm));
        }
        else
            CheckMpi(
                MPI_Reduce(
                    buf, 0, count, TypeMap<Complex<T>>(), opC,
                    root, comm.comm));
    }
#else
    MPI_Op opC = NativeOp<Complex<T>>(op);
    if (commRank == root)
    {
        CheckMpi(
            MPI_Reduce(
                MPI_IN_PLACE, buf, count, TypeMap<Complex<T>>(), opC,
                root, comm.comm));
    }
    else
        CheckMpi(
            MPI_Reduce(buf, 0, count, TypeMap<Complex<T>>(), opC,
                       root, comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void Reduce(T* buf, int count, Op op, int root, Comm comm,
            SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);

    // TODO(poulson): Use in-place option?

    const int commRank = mpi::Rank(comm);
    std::vector<byte> packedSend, packedRecv;
    Serialize(count, buf, packedSend);

    if (commRank == root)
        ReserveSerialized(count, buf, packedRecv);
    CheckMpi(
        MPI_Reduce(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(), opC,
            root, comm.comm));
    if (commRank == root)
        Deserialize(count, packedRecv, buf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void Reduce(T*, int, Op, int, Comm, SyncInfo<D> const&)
{
    LogicError("Reduce: Bad device/type combination.");
}

template <typename T, Device D>
void Reduce(T* buf, int count, int root, Comm comm,
            SyncInfo<D> const& syncInfo)
{
    Reduce(buf, count, SUM, root, std::move(comm), syncInfo);
}



#define MPI_REDUCE_PROTO_DEV(T,D)                                       \
    template void Reduce(                                               \
        T const*, T*, int, Op, int, Comm, SyncInfo<D> const&);          \
    template void Reduce(                                               \
        T const*, T*, int, int, Comm, SyncInfo<D> const&);              \
    template T Reduce(T, Op, int, Comm, SyncInfo<D> const&);            \
    template T Reduce(T, int, Comm, SyncInfo<D> const&);                \
    template void Reduce(T*, int, Op, int, Comm, SyncInfo<D> const&);   \
    template void Reduce(T*, int, int, Comm, SyncInfo<D> const&);

#ifndef HYDROGEN_HAVE_CUDA
#define MPI_REDUCE_PROTO(T)                     \
    MPI_REDUCE_PROTO_DEV(T,Device::CPU)
#else
#define MPI_REDUCE_PROTO(T)                     \
    MPI_REDUCE_PROTO_DEV(T,Device::CPU)         \
    MPI_REDUCE_PROTO_DEV(T,Device::GPU)
#endif // HYDROGEN_HAVE_CUDA

MPI_REDUCE_PROTO(byte)
MPI_REDUCE_PROTO(int)
MPI_REDUCE_PROTO(unsigned)
MPI_REDUCE_PROTO(long int)
MPI_REDUCE_PROTO(unsigned long)
MPI_REDUCE_PROTO(float)
MPI_REDUCE_PROTO(double)
#ifdef EL_HAVE_MPI_LONG_LONG
MPI_REDUCE_PROTO(long long int)
MPI_REDUCE_PROTO(unsigned long long)
#endif
MPI_REDUCE_PROTO(ValueInt<Int>)
MPI_REDUCE_PROTO(Entry<Int>)
MPI_REDUCE_PROTO(Complex<float>)
MPI_REDUCE_PROTO(ValueInt<float>)
MPI_REDUCE_PROTO(ValueInt<Complex<float>>)
MPI_REDUCE_PROTO(Entry<float>)
MPI_REDUCE_PROTO(Entry<Complex<float>>)
MPI_REDUCE_PROTO(Complex<double>)
MPI_REDUCE_PROTO(ValueInt<double>)
MPI_REDUCE_PROTO(ValueInt<Complex<double>>)
MPI_REDUCE_PROTO(Entry<double>)
MPI_REDUCE_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_REDUCE_PROTO(DoubleDouble)
MPI_REDUCE_PROTO(QuadDouble)
MPI_REDUCE_PROTO(Complex<DoubleDouble>)
MPI_REDUCE_PROTO(Complex<QuadDouble>)
MPI_REDUCE_PROTO(ValueInt<DoubleDouble>)
MPI_REDUCE_PROTO(ValueInt<QuadDouble>)
MPI_REDUCE_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_REDUCE_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_REDUCE_PROTO(Entry<DoubleDouble>)
MPI_REDUCE_PROTO(Entry<QuadDouble>)
MPI_REDUCE_PROTO(Entry<Complex<DoubleDouble>>)
MPI_REDUCE_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_REDUCE_PROTO(Quad)
MPI_REDUCE_PROTO(Complex<Quad>)
MPI_REDUCE_PROTO(ValueInt<Quad>)
MPI_REDUCE_PROTO(ValueInt<Complex<Quad>>)
MPI_REDUCE_PROTO(Entry<Quad>)
MPI_REDUCE_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_REDUCE_PROTO(BigInt)
MPI_REDUCE_PROTO(BigFloat)
MPI_REDUCE_PROTO(Complex<BigFloat>)
MPI_REDUCE_PROTO(ValueInt<BigInt>)
MPI_REDUCE_PROTO(ValueInt<BigFloat>)
MPI_REDUCE_PROTO(ValueInt<Complex<BigFloat>>)
MPI_REDUCE_PROTO(Entry<BigInt>)
MPI_REDUCE_PROTO(Entry<BigFloat>)
MPI_REDUCE_PROTO(Entry<Complex<BigFloat>>)
#endif

}// namespace mpi
}// namespace El

namespace El
{
namespace mpi
{

// IsValidAluminumDeviceType should mean both that the device/type
// combo is valid and that the backend supports this collective.
//
#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    Al::Reduce_scatter<BestBackend<T,D>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,D,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
                   SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Reduce_scatter<BestBackend<T,Device::GPU>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                  Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
                    SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

#ifdef EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");

    const int commSize = Size(comm);
    const int commRank = Rank(comm);
    AllReduce(sbuf, count*commSize, op, comm, syncInfo);
    MemCopy(rbuf, &sbuf[commRank*count], count);
#elif defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, count, TypeMap<T>(), NativeOp<T>(op), comm.comm));
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");

    const int commSize = Size(comm);
    Reduce(sbuf, count*commSize, op, 0, comm);
    Scatter(sbuf, count, rbuf, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void ReduceScatter(Complex<T> const* sbuf, Complex<T>* rbuf,
                   int count, Op op, Comm comm, SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

#ifdef EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    const int commSize = Size(comm);
    const int commRank = Rank(comm);
    AllReduce(sbuf, count*commSize, op, comm);
    MemCopy(rbuf, &sbuf[commRank*count], count);
#elif defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
# ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, 2*count, TypeMap<T>(), NativeOp<T>(op), comm.comm));
# else
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            sbuf, rbuf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op),
            comm.comm));
# endif
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");

    const int commSize = Size(comm);
    Reduce(sbuf, count*commSize, op, 0, comm);
    Scatter(sbuf, count, rbuf, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void ReduceScatter(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    const int commSize = mpi::Size(comm);
    const int totalSend = count*commSize;
    const int totalRecv = count;

    // TODO(poulson): Add AllReduce approach via
    // EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
#if defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
    std::vector<byte> packedSend, packedRecv;
    Serialize(totalSend, sbuf, packedSend);

    ReserveSerialized(totalRecv, rbuf, packedRecv);
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            NativeOp<T>(op), comm.comm));

    Deserialize(totalRecv, packedRecv, rbuf);
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");

    Reduce(sbuf, totalSend, op, 0, comm);
    Scatter(sbuf, count, rbuf, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void ReduceScatter(T const*, T*, int, Op, Comm, SyncInfo<D> const&)
{
    LogicError("ReduceScatter: Bad device/type combination.");
}

//
// The "IN_PLACE" Reduce_scatter
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& /*syncInfo*/)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    // FIXME Synchronize
    Al::Reduce_scatter<BestBackend<T,D>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}

template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,
                                  Collectives::REDUCESCATTER>>*/>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Reduce_scatter<BestBackend<T,Device::GPU>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    Synchronize(syncInfo);

#ifdef EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    const int commSize = Size(comm);
    const int commRank = Rank(comm);
    AllReduce(buf, count*commSize, op, comm);
    if (commRank != 0)
        MemCopy(buf, &buf[commRank*count], count);
#elif defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, count,
            TypeMap<T>(), NativeOp<T>(op), comm.comm));
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    const int commSize = Size(comm);
    Reduce(buf, count*commSize, op, 0, comm);
    Scatter(buf, count, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void ReduceScatter(Complex<T>* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    Synchronize(syncInfo);

#ifdef EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    const int commSize = Size(comm);
    const int commRank = Rank(comm);
    AllReduce(buf, count*commSize, op, comm);
    if (commRank != 0)
        MemCopy(buf, &buf[commRank*count], count);
#elif defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
# ifdef EL_AVOID_COMPLEX_MPI
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, 2*count,
            TypeMap<T>(), NativeOp<T>(op), comm.comm));
# else
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            MPI_IN_PLACE, buf, count,
            TypeMap<Complex<T>>(),
            NativeOp<Complex<T>>(op), comm.comm));
# endif
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    const int commSize = Size(comm);
    Reduce(buf, count*commSize, op, 0, comm);
    Scatter(buf, count, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;
    const int commSize = mpi::Size(comm);
    const int totalSend = count*commSize;
    const int totalRecv = count;

    Synchronize(syncInfo);

    // TODO(poulson): Add AllReduce approach via
    // EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
#if defined(EL_HAVE_MPI_REDUCE_SCATTER_BLOCK)
    std::vector<byte> packedSend, packedRecv;
    Serialize(totalSend, buf, packedSend);

    ReserveSerialized(totalRecv, buf, packedRecv);
    EL_CHECK_MPI(
        MPI_Reduce_scatter_block(
            packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
            NativeOp<T>(op), comm.comm));

    Deserialize(totalRecv, packedRecv, buf);
#else
    LogicError("ReduceScatter: Let Tom know if you go down this code path.");
    Reduce(buf, totalSend, op, 0, comm);
    Scatter(buf, count, count, 0, comm);
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D>>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void ReduceScatter(T*, int, Op, Comm, SyncInfo<D> const&)
{
    LogicError("ReduceScatter: Bad device/type combination.");
}

//
// The other stuff
//

template <typename T, Device D>
void ReduceScatter(T const* sbuf, T* rbuf, int rc, Comm comm,
                   SyncInfo<D> const& syncInfo)
{
    ReduceScatter(sbuf, rbuf, rc, SUM, comm, syncInfo);
}

template <typename T, Device D>
T ReduceScatter(T sb, Op op, Comm comm, SyncInfo<D> const& syncInfo)
{
    T rb; ReduceScatter(&sb, &rb, 1, op, comm, syncInfo); return rb;
}

template <typename T, Device D>
T ReduceScatter(T sb, Comm comm, SyncInfo<D> const& syncInfo)
{
    return ReduceScatter(sb, SUM, comm, syncInfo);
}

template <typename T, Device D>
void ReduceScatter(T* buf, int rc, Comm comm, SyncInfo<D> const& syncInfo)
{
    ReduceScatter(buf, rc, SUM, comm, syncInfo);
}

#define MPI_REDUCESCATTER_PROTO_DEV(T,D)                                \
    template void ReduceScatter(T const*, T*, int rc, Op op, Comm,      \
                                SyncInfo<D> const&);                    \
    template void ReduceScatter(T const*, T*, int rc, Comm,             \
                                SyncInfo<D> const&);                    \
    template T ReduceScatter(T, Op, Comm, SyncInfo<D> const&);          \
    template T ReduceScatter(T, Comm, SyncInfo<D> const&);              \
    template void ReduceScatter(T*, int, Op, Comm, SyncInfo<D> const&); \
    template void ReduceScatter(T*, int, Comm, SyncInfo<D> const&);

#define MPI_REDUCESCATTER_PROTO(T)             \
    MPI_REDUCESCATTER_PROTO_DEV(T,Device::CPU) \
    MPI_REDUCESCATTER_PROTO_DEV(T,Device::GPU)

MPI_REDUCESCATTER_PROTO(byte)
MPI_REDUCESCATTER_PROTO(int)
MPI_REDUCESCATTER_PROTO(unsigned)
MPI_REDUCESCATTER_PROTO(long int)
MPI_REDUCESCATTER_PROTO(unsigned long)
MPI_REDUCESCATTER_PROTO(float)
MPI_REDUCESCATTER_PROTO(double)
#ifdef EL_HAVE_MPI_LONG_LONG
MPI_REDUCESCATTER_PROTO(long long int)
MPI_REDUCESCATTER_PROTO(unsigned long long)
#endif
MPI_REDUCESCATTER_PROTO(ValueInt<Int>)
MPI_REDUCESCATTER_PROTO(Entry<Int>)
MPI_REDUCESCATTER_PROTO(Complex<float>)
MPI_REDUCESCATTER_PROTO(ValueInt<float>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<float>>)
MPI_REDUCESCATTER_PROTO(Entry<float>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<float>>)
MPI_REDUCESCATTER_PROTO(Complex<double>)
MPI_REDUCESCATTER_PROTO(ValueInt<double>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<double>>)
MPI_REDUCESCATTER_PROTO(Entry<double>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_REDUCESCATTER_PROTO(DoubleDouble)
MPI_REDUCESCATTER_PROTO(QuadDouble)
MPI_REDUCESCATTER_PROTO(Complex<DoubleDouble>)
MPI_REDUCESCATTER_PROTO(Complex<QuadDouble>)
MPI_REDUCESCATTER_PROTO(ValueInt<DoubleDouble>)
MPI_REDUCESCATTER_PROTO(ValueInt<QuadDouble>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_REDUCESCATTER_PROTO(Entry<DoubleDouble>)
MPI_REDUCESCATTER_PROTO(Entry<QuadDouble>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<DoubleDouble>>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_REDUCESCATTER_PROTO(Quad)
MPI_REDUCESCATTER_PROTO(Complex<Quad>)
MPI_REDUCESCATTER_PROTO(ValueInt<Quad>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<Quad>>)
MPI_REDUCESCATTER_PROTO(Entry<Quad>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_REDUCESCATTER_PROTO(BigInt)
MPI_REDUCESCATTER_PROTO(BigFloat)
MPI_REDUCESCATTER_PROTO(Complex<BigFloat>)
MPI_REDUCESCATTER_PROTO(ValueInt<BigInt>)
MPI_REDUCESCATTER_PROTO(ValueInt<BigFloat>)
MPI_REDUCESCATTER_PROTO(ValueInt<Complex<BigFloat>>)
MPI_REDUCESCATTER_PROTO(Entry<BigInt>)
MPI_REDUCESCATTER_PROTO(Entry<BigFloat>)
MPI_REDUCESCATTER_PROTO(Entry<Complex<BigFloat>>)
#endif

}// namespace mpi
}// namespace El

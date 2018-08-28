namespace El
{
namespace mpi
{

//
// The "normal" allreduce (not "IN_PLACE").
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumDeviceType<T,D>>*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const&)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    // FIXME Synchronize
    Al::Allreduce<BestBackend<T,D>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumDeviceType<T,D>>*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    AddSynchronizationPoint(syncInfo, alSyncInfo);

    Al::Allreduce<BestBackend<T,Device::GPU>>(
        sbuf, rbuf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);

    AddSynchronizationPoint(alSyncInfo, syncInfo);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                  Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    CheckMpi(
        MPI_Allreduce(
            const_cast<T*>(sbuf), rbuf,
            count, TypeMap<T>(), NativeOp<T>(op), comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllReduce(Complex<T> const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        CheckMpi(
            MPI_Allreduce(
                const_cast<Complex<T>*>(sbuf), rbuf, 2*count,
                TypeMap<T>(), NativeOp<T>(op), comm.comm));
    }
    else
    {
        CheckMpi(
            MPI_Allreduce(
                const_cast<Complex<T>*>(sbuf), rbuf, count,
                TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op), comm.comm));
    }
#else
    CheckMpi(
        MPI_Allreduce(
            const_cast<Complex<T>*>(sbuf), rbuf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op), comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;

    Serialize(count, sbuf, packedSend);

    ReserveSerialized(count, rbuf, packedRecv);
    CheckMpi(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.comm));
    Deserialize(count, packedRecv, rbuf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void AllReduce(T const*, T*, int, Op, Comm, SyncInfo<D> const&)
{
    LogicError("AllReduce: Bad device/type combination.");
}

//
// The "IN_PLACE" allreduce
//

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumDeviceType<T,D>>*/>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& /*syncInfo*/)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    // FIXME Synchronize
    Al::Allreduce<BestBackend<T,D>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumDeviceType<T,D>>*/>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    SyncInfo<Device::GPU> alSyncInfo(comm.aluminum_comm->get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Allreduce<BestBackend<T,Device::GPU>>(
        buf, count, MPI_Op2ReductionOperator(NativeOp<T>(op)),
        *comm.aluminum_comm);
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    Synchronize(syncInfo);

    CheckMpi(
        MPI_Allreduce(
            MPI_IN_PLACE, buf,
            count, TypeMap<T>(), NativeOp<T>(op), comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllReduce(Complex<T>* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        CheckMpi(
            MPI_Allreduce(
                MPI_IN_PLACE, buf, 2*count,
                TypeMap<T>(), NativeOp<T>(op), comm.comm));
    }
    else
    {
        CheckMpi(
            MPI_Allreduce(
                MPI_IN_PLACE, buf, count, TypeMap<Complex<T>>(),
                NativeOp<Complex<T>>(op), comm.comm));
    }
#else
    CheckMpi(
        MPI_Allreduce(
            MPI_IN_PLACE, buf, count,
            TypeMap<Complex<T>>(), NativeOp<Complex<T>>(op),
            comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    Synchronize(syncInfo);

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;
    Serialize(count, buf, packedSend);

    ReserveSerialized(count, buf, packedRecv);
    CheckMpi(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.comm));
    Deserialize(count, packedRecv, buf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumDeviceType<T,D>>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void AllReduce(T*, int, Op, Comm, SyncInfo<D> const&)
{
    LogicError("AllReduce: Bad device/type combination.");
}

template<typename T, Device D>
void AllReduce(const T* sbuf, T* rbuf, int count, Comm comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ AllReduce(sbuf, rbuf, count, SUM, std::move(comm), syncInfo); }

template<typename T, Device D>
T AllReduce(T sb, Op op, Comm comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ T rb; AllReduce(&sb, &rb, 1, op, std::move(comm), syncInfo); return rb; }

template<typename T, Device D>
T AllReduce(T sb, Comm comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ return AllReduce(sb, SUM, std::move(comm), syncInfo); }

template<typename T, Device D>
void AllReduce(T* buf, int count, Comm comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{ AllReduce(buf, count, SUM, std::move(comm), syncInfo); }

#define MPI_ALLREDUCE_PROTO_DEV(T,D)                                    \
    template void AllReduce(                                            \
        const T*, T*, int, Op, Comm, SyncInfo<D> const&);               \
    template void AllReduce(                                            \
        const T*, T*, int, Comm, SyncInfo<D> const&);                   \
    template T AllReduce(T, Op, Comm, SyncInfo<D> const&);              \
    template T AllReduce(T, Comm, SyncInfo<D> const&);                  \
    template void AllReduce(T*, int, Op, Comm, SyncInfo<D> const&);     \
    template void AllReduce(T*, int, Comm, SyncInfo<D> const&);

#define MPI_ALLREDUCE_PROTO(T)             \
    MPI_ALLREDUCE_PROTO_DEV(T,Device::CPU) \
    MPI_ALLREDUCE_PROTO_DEV(T,Device::GPU)

MPI_ALLREDUCE_PROTO(byte)
MPI_ALLREDUCE_PROTO(int)
MPI_ALLREDUCE_PROTO(unsigned)
MPI_ALLREDUCE_PROTO(long int)
MPI_ALLREDUCE_PROTO(unsigned long)
MPI_ALLREDUCE_PROTO(float)
MPI_ALLREDUCE_PROTO(double)
#ifdef EL_HAVE_MPI_LONG_LONG
MPI_ALLREDUCE_PROTO(long long int)
MPI_ALLREDUCE_PROTO(unsigned long long)
#endif
MPI_ALLREDUCE_PROTO(ValueInt<Int>)
MPI_ALLREDUCE_PROTO(Entry<Int>)
MPI_ALLREDUCE_PROTO(Complex<float>)
MPI_ALLREDUCE_PROTO(ValueInt<float>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<float>>)
MPI_ALLREDUCE_PROTO(Entry<float>)
MPI_ALLREDUCE_PROTO(Entry<Complex<float>>)
MPI_ALLREDUCE_PROTO(Complex<double>)
MPI_ALLREDUCE_PROTO(ValueInt<double>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<double>>)
MPI_ALLREDUCE_PROTO(Entry<double>)
MPI_ALLREDUCE_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_ALLREDUCE_PROTO(DoubleDouble)
MPI_ALLREDUCE_PROTO(QuadDouble)
MPI_ALLREDUCE_PROTO(Complex<DoubleDouble>)
MPI_ALLREDUCE_PROTO(Complex<QuadDouble>)
MPI_ALLREDUCE_PROTO(ValueInt<DoubleDouble>)
MPI_ALLREDUCE_PROTO(ValueInt<QuadDouble>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_ALLREDUCE_PROTO(Entry<DoubleDouble>)
MPI_ALLREDUCE_PROTO(Entry<QuadDouble>)
MPI_ALLREDUCE_PROTO(Entry<Complex<DoubleDouble>>)
MPI_ALLREDUCE_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_ALLREDUCE_PROTO(Quad)
MPI_ALLREDUCE_PROTO(Complex<Quad>)
MPI_ALLREDUCE_PROTO(ValueInt<Quad>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<Quad>>)
MPI_ALLREDUCE_PROTO(Entry<Quad>)
MPI_ALLREDUCE_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_ALLREDUCE_PROTO(BigInt)
MPI_ALLREDUCE_PROTO(BigFloat)
MPI_ALLREDUCE_PROTO(Complex<BigFloat>)
MPI_ALLREDUCE_PROTO(ValueInt<BigInt>)
MPI_ALLREDUCE_PROTO(ValueInt<BigFloat>)
MPI_ALLREDUCE_PROTO(ValueInt<Complex<BigFloat>>)
MPI_ALLREDUCE_PROTO(Entry<BigInt>)
MPI_ALLREDUCE_PROTO(Entry<BigFloat>)
MPI_ALLREDUCE_PROTO(Entry<Complex<BigFloat>>)
#endif

}// namespace mpi
}// namespace El

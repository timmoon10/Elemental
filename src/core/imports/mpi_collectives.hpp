/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   Copyright (c) 2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include "mpi_utils.hpp"
#ifdef HYDROGEN_HAVE_ALUMINUM
#include <El/core/imports/aluminum.hpp>
#endif //HYDROGEN_HAVE_ALUMINUM

namespace El
{

Al::ReductionOperator MPI_Op2ReductionOperator(MPI_Op op)
{
    switch (op)
    {
    case MPI_SUM:
        return Al::ReductionOperator::sum;
    case MPI_PROD:
        return Al::ReductionOperator::prod;
    case MPI_MIN:
        return Al::ReductionOperator::min;
    case MPI_MAX:
        return Al::ReductionOperator::max;
    default:
        LogicError("Given reduction operator not supported.");
    }
    // Silence compiler warning
    return Al::ReductionOperator::sum;
}

namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM

// Attempted Aluminum dispatch and it was successful ; call Aluminum
template <typename BackendT, typename Real,
          typename=EnableIf<IsAlTypeT<Real,BackendT>>>
void aluminum_allreduce(
    Real const* sbuf, Real* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    auto red_op = MPI_Op2ReductionOperator(op.op);
    Al::Allreduce<BackendT>(sbuf, rbuf, count, red_op, *comm.aluminum_comm);
}

// Attempted Aluminum dispatch and it failed ; call Hydrogen's fallback
template <typename BackendT, typename Real,
          typename=DisableIf<IsAlTypeT<Real,BackendT>>,
          typename=void>
void aluminum_allreduce(
    Real const* sbuf, Real* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    fallback_allreduce(sbuf, rbuf, count, op, comm);
}

// IN_PLACE version

// Attempted Aluminum dispatch and it was successful ; call Aluminum
template <typename BackendT, typename Real,
          typename=EnableIf<IsAlTypeT<Real,BackendT>>>
void aluminum_allreduce(
    Real* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    auto red_op = MPI_Op2ReductionOperator(op.op);
    Al::Allreduce<BackendT>(rbuf, count, red_op, *comm.aluminum_comm);
}

// Attempted Aluminum dispatch and it failed ; call Hydrogen's fallback
template <typename BackendT, typename Real,
          typename=DisableIf<IsAlTypeT<Real,BackendT>>,
          typename=void>
void aluminum_allreduce(
    Real* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    fallback_allreduce(rbuf, count, op, std::move(comm));
}

#endif // HYDROGEN_HAVE_ALUMINUM


// Fallback impls in plain ol' MPI

template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void fallback_allreduce(
    Real const* sbuf, Real* rbuf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    MPI_Op opC = NativeOp<Real>(op);
    EL_CHECK_MPI(
        MPI_Allreduce(
            const_cast<Real*>(sbuf), rbuf,
            count, TypeMap<Real>(), opC, comm.comm));

}

template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void fallback_allreduce(
    Complex<Real> const* sbuf, Complex<Real>* rbuf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count != 0)
    {
#ifdef EL_AVOID_COMPLEX_MPI
        if (op == SUM)
        {
            MPI_Op opC = NativeOp<Real>(op);
            EL_CHECK_MPI
            (MPI_Allreduce
                (const_cast<Complex<Real>*>(sbuf),
                  rbuf, 2*count, TypeMap<Real>(), opC, comm.comm));
        }
        else
        {
            MPI_Op opC = NativeOp<Complex<Real>>(op);
            EL_CHECK_MPI
            (MPI_Allreduce
              (const_cast<Complex<Real>*>(sbuf),
                rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm));
        }
#else
        MPI_Op opC = NativeOp<Complex<Real>>(op);
        EL_CHECK_MPI
        (MPI_Allreduce
          (const_cast<Complex<Real>*>(sbuf),
            rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm));
#endif
    }
}

template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void fallback_allreduce
(const T* sbuf, T* rbuf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;

    Serialize(count, sbuf, packedSend);

    ReserveSerialized(count, rbuf, packedRecv);
    EL_CHECK_MPI(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.comm));
    Deserialize(count, packedRecv, rbuf);
}

template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void fallback_allreduce(Real* buf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

    MPI_Op opC = NativeOp<Real>(op);
    EL_CHECK_MPI(
        MPI_Allreduce(
            MPI_IN_PLACE, buf, count, TypeMap<Real>(), opC, comm.comm));
}

template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void fallback_allreduce(Complex<Real>* buf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef EL_AVOID_COMPLEX_MPI
    if (op == SUM)
    {
        MPI_Op opC = NativeOp<Real>(op);
        EL_CHECK_MPI
        (MPI_Allreduce
          (MPI_IN_PLACE, buf, 2*count, TypeMap<Real>(), opC, comm.comm));
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>(op);
        EL_CHECK_MPI
        (MPI_Allreduce
          (MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(),
            opC, comm.comm));
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>(op);
    EL_CHECK_MPI
    (MPI_Allreduce
      (MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(), opC,
        comm.comm));
#endif
}

template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void fallback_allreduce(T* buf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

    MPI_Op opC = NativeOp<T>(op);
    std::vector<byte> packedSend, packedRecv;
    Serialize(count, buf, packedSend);

    ReserveSerialized(count, buf, packedRecv);
    EL_CHECK_MPI(
        MPI_Allreduce(
            packedSend.data(), packedRecv.data(),
            count, TypeMap<T>(), opC, comm.comm));
    Deserialize(count, packedRecv, buf);
}

//
// THE REAL THING
//
template <typename T,
          typename/*=EnableIf<IsAluminumTypeT<T>>*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    if (count == 0)
        return;

#ifdef HYDROGEN_HAVE_CUDA
    auto sbuf_on_device = IsGPUMemory(sbuf), rbuf_on_device = IsGPUMemory(rbuf);

    // All memory is on the device
    if (sbuf_on_device && rbuf_on_device)
        aluminum_allreduce<GPUBackend>(sbuf, rbuf, count, op, std::move(comm));
    // All memory is on the host
    else if (!sbuf_on_device && !rbuf_on_device)
        aluminum_allreduce<CPUBackend>(sbuf, rbuf, count, op, std::move(comm));
    // Some memory is on the host; some is on the device
    else
        fallback_allreduce(sbuf, rbuf, count, op, std::move(comm));
#else
    aluminum_allreduce<CPUBackend>(sbuf, rbuf, count, op, std::move(comm));
#endif // HYDROGEN_HAVE_CUDA

}

template <typename T,
          typename/*=DisableIf<IsAluminumTypeT<T>>*/,
          typename/*=void*/>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    if (count != 0)
        fallback_allreduce(sbuf, rbuf, count, op, std::move(comm));
}

// The IN_PLACE versions
template<typename T,
         typename/*=EnableIf<IsAluminumTypeT<T>>*/>
void AllReduce(T* rbuf, int count, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (count == 0 || Size(comm) == 1)
        return;

#ifdef HYDROGEN_HAVE_CUDA
    auto rbuf_on_device = IsGPUMemory(rbuf);

    // All memory is on the device
    if (rbuf_on_device)
        aluminum_allreduce<GPUBackend>(rbuf, count, op, std::move(comm));
    else
        aluminum_allreduce<CPUBackend>(rbuf, count, op, std::move(comm));
#else
    aluminum_allreduce<CPUBackend>(rbuf, count, op, std::move(comm));
#endif // HYDROGEN_HAVE_CUDA
}

template <typename T,
          typename/*=DisableIf<IsAluminumTypeT<T>>*/,
          typename/*=void*/>
void AllReduce(T* rbuf, int count, Op op, Comm comm)
{
    EL_DEBUG_CSE
    if (count != 0)
        fallback_allreduce(rbuf, count, op, std::move(comm));
}

//
// Things that call the "REAL" thing
//

template<typename T>
void AllReduce(const T* sbuf, T* rbuf, int count, Comm comm)
EL_NO_RELEASE_EXCEPT
{ AllReduce(sbuf, rbuf, count, SUM, std::move(comm)); }

template<typename T>
T AllReduce(T sb, Op op, Comm comm)
EL_NO_RELEASE_EXCEPT
{ T rb; AllReduce(&sb, &rb, 1, op, std::move(comm)); return rb; }

template<typename T>
T AllReduce(T sb, Comm comm)
EL_NO_RELEASE_EXCEPT
{ return AllReduce(sb, SUM, std::move(comm)); }

template<typename T>
void AllReduce(T* buf, int count, Comm comm)
EL_NO_RELEASE_EXCEPT
{ AllReduce(buf, count, SUM, std::move(comm)); }


#define MPI_ALLREDUCE_PROTO(T) \
  template void AllReduce \
  (const T* sbuf, T* rbuf, int count, Op op, Comm comm) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce(const T* sbuf, T* rbuf, int count, Comm comm) \
  EL_NO_RELEASE_EXCEPT; \
  template T AllReduce(T sb, Op op, Comm comm) \
  EL_NO_RELEASE_EXCEPT; \
  template T AllReduce(T sb, Comm comm) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce(T* buf, int count, Op op, Comm comm) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce(T* buf, int count, Comm comm) \
  EL_NO_RELEASE_EXCEPT;

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

} // namespace mpi
} // namespace El

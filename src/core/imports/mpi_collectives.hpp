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
#include "Al.hpp"
#ifdef HYDROGEN_HAVE_NCCL2
#include "nccl_impl.hpp"
#endif
#endif //HYDROGEN_HAVE_ALUMINUM

typedef unsigned char* UCP;

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
#ifndef HYDROGEN_HAVE_NCCL2
using backend = ::Al::MPIBackend;
#else
using backend = ::Al::NCCLBackend;
#endif

template<typename Real>
void aluminum_allreduce( const Real* sbuf, Real* rbuf, int count, Op op, Comm comm )
{
    MPI_Op mpi_op = op.op;
    Al::ReductionOperator red_op = Al::internal::mpi::MPI_Op2ReductionOperator (mpi_op);

    if(TypeMap<Real>() == MPI_UNSIGNED_CHAR){
      Al::Allreduce<backend, unsigned char>((unsigned char*)sbuf, (unsigned char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_INT){
      Al::Allreduce<backend, int>((int*)sbuf, (int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_CHAR){
      Al::Allreduce<backend, char>((char*)sbuf, (char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_UNSIGNED){
      Al::Allreduce<backend, unsigned int>((unsigned int*)sbuf, (unsigned int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_LONG_LONG){
      Al::Allreduce<backend, long long int>((long long int*)sbuf, (long long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_UNSIGNED_LONG_LONG){
      Al::Allreduce<backend, unsigned long long int>((unsigned long long int*)sbuf, (unsigned long long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_FLOAT){
      Al::Allreduce<backend, float>((float*)sbuf, (float*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_DOUBLE){
      Al::Allreduce<backend, double>((double*)sbuf, (double*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
#ifndef HYDROGEN_USES_NCCL2
    else if(TypeMap<Real>() == MPI_SIGNED_CHAR){
      Al::Allreduce<backend, signed char>((signed char*)sbuf, (signed char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_SHORT){
      Al::Allreduce<backend, short>((short*)sbuf, (short*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_UNSIGNED_SHORT){
      Al::Allreduce<backend, unsigned short>((unsigned short*)sbuf, (unsigned short*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_LONG_INT){
      Al::Allreduce<backend, long int>((long int*)sbuf, (long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_UNSIGNED_LONG){
      Al::Allreduce<backend, unsigned long int>((unsigned long int*)sbuf, (unsigned long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(TypeMap<Real>() == MPI_LONG_DOUBLE){
      Al::Allreduce<backend, long double>((long double*)sbuf, (long double*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
#endif
    else{
      MPI_Op opC = NativeOp<Real>( op );
      EL_CHECK_MPI
      ( MPI_Allreduce
            ( const_cast<Real*>(sbuf), rbuf, count, TypeMap<Real>(), opC,
              comm.comm ) );
    }
}

template<typename Real>
void aluminum_allreduce( Real* rbuf, int count, Op op, Comm comm )
{
    MPI_Op mpi_op = op.op;
    Al::ReductionOperator red_op = Al::internal::mpi::MPI_Op2ReductionOperator (mpi_op);

    if(typeid(TypeMap<Real>()) == typeid(MPI_CHAR)){
      Al::Allreduce<backend, char>((char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_UNSIGNED_CHAR)){
      Al::Allreduce<backend, unsigned char>((unsigned char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_INT)){
      Al::Allreduce<backend, int>((int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_UNSIGNED)){
      Al::Allreduce<backend, unsigned int>((unsigned int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_LONG_LONG)){
      Al::Allreduce<backend, long long int>((long long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_UNSIGNED_LONG_LONG)){
      Al::Allreduce<backend, unsigned long long int>((unsigned long long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_FLOAT)){
      Al::Allreduce<backend, float>((float*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_DOUBLE)){
      Al::Allreduce<backend, double>((double*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
#ifndef HYDROGEN_USES_NCCL2
    else if(typeid(TypeMap<Real>()) == typeid(MPI_SIGNED_CHAR)){
      Al::Allreduce<backend, signed char>((signed char*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_SHORT)){
      Al::Allreduce<backend, short>((short*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_UNSIGNED_SHORT)){
      Al::Allreduce<backend, unsigned short>((unsigned short*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_LONG_INT)){
      Al::Allreduce<backend, long int>((long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_UNSIGNED_LONG)){
      Al::Allreduce<backend, unsigned long int>((unsigned long int*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
    else if(typeid(TypeMap<Real>()) == typeid(MPI_LONG_DOUBLE)){
      Al::Allreduce<backend, long double>((long double*)rbuf, count, red_op, *(comm.aluminum_comm));
    }
#endif
    else{
      MPI_Op opC = NativeOp<Real>( op );
      EL_CHECK_MPI
      ( MPI_Allreduce
        ( MPI_IN_PLACE, rbuf, count, TypeMap<Real>(), opC, comm.comm ) );
    }
}
#endif

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( const Real* sbuf, Real* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count != 0 )
    {
#ifndef HYDROGEN_HAVE_ALUMINUM
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( const_cast<Real*>(sbuf), rbuf, count, TypeMap<Real>(), opC,
            comm.comm ) );
#else
    aluminum_allreduce<Real>(sbuf, rbuf, count, op, comm );
#endif
    }
}

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce
( const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count != 0 )
    {
#ifdef EL_AVOID_COMPLEX_MPI
        if( op == SUM )
        {
            MPI_Op opC = NativeOp<Real>( op );
            EL_CHECK_MPI
            ( MPI_Allreduce
                ( const_cast<Complex<Real>*>(sbuf),
                  rbuf, 2*count, TypeMap<Real>(), opC, comm.comm ) );
        }
        else
        {
            MPI_Op opC = NativeOp<Complex<Real>>( op );
            EL_CHECK_MPI
            ( MPI_Allreduce
              ( const_cast<Complex<Real>*>(sbuf),
                rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm ) );
        }
#else
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( const_cast<Complex<Real>*>(sbuf),
            rbuf, count, TypeMap<Complex<Real>>(), opC, comm.comm ) );
#endif
    }
}

template<typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllReduce
( const T* sbuf, T* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 )
        return;

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
//XXX
//CHECK WHAT SERIALIZE AND DESERIALIZE DO
//
    Serialize( count, sbuf, packedSend );

    ReserveSerialized( count, rbuf, packedRecv );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
        opC, comm.comm ) );
    Deserialize( count, packedRecv, rbuf );
}

template<typename T>
void AllReduce( const T* sbuf, T* rbuf, int count, Comm comm )
EL_NO_RELEASE_EXCEPT
{ AllReduce( sbuf, rbuf, count, SUM, comm ); }

template<typename T>
T AllReduce( T sb, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{ T rb; AllReduce( &sb, &rb, 1, op, comm ); return rb; }

template<typename T>
T AllReduce( T sb, Comm comm )
EL_NO_RELEASE_EXCEPT
{ return AllReduce( sb, SUM, comm ); }

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( Real* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 || Size(comm) == 1 )
        return;

#ifndef HYDROGEN_HAVE_ALUMINUM
    MPI_Op opC = NativeOp<Real>( op );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( MPI_IN_PLACE, buf, count, TypeMap<Real>(), opC, comm.comm ) );
#else
    aluminum_allreduce<Real>( buf, count, op, comm );
#endif
}

template<typename Real,
         typename/*=EnableIf<IsPacked<Real>>*/>
void AllReduce( Complex<Real>* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 || Size(comm) == 1 )
        return;

#ifdef EL_AVOID_COMPLEX_MPI
    if( op == SUM )
    {
        MPI_Op opC = NativeOp<Real>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( MPI_IN_PLACE, buf, 2*count, TypeMap<Real>(), opC, comm.comm ) );
    }
    else
    {
        MPI_Op opC = NativeOp<Complex<Real>>( op );
        EL_CHECK_MPI
        ( MPI_Allreduce
          ( MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(),
            opC, comm.comm ) );
    }
#else
    MPI_Op opC = NativeOp<Complex<Real>>( op );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( MPI_IN_PLACE, buf, count, TypeMap<Complex<Real>>(), opC,
        comm.comm ) );
#endif
}

template<typename T,
         typename/*=DisableIf<IsPacked<T>>*/,
         typename/*=void*/>
void AllReduce( T* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if( count == 0 )
        return;

    MPI_Op opC = NativeOp<T>( op );
    std::vector<byte> packedSend, packedRecv;
    Serialize( count, buf, packedSend );

    ReserveSerialized( count, buf, packedRecv );
    EL_CHECK_MPI
    ( MPI_Allreduce
      ( packedSend.data(), packedRecv.data(), count, TypeMap<T>(),
        opC, comm.comm ) );
    Deserialize( count, packedRecv, buf );
}

template<typename T>
void AllReduce( T* buf, int count, Comm comm )
EL_NO_RELEASE_EXCEPT
{ AllReduce( buf, count, SUM, comm ); }


#define MPI_ALLREDUCE_PROTO(T) \
  template void AllReduce \
  ( const T* sbuf, T* rbuf, int count, Op op, Comm comm ) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce( const T* sbuf, T* rbuf, int count, Comm comm ) \
  EL_NO_RELEASE_EXCEPT; \
  template T AllReduce( T sb, Op op, Comm comm ) \
  EL_NO_RELEASE_EXCEPT; \
  template T AllReduce( T sb, Comm comm ) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce( T* buf, int count, Op op, Comm comm ) \
  EL_NO_RELEASE_EXCEPT; \
  template void AllReduce( T* buf, int count, Comm comm ) \
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

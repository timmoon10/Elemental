/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_IMPORTS_MPI_HPP
#define EL_IMPORTS_MPI_HPP

#if defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)
#include "cuda.hpp"
#endif // defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)

#include "aluminum.hpp"

namespace El
{

using std::function;
using std::vector;

namespace mpi
{

#if defined(EL_HAVE_MPI3_NONBLOCKING_COLLECTIVES) || \
    defined(EL_HAVE_MPIX_NONBLOCKING_COLLECTIVES)
#define EL_HAVE_NONBLOCKING 1
#else
#define EL_HAVE_NONBLOCKING 0
#endif

#ifdef EL_HAVE_NONBLOCKING_COLLECTIVES
#ifdef EL_HAVE_MPI3_NONBLOCKING_COLLECTIVES
#define EL_NONBLOCKING_COLL(name) MPI_ ## name
#else
#define EL_NONBLOCKING_COLL(name) MPIX_ ## name
#endif
#endif

// Yes, I realize there's some code duplication here, but it's SO MUCH
// EASIER for the human to read I really don't care... The
// preprocessor is the only thing that could care, but it doesn't get
// feelings. Bwahahaha.

#ifndef HYDROGEN_HAVE_ALUMINUM

struct Comm
{
    MPI_Comm comm;
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD) EL_NO_EXCEPT : comm(mpiComm) { }

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;

};

#else
namespace internal
{
struct DelayCtorType {};
}// namespace internal

struct Comm
{
#if defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)
    using aluminum_comm_type = Head<BackendsForDevice<Device::GPU>>::comm_type;
#else
    using aluminum_comm_type = Head<BackendsForDevice<Device::CPU>>::comm_type;
#endif // defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)

    // Hack to handle global objects, MPI_COMM could be int or void*...
    explicit Comm(internal::DelayCtorType const&,
                  MPI_Comm mpicomm) EL_NO_EXCEPT : comm(mpicomm) {}
#ifdef HYDROGEN_HAVE_NCCL2
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD);
#else
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD) EL_NO_EXCEPT;
#endif

    // These do soft copies
    Comm(Comm const& comm_cpy) EL_NO_EXCEPT = default;
    Comm& operator=(Comm const& comm_cpy) EL_NO_EXCEPT = default;
    Comm(Comm&& comm_cpy) EL_NO_EXCEPT = default;
    Comm& operator=(Comm&& comm_cpy) EL_NO_EXCEPT = default;

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;

    MPI_Comm comm;
    std::shared_ptr<aluminum_comm_type> aluminum_comm;
};


inline
#ifdef HYDROGEN_HAVE_NCCL2
Comm::Comm(MPI_Comm mpiComm)
#else
Comm::Comm(MPI_Comm mpiComm) EL_NO_EXCEPT
#endif
: comm(mpiComm),
    aluminum_comm{
    std::make_shared<aluminum_comm_type>(
        mpiComm
#if defined(HYDROGEN_HAVE_NCCL2) || defined(HYDROGEN_HAVE_AL_MPI_CUDA)
        , GPUManager::Stream()
#endif
        )}
{}


#endif // HYDROGEN_HAVE_ALUMINUM

inline bool operator==( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm == b.comm; }
inline bool operator!=( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm != b.comm; }
// Hopefully, despite the fact that MPI_Comm is opaque, the following will
// reliably hold (otherwise it must be extended). Typically, MPI_Comm is
// either 'int' or 'void*'.
inline bool operator<( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm < b.comm; }

struct Group
{
    MPI_Group group;
    Group( MPI_Group mpiGroup=MPI_GROUP_NULL ) EL_NO_EXCEPT
    : group(mpiGroup) { }

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;
};
inline bool operator==( const Group& a, const Group& b ) EL_NO_EXCEPT
{ return a.group == b.group; }
inline bool operator!=( const Group& a, const Group& b ) EL_NO_EXCEPT
{ return a.group != b.group; }

struct Op
{
    MPI_Op op;
    Op( MPI_Op mpiOp=MPI_SUM ) EL_NO_EXCEPT : op(mpiOp) { }
};
inline bool operator==( const Op& a, const Op& b ) EL_NO_EXCEPT
{ return a.op == b.op; }
inline bool operator!=( const Op& a, const Op& b ) EL_NO_EXCEPT
{ return a.op != b.op; }

// Datatype definitions
// TODO(poulson): Convert these to structs/classes
typedef MPI_Aint Aint;
typedef MPI_Datatype Datatype;
typedef MPI_Errhandler ErrorHandler;
typedef MPI_Status Status;
typedef MPI_User_function UserFunction;

template<typename T>
struct Request
{
    Request() { }

    MPI_Request backend;

    vector<byte> buffer;
    bool receivingPacked=false;
    int recvCount;
    T* unpackedRecvBuf;
};

// Standard constants
extern const int ANY_SOURCE;
extern const int ANY_TAG;
#ifdef EL_HAVE_MPI_QUERY_THREAD
extern const int THREAD_SINGLE;
extern const int THREAD_FUNNELED;
extern const int THREAD_SERIALIZED;
extern const int THREAD_MULTIPLE;
#else
extern const int THREAD_SINGLE;
extern const int THREAD_FUNNELED;
extern const int THREAD_SERIALIZED;
extern const int THREAD_MULTIPLE;
#endif

extern const int UNDEFINED;
extern const Group GROUP_NULL;
extern const Comm COMM_NULL;// = MPI_COMM_NULL;
extern const Comm COMM_SELF;// = MPI_COMM_SELF;
extern const Comm COMM_WORLD;// = MPI_COMM_WORLD;
extern const ErrorHandler ERRORS_RETURN;
extern const ErrorHandler ERRORS_ARE_FATAL;
extern const Group GROUP_EMPTY;
extern const Op MAX;
extern const Op MIN;
extern const Op MAXLOC;
extern const Op MINLOC;
extern const Op PROD;
extern const Op SUM;
extern const Op LOGICAL_AND;
extern const Op LOGICAL_OR;
extern const Op LOGICAL_XOR;
extern const Op BINARY_AND;
extern const Op BINARY_OR;
extern const Op BINARY_XOR;

template<typename T>
struct Types
{
    static bool createdTypeBeforeResize;
    static El::mpi::Datatype typeBeforeResize;

    static bool createdType;
    static El::mpi::Datatype type;

    static bool haveSumOp;
    static bool createdSumOp;
    static El::mpi::Op sumOp;

    static bool haveProdOp;
    static bool createdProdOp;
    static El::mpi::Op prodOp;

    static bool haveMinOp;
    static bool createdMinOp;
    static El::mpi::Op minOp;

    static bool haveMaxOp;
    static bool createdMaxOp;
    static El::mpi::Op maxOp;

    static bool haveUserOp;
    static bool createdUserOp;
    static El::mpi::Op userOp;

    static bool haveUserCommOp;
    static bool createdUserCommOp;
    static El::mpi::Op userCommOp;

    static function<T(const T&,const T&)> userFunc, userCommFunc;

    // Internally called once per type between MPI_Init and MPI_Finalize
    static void Destroy();
};

template<typename T>
struct MPIBaseHelper { typedef T value; };
template<typename T>
struct MPIBaseHelper<ValueInt<T>> { typedef T value; };
template<typename T>
struct MPIBaseHelper<Entry<T>> { typedef T value; };
template<typename T>
using MPIBase = typename MPIBaseHelper<T>::value;

template<typename T>
Datatype& TypeMap() EL_NO_EXCEPT
{ return Types<T>::type; }

template<typename T>
Op& UserOp() { return Types<T>::userOp; }
template<typename T>
Op& UserCommOp() { return Types<T>::userCommOp; }
template<typename T>
Op& SumOp() { return Types<T>::sumOp; }
template<typename T>
Op& ProdOp() { return Types<T>::prodOp; }
// The following are currently only defined for real datatypes but could
// potentially use lexicographic ordering for complex numbers
template<typename T>
Op& MaxOp() { return Types<T>::maxOp; }
template<typename T>
Op& MinOp() { return Types<T>::minOp; }
template<typename T>
Op& MaxLocOp() { return Types<ValueInt<T>>::maxOp; }
template<typename T>
Op& MinLocOp() { return Types<ValueInt<T>>::minOp; }
template<typename T>
Op& MaxLocPairOp() { return Types<Entry<T>>::maxOp; }
template<typename T>
Op& MinLocPairOp() { return Types<Entry<T>>::minOp; }

// Added constant(s)
const int MIN_COLL_MSG = 1; // minimum message size for collectives
inline int Pad( int count ) EL_NO_EXCEPT
{ return std::max(count,MIN_COLL_MSG); }

bool CommSameSizeAsInteger() EL_NO_EXCEPT;
bool GroupSameSizeAsInteger() EL_NO_EXCEPT;

// Environment routines
void Initialize( int& argc, char**& argv ) EL_NO_EXCEPT;
int InitializeThread( int& argc, char**& argv, int required ) EL_NO_EXCEPT;
void Finalize() EL_NO_EXCEPT;
bool Initialized() EL_NO_EXCEPT;
bool Finalized() EL_NO_EXCEPT;
int QueryThread() EL_NO_EXCEPT;
void Abort( Comm comm, int errCode ) EL_NO_EXCEPT;
double Time() EL_NO_EXCEPT;
void Create( UserFunction* func, bool commutes, Op& op ) EL_NO_RELEASE_EXCEPT;
void Free( Op& op ) EL_NO_RELEASE_EXCEPT;
void Free( Datatype& type ) EL_NO_RELEASE_EXCEPT;

// Communicator manipulation
int Rank( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;
int Size( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;
void Create
( Comm parentComm, Group subsetGroup, Comm& subsetComm ) EL_NO_RELEASE_EXCEPT;
void Dup( Comm original, Comm& duplicate ) EL_NO_RELEASE_EXCEPT;
void Split( Comm comm, int color, int key, Comm& newComm ) EL_NO_RELEASE_EXCEPT;
void Free( Comm& comm ) EL_NO_RELEASE_EXCEPT;
bool Congruent( Comm comm1, Comm comm2 ) EL_NO_RELEASE_EXCEPT;
void ErrorHandlerSet
( Comm comm, ErrorHandler errorHandler ) EL_NO_RELEASE_EXCEPT;

// Cartesian communicator routines
void CartCreate
( Comm comm, int numDims, const int* dimensions, const int* periods,
  bool reorder, Comm& cartComm ) EL_NO_RELEASE_EXCEPT;
void CartSub
( Comm comm, const int* remainingDims, Comm& subComm ) EL_NO_RELEASE_EXCEPT;

// Group manipulation
int Rank( Group group ) EL_NO_RELEASE_EXCEPT;
int Size( Group group ) EL_NO_RELEASE_EXCEPT;
void CommGroup( Comm comm, Group& group ) EL_NO_RELEASE_EXCEPT;
void Dup( Group group, Group& newGroup ) EL_NO_RELEASE_EXCEPT;
void Union( Group groupA, Group groupB, Group& newGroup ) EL_NO_RELEASE_EXCEPT;
void Incl
( Group group, int n, const int* ranks, Group& subGroup ) EL_NO_RELEASE_EXCEPT;
void Excl
( Group group, int n, const int* ranks, Group& subGroup ) EL_NO_RELEASE_EXCEPT;
void Difference
( Group parent, Group subset, Group& complement ) EL_NO_RELEASE_EXCEPT;
void Free( Group& group ) EL_NO_RELEASE_EXCEPT;
bool Congruent( Group group1, Group group2 ) EL_NO_RELEASE_EXCEPT;
int Translate
( Group origGroup, int origRank, Group newGroup ) EL_NO_RELEASE_EXCEPT;
int Translate
( Comm  origComm,  int origRank, Group newGroup ) EL_NO_RELEASE_EXCEPT;
int Translate
( Group origGroup, int origRank, Comm  newComm  ) EL_NO_RELEASE_EXCEPT;
int Translate
( Comm  origComm,  int origRank, Comm  newComm  ) EL_NO_RELEASE_EXCEPT;
void Translate
( Group origGroup, int size, const int* origRanks,
  Group newGroup,                  int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Comm origComm,  int size, const int* origRanks,
  Group newGroup,                 int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Group origGroup, int size, const int* origRanks,
  Comm newComm,                    int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Comm origComm, int size, const int* origRanks,
  Comm newComm,                  int* newRanks ) EL_NO_RELEASE_EXCEPT;

// Utilities
void Barrier( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void Wait( Request<T>& request ) EL_NO_RELEASE_EXCEPT;

template<typename T,
         typename=EnableIf<IsPacked<T>>>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void WaitAll( int numRequests, Request<T>* requests ) EL_NO_RELEASE_EXCEPT;

template<typename T,
         typename=EnableIf<IsPacked<T>>>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT;

template<typename T>
bool Test( Request<T>& request ) EL_NO_RELEASE_EXCEPT;
bool IProbe
( int source, int tag, Comm comm, Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
int GetCount( Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void SetUserReduceFunc
( function<T(const T&,const T&)> func, bool commutative=true )
{
    if( commutative )
        Types<T>::userCommFunc = func;
    else
        Types<T>::userFunc = func;
}

// Point-to-point communication
// ============================

// Send
// ----
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSend
( const Real* buf, int count, int to, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedSend( const T* buf, int count, int to, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;


// If the tag is irrelevant
template<typename T>
void Send( const T* buf, int count, int to, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the send-count is one
template<typename T>
void TaggedSend( T b, int to, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the send-count is one and the tag is irrelevant
template<typename T>
void Send( T b, int to, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Non-blocking send
// -----------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedISend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void ISend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedISend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void ISend( T b, int to, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// Non-blocking ready-mode send
// ----------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRSend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRSend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedIRSend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void IRSend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedIRSend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void IRSend( T b, int to, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// Non-blocking synchronous Send
// -----------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISSend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISSend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedISSend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void ISSend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedISSend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void ISSend( T b, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// Recv
// ----
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedRecv
( Real* buf, int count, int from, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedRecv
( Complex<Real>* buf, int count, int from, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedRecv
( T* buf, int count, int from, int tag, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void Recv( T* buf, int count, int from, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the recv count is one
template<typename T>
T TaggedRecv( int from, int tag, Comm comm ) EL_NO_RELEASE_EXCEPT;

// If the recv count is one and the tag is irrelevant
template<typename T>
T Recv( int from, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Non-blocking recv
// -----------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRecv
( Real* buf, int count, int from, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRecv
( Complex<Real>* buf, int count, int from, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedIRecv
( T* buf, int count, int from, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void IRecv( T* buf, int count, int from, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the recv count is one
template<typename T>
T TaggedIRecv( int from, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the recv count is one and the tag is irrelevant
template<typename T>
T IRecv( int from, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// SendRecv
// --------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv
( const Real* sbuf, int sc, int to,   int stag,
        Real* rbuf, int rc, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv
( const Complex<Real>* sbuf, int sc, int to,   int stag,
        Complex<Real>* rbuf, int rc, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedSendRecv
( const T* sbuf, int sc, int to,   int stag,
        T* rbuf, int rc, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the tags are irrelevant
template<typename T>
void SendRecv
( const T* sbuf, int sc, int to,
        T* rbuf, int rc, int from, Comm comm ) EL_NO_RELEASE_EXCEPT;

// If the send and recv counts are one
template<typename T>
T TaggedSendRecv
( T sb, int to, int stag, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the send and recv counts are one and the tags don't matter
template<typename T>
T SendRecv( T sb, int to, int from, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Single-buffer SendRecv
// ----------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv
( Real* buf, int count, int to, int stag, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv
( Complex<Real>* buf, int count, int to, int stag, int from, int rtag,
  Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedSendRecv
( T* buf, int count, int to, int stag, int from, int rtag, Comm comm )
EL_NO_RELEASE_EXCEPT;

// If the tags don't matter
template<typename T>
void SendRecv( T* buf, int count, int to, int from, Comm comm )
EL_NO_RELEASE_EXCEPT;

// Collective communication
// ========================

// Broadcast
// ---------
#define COLL Collective::BROADCAST

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void Broadcast(T* buffer, int count, int root, Comm comm, SyncInfo<D> const&);

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<Device::GPU> const& syncInfo);
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void Broadcast(Complex<T>* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void Broadcast(T* buffer, int count, int root, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=void, typename=void, typename=void>
void Broadcast(T*, int, int, Comm, SyncInfo<D> const&);

// If the message length is one
template<typename T, Device D>
void Broadcast( T& b, int root, Comm comm, SyncInfo<D> const& );

#undef COLL // Collective::BROADCAST

// Non-blocking broadcast
// ----------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IBroadcast
( Real* buf, int count, int root, Comm comm, Request<Real>& request );
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IBroadcast
( Complex<Real>* buf, int count, int root, Comm comm,
  Request<Complex<Real>>& request );
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void IBroadcast
( T* buf, int count, int root, Comm comm, Request<T>& request );

// If the message length is one
template<typename T>
void IBroadcast( T& b, int root, Comm comm, Request<T>& request );

// Gather
// ------
// Even though EL_AVOID_COMPLEX_MPI being defined implies that an std::vector
// copy of the input data will be created, and the memory allocation can clearly
// fail and throw an exception, said exception is not necessarily thrown on
// Linux platforms due to the "optimistic" allocation policy. Therefore we will
// go ahead and allow std::terminate to be called should such an std::bad_alloc
// exception occur in a Release build
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Gather
( const Real* sbuf, int sc,
        Real* rbuf, int rc, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void  Gather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,typename=void>
void Gather
( const T* sbuf, int sc,
        T* rbuf, int rc, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Non-blocking gather
// -------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IGather
( const Real* sbuf, int sc,
        Real* rbuf, int rc, int root, Comm comm,
  Request<Real>& request );
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IGather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc,
  int root, Comm comm,
  Request<Complex<Real>>& request );
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void IGather
( const T* sbuf, int sc,
        T* rbuf, int rc, int root, Comm comm,
  Request<T>& request );

// Gather with variable recv sizes
// -------------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Gather
( const Real* sbuf, int sc,
        Real* rbuf, const int* rcs, const int* rds,
  int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Gather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  int root, Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Gather
( const T* sbuf, int sc,
        T* rbuf, const int* rcs, const int* rds,
  int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

// AllGather
// ---------
// NOTE: See the corresponding note for Gather on std::bad_alloc exceptions

#define COLL Collective::ALLGATHER

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo);

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<Device::GPU> const& syncInfo);
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllGather(
    const T* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllGather(
    const Complex<T>* sbuf, int sc,
    Complex<T>* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void AllGather(
    T const* sbuf, int sc, T* rbuf, int rc, Comm comm,
    SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=void, typename=void, typename=void>
void AllGather(T const*, int, T*, int, Comm, SyncInfo<D> const&);

#undef COLL // Collective::ALLGATHER

// AllGather with variable recv sizes
// ----------------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllGather
( const Real* sbuf, int sc,
        Real* rbuf, const int* rcs, const int* rds, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllGather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void AllGather
( const T* sbuf, int sc,
        T* rbuf, const int* rcs, const int* rds, Comm comm )
EL_NO_RELEASE_EXCEPT;

// Scatter
// -------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scatter
( const Real* sbuf, int sc,
        Real* rbuf, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scatter
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scatter
( const T* sbuf, int sc,
        T* rbuf, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

// In-place option
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scatter( Real* buf, int sc, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scatter( Complex<Real>* buf, int sc, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scatter( T* buf, int sc, int rc, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

// TODO(poulson): MPI_Scatterv support

// AllToAll
// --------
// NOTE: See the corresponding note on std::bad_alloc for Gather
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Real* sbuf, int sc,
        Real* rbuf, int rc, Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void AllToAll
( const T* sbuf, int sc,
        T* rbuf, int rc, Comm comm ) EL_NO_RELEASE_EXCEPT;

// AllToAll with non-uniform send/recv sizes
// -----------------------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Real* sbuf, const int* scs, const int* sds,
        Real* rbuf, const int* rcs, const int* rds, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Complex<Real>* sbuf, const int* scs, const int* sds,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void AllToAll
( const T* sbuf, const int* scs, const int* sds,
        T* rbuf, const int* rcs, const int* rds, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T>
vector<T> AllToAll
( const vector<T>& sendBuf,
  const vector<int>& sendCounts,
  const vector<int>& sendDispls,
  Comm comm ) EL_NO_RELEASE_EXCEPT;

// Reduce
// ------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Reduce
( const Real* sbuf, Real* rbuf, int count, Op op, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Reduce
( const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op,
  int root, Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Reduce
( const T* sbuf, T* rbuf, int count, Op op, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Reduce
( const T* sb, T* rb, int count, OpClass op, bool commutative,
  int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Reduce( sb, rb, count, UserCommOp<T>(), root, comm );
    else
        Reduce( sb, rb, count, UserOp<T>(), root, comm );
}

// Default to SUM
template<typename T>
void Reduce( const T* sbuf, T* rbuf, int count, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

// With a message-size of one
template<typename T>
T Reduce( T sb, Op op, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
T Reduce
( T sb, OpClass op, bool commutative, int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        return Reduce( sb, UserCommOp<T>(), root, comm );
    else
        return Reduce( sb, UserOp<T>(), root, comm );
}

// With a message-size of one and default to SUM
template<typename T>
T Reduce( T sb, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Single-buffer reduce
// --------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Reduce( Real* buf, int count, Op op, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Reduce( Complex<Real>* buf, int count, Op op, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Reduce( T* buf, int count, Op op, int root, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Reduce
( T* buf, int count, OpClass op, bool commutative, int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Reduce( buf, count, UserCommOp<T>(), root, comm );
    else
        Reduce( buf, count, UserOp<T>(), root, comm );
}

// Default to SUM
template<typename T>
void Reduce( T* buf, int count, int root, Comm comm ) EL_NO_RELEASE_EXCEPT;

// AllReduce
// ---------

template <typename T, Device D,
          typename=EnableIf<IsAluminumDeviceType<T,D>>>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const&);

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumDeviceType<T,Device::GPU>>>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<Device::GPU> const&);
#endif // HYDROGEN_HAVE_CUDA

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllReduce(Complex<T> const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void AllReduce(T const* sbuf, T* rbuf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=void, typename=void, typename=void>
void AllReduce(T const*, T*, int, Op, Comm, SyncInfo<D> const&);

//
// The "IN_PLACE" allreduce
//

template <typename T, Device D,
          typename=EnableIf<IsAluminumDeviceType<T,D>>>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& /*syncInfo*/);
#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumDeviceType<T,Device::GPU>>>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<Device::GPU> const& /*syncInfo*/);
#endif // HYDROGEN_HAVE_CUDA

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=EnableIf<IsPacked<T>>>
void AllReduce(Complex<T>* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void AllReduce(T* buf, int count, Op op, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumDeviceType<T,D>>>>,
          typename=void, typename=void, typename=void>
void AllReduce(T*, int, Op, Comm, SyncInfo<D> const&);

template <typename T, Device D>
void AllReduce(const T* sbuf, T* rbuf, int count, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D>
T AllReduce(T sb, Op op, Comm comm, SyncInfo<D> const& syncInfo);

template <typename T, Device D>
T AllReduce(T sb, Comm comm, SyncInfo<D> const& syncInfo);

template <typename T, Device D>
void AllReduce(T* buf, int count, Comm comm, SyncInfo<D> const& syncInfo);

// ReduceScatter
// -------------
#define COLL Collective::REDUCESCATTER

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void ReduceScatter( T const* sbuf, T* rbuf, int rc, Op op, Comm comm,
                    SyncInfo<D> const& syncInfo );
#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>>
void ReduceScatter( T const* sbuf, T* rbuf, int rc, Op op, Comm comm,
                    SyncInfo<Device::GPU> const& syncInfo );
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template<typename T, Device D,
         typename=EnableIf<And<IsDeviceValidType<T,D>,
                               Not<IsAluminumSupported<T,D,COLL>>>>,
         typename=EnableIf<IsPacked<T>>>
void ReduceScatter(
    T const* sbuf, T* rbuf, int rc, Op op, Comm comm,
    SyncInfo<D> const& syncInfo );

template<typename T, Device D,
         typename=EnableIf<And<IsDeviceValidType<T,D>,
                               Not<IsAluminumSupported<T,D,COLL>>>>,
         typename=EnableIf<IsPacked<T>>>
void ReduceScatter(
    Complex<T> const* sbuf, Complex<T>* rbuf, int rc, Op op, Comm comm,
    SyncInfo<D> const& syncInfo );

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void ReduceScatter(
    T const* sbuf, T* rbuf, int rc, Op op, Comm comm,
    SyncInfo<D> const& syncInfo );

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=void, typename=void, typename=void>
void ReduceScatter(T const*, T*, int, Op, Comm, SyncInfo<D> const&);

// FIXME: WHAT TO DO HERE??
template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter
( const T* sb, T* rb, int count, OpClass op, bool commutative, Comm comm )
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( sb, rb, count, UserCommOp<T>(), comm );
    else
        ReduceScatter( sb, rb, count, UserOp<T>(), comm );
}

// Default to SUM
template <typename T, Device D>
void ReduceScatter( T const* sbuf, T* rbuf, int rc, Comm comm,
                    SyncInfo<D> const& syncInfo);

// Single-buffer ReduceScatter
// ---------------------------

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo);
#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>>
void ReduceScatter( T* buf, int count, Op op, Comm comm,
                    SyncInfo<Device::GPU> const& syncInfo );
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=EnableIf<IsPacked<T>>>
void ReduceScatter(Complex<T>* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void ReduceScatter(T* buf, int count, Op op, Comm comm,
                   SyncInfo<D> const& syncInfo);

template <typename T, Device D,
          typename=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>>,
          typename=void, typename=void, typename=void>
void ReduceScatter(T*, int, Op, Comm, SyncInfo<D> const&);

// FIXME: WHAT TO DO HERE??
template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter
( T* buf, int count, OpClass op, bool commutative, Comm comm )
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( buf, count, UserCommOp<T>(), comm );
    else
        ReduceScatter( buf, count, UserOp<T>(), comm );
}

// Default to SUM
template <typename T, Device D>
void ReduceScatter(T* buf, int rc, Comm comm, SyncInfo<D> const& syncInfo);

#undef COLL // Collectives::REDUCESCATTER

// Variable-length ReduceScatter
// -----------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void ReduceScatter
( const Real* sbuf, Real* rbuf, const int* rcs, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void ReduceScatter
( const Complex<Real>* sbuf, Complex<Real>* rbuf, const int* rcs, Op op,
  Comm comm ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void ReduceScatter
( const T* sbuf, T* rbuf, const int* rcs, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter
( const T* sb, T* rb, const int* rcs, OpClass op, bool commutative,
  Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( sb, rb, rcs, UserCommOp<T>(), comm );
    else
        ReduceScatter( sb, rb, rcs, UserOp<T>(), comm );
}

// Default to SUM
template<typename T>
void ReduceScatter( const T* sbuf, T* rbuf, const int* rcs, Comm comm )
EL_NO_RELEASE_EXCEPT;

// Scan
// ----
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scan( const Real* sbuf, Real* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scan
( const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scan( const T* sbuf, T* rbuf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Scan
( const T* sb, T* rb, int count, OpClass op, bool commutative,
  int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Scan( sb, rb, count, UserCommOp<T>(), root, comm );
    else
        Scan( sb, rb, count, UserOp<T>(), root, comm );
}

// Default to SUM
template<typename T>
void Scan( const T* sbuf, T* rbuf, int count, Comm comm )
EL_NO_RELEASE_EXCEPT;

// With a message-size of one
template<typename T>
T Scan( T sb, Op op, Comm comm ) EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
T Scan( T sb, OpClass op, bool commutative, int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        return Scan( sb, UserCommOp<T>(), root, comm );
    else
        return Scan( sb, UserOp<T>(), root, comm );
}

// With a message-size of one and default to SUM
template<typename T>
T Scan( T sb, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Single-buffer scan
// ------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scan( Real* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void Scan( Complex<Real>* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scan( T* buf, int count, Op op, Comm comm )
EL_NO_RELEASE_EXCEPT;

template<typename T,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Scan
( T* buf, int count, OpClass op, bool commutative, int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Scan( buf, count, UserCommOp<T>(), root, comm );
    else
        Scan( buf, count, UserOp<T>(), root, comm );
}

// Default to SUM
template<typename T>
void Scan( T* buf, int count, Comm comm ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void SparseAllToAll
( const vector<T>& sendBuffer,
  const vector<int>& sendCounts,
  const vector<int>& sendOffs,
        vector<T>& recvBuffer,
  const vector<int>& recvCounts,
  const vector<int>& recvOffs,
        Comm comm ) EL_NO_RELEASE_EXCEPT;

void VerifySendsAndRecvs
( const vector<int>& sendCounts,
  const vector<int>& recvCounts, Comm comm );

void CreateCustom() EL_NO_RELEASE_EXCEPT;
void DestroyCustom() EL_NO_RELEASE_EXCEPT;

#ifdef HYDROGEN_HAVE_MPC
void CreateBigIntFamily();
void DestroyBigIntFamily();
void CreateBigFloatFamily();
void DestroyBigFloatFamily();
#endif

// Convenience functions which might not be very useful
int Comm::Rank() const EL_NO_RELEASE_EXCEPT { return mpi::Rank(*this); }
int Comm::Size() const EL_NO_RELEASE_EXCEPT { return mpi::Size(*this); }
int Group::Rank() const EL_NO_RELEASE_EXCEPT { return mpi::Rank(*this); }
int Group::Size() const EL_NO_RELEASE_EXCEPT { return mpi::Size(*this); }

} // mpi
} // elem

#endif // ifndef EL_IMPORTS_MPI_HPP

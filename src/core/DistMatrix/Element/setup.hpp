/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

// This file should be included into each of the DistMatrix specializations
// as a workaround for the fact that C++11 constructor inheritance is not
// yet widely supported.

#include "El/blas_like/level1/Copy/internal_impl.hpp"

namespace El {

#define DM DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>
#define EM ElementalMatrix<T>
#define ADM AbstractDistMatrix<T>

// Public section
// ##############

// Constructors and destructors
// ============================

template <typename T, Device D>
DM::DistMatrix(const El::Grid& grid, int root)
: EM(grid,root)
{
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
}

template <typename T, Device D>
DM::DistMatrix(Int height, Int width, const El::Grid& grid, int root)
: EM(grid,root)
{
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
    this->Resize(height,width);
}

template <typename T, Device D>
DM::DistMatrix(const DM& A)
: EM(A.Grid())
{
    EL_DEBUG_CSE
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
    if (&A != this)
        *this = A;
    else
        LogicError("Tried to construct DistMatrix with itself");
}

template <typename T, Device D>
template<Dist U,Dist V>
DM::DistMatrix(const DistMatrix<T,U,V,ELEMENT,D>& A)
: EM(A.Grid())
{
    EL_DEBUG_CSE
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
    if (COLDIST != U || ROWDIST != V ||
        reinterpret_cast<const DM*>(&A) != this)
        *this = A;
    else
        LogicError("Tried to construct DistMatrix with itself");
}

template <typename T, Device D>
DM::DistMatrix(const AbstractDistMatrix<T>& A)
: EM(A.Grid())
{
    EL_DEBUG_CSE

    if (A.GetLocalDevice() != D)
        LogicError("No cross-device construction yet!");
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                  \
    (A.ColDist() == CDIST) && (A.RowDist() == RDIST) && (A.Wrap() == WRAP) \
        && (A.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
    auto& ACast =                                                       \
        static_cast<const DistMatrix<T,CDIST,RDIST,WRAP,DEVICE>&>(A);   \
    if (COLDIST != CDIST || ROWDIST != RDIST || ELEMENT != WRAP ||      \
        reinterpret_cast<const DM*>(&A) != this)                        \
        *this = ACast;                                                  \
    else                                                                \
        LogicError("Tried to construct DistMatrix with itself");
#include "El/macros/DeviceGuardAndPayload.h"
}

template <typename T, Device D>
DM::DistMatrix(const ElementalMatrix<T>& A)
: EM(A.Grid())
{
    EL_DEBUG_CSE
    if (A.GetLocalDevice() != D)
        LogicError("No cross-device construction yet!");
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                 \
    (A.DistData().colDist == CDIST) && (A.DistData().rowDist == RDIST) \
        && (ELEMENT == WRAP) && (A.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
    auto& ACast =                                                       \
        static_cast<const DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(A); \
    if (COLDIST != CDIST || ROWDIST != RDIST ||                         \
        reinterpret_cast<const DM*>(&A) != this)                       \
        *this = ACast;                                                  \
    else                                                                \
        LogicError("Tried to construct DistMatrix with itself");
#include "El/macros/DeviceGuardAndPayload.h"
}

template <typename T, Device D>
template<Dist U,Dist V>
DM::DistMatrix(const DistMatrix<T,U,V,BLOCK,D>& A)
: EM(A.Grid())
{
    EL_DEBUG_CSE
    if (COLDIST != CIRC || ROWDIST != CIRC)
        this->Matrix().FixSize();
    this->SetShifts();
    *this = A;
}

template <typename T, Device D>
DM::DistMatrix(DM&& A) EL_NO_EXCEPT : EM(std::move(A)) { }

template <typename T, Device D> DM::~DistMatrix() { }

template <typename T, Device D>
DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>* DM::Copy() const
{ return new DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>(*this); }

template <typename T, Device D>
DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>* DM::Construct
(const El::Grid& g, int root) const
{ return new DistMatrix<T,COLDIST,ROWDIST,ELEMENT,D>(g,root); }

template <typename T, Device D>
DistMatrix<T,ROWDIST,COLDIST,ELEMENT,D>* DM::ConstructTranspose
(const El::Grid& g, int root) const
{ return new DistMatrix<T,ROWDIST,COLDIST,ELEMENT,D>(g,root); }

template <typename T, Device D>
typename DM::diagType*
DM::ConstructDiagonal
(const El::Grid& g, int root) const
{ return new DistMatrix<T,DiagCol<COLDIST,ROWDIST>(),
                        DiagRow<COLDIST,ROWDIST>(),ELEMENT,D>(g,root); }

// Operator overloading
// ====================

// Return a view
// -------------
template <typename T, Device D>
DM DM::operator()(Range<Int> I, Range<Int> J)
{
    EL_DEBUG_CSE
    return View(*this, I, J);
}

template <typename T, Device D>
const DM DM::operator()(Range<Int> I, Range<Int> J) const
{
    EL_DEBUG_CSE
    return LockedView(*this, I, J);
}

// Non-contiguous
// --------------
template <typename T, Device D>
DM DM::operator()(Range<Int> I, const vector<Int>& J) const
{
    EL_DEBUG_CSE
    DM ASub(this->Grid());
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template <typename T, Device D>
DM DM::operator()(const vector<Int>& I, Range<Int> J) const
{
    EL_DEBUG_CSE
    DM ASub(this->Grid());
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template <typename T, Device D>
DM DM::operator()(const vector<Int>& I, const vector<Int>& J) const
{
    EL_DEBUG_CSE
    DM ASub(this->Grid());
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

// Copy
// ----
template <typename T, Device D>
DM& DM::operator=(const DM& A)
{
    EL_DEBUG_CSE
    copy::Translate(A, *this);
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(const AbstractDistMatrix<T>& A)
{
    EL_DEBUG_CSE;
    if (A.GetLocalDevice() != D)
        LogicError("No cross-device construction yet!");

    // TODO: Use either AllGather or Gather if the distribution of this matrix
    //       is respectively either (STAR,STAR) or (CIRC,CIRC)
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                        \
    (A.ColDist() == CDIST) && (A.RowDist() == RDIST) && (A.Wrap() == WRAP) \
        && (A.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
    auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST,WRAP,DEVICE>&>(A); \
    *this = ACast;
#include "El/macros/DeviceGuardAndPayload.h"
    return *this;
}

template <typename T, Device D>
template<Dist U,Dist V>
DM& DM::operator=(const DistMatrix<T,U,V,BLOCK,D>& A)
{
    EL_DEBUG_CSE
    // TODO(poulson):
    // Use either AllGather or Gather if the distribution of this matrix
    // is respectively either (STAR,STAR) or (CIRC,CIRC)
    //
    // TODO(poulson): Avoid the GeneralPurpose redistribution in more cases
    const bool elemColCompat =
      (A.BlockHeight() == 1 || A.ColStride() == 1);
    const bool elemRowCompat =
      (A.BlockWidth() == 1 || A.RowStride() == 1);
    if (elemColCompat && elemRowCompat)
    {
        DistMatrix<T,U,V> AElemView(A.Grid());
#if 0
        AElemView.LockedAttach
        (A.Height(), A.Width(), A.Grid(),
          A.ColAlign(), A.RowAlign(), A.LockedBuffer(), A.LDim(), A.Root());
#else
        throw std::runtime_error("This don't work yet!");
#endif
        *this = AElemView;
    }
    else
    {
        copy::GeneralPurpose(A, *this);
    }
    return *this;
}

template <typename T, Device D>
DM& DM::operator=(DM&& A)
{
    EL_DEBUG_CSE
    if (this->Viewing() || A.Viewing())
        this->operator=((const DM&)A);
    else
        EM::operator=(std::move(A));
    return *this;
}

// Rescaling
// ---------
template <typename T, Device D>
const DM& DM::operator*=(T alpha)
{
    EL_DEBUG_CSE
    Scale(alpha, *this);
    return *this;
}

// Addition/subtraction
// --------------------
template <typename T, Device D>
const DM& DM::operator+=(const EM& A)
{
    EL_DEBUG_CSE
    Axpy(T(1), A, *this);
    return *this;
}

template <typename T, Device D>
const DM& DM::operator+=(const ADM& A)
{
    EL_DEBUG_CSE
    Axpy(T(1), A, *this);
    return *this;
}

template <typename T, Device D>
const DM& DM::operator-=(const EM& A)
{
    EL_DEBUG_CSE
    Axpy(T(-1), A, *this);
    return *this;
}

template <typename T, Device D>
const DM& DM::operator-=(const ADM& A)
{
    EL_DEBUG_CSE
    Axpy(T(-1), A, *this);
    return *this;
}

// Distribution data
// =================
template <typename T, Device D>
Dist DM::ColDist() const EL_NO_EXCEPT { return COLDIST; }
template <typename T, Device D>
Dist DM::RowDist() const EL_NO_EXCEPT { return ROWDIST; }

template <typename T, Device D>
Dist DM::PartialColDist() const EL_NO_EXCEPT { return Partial<COLDIST>(); }
template <typename T, Device D>
Dist DM::PartialRowDist() const EL_NO_EXCEPT { return Partial<ROWDIST>(); }

template <typename T, Device D>
Dist DM::PartialUnionColDist() const EL_NO_EXCEPT
{ return PartialUnionCol<COLDIST,ROWDIST>(); }
template <typename T, Device D>
Dist DM::PartialUnionRowDist() const EL_NO_EXCEPT
{ return PartialUnionRow<COLDIST,ROWDIST>(); }

template <typename T, Device D>
Dist DM::CollectedColDist() const EL_NO_EXCEPT { return Collect<COLDIST>(); }
template <typename T, Device D>
Dist DM::CollectedRowDist() const EL_NO_EXCEPT { return Collect<ROWDIST>(); }

// Single-entry manipulation
// =========================

// Global entry manipulation
// -------------------------

template <typename T, Device D>
T
DM::Get(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (!grid_->InGrid())
          LogicError("Get should only be called in-grid");
  )
    T value;
    if (CrossRank() == this->Root())
    {
        const int owner = this->Owner(i, j);
        if (owner == DistRank())
            value = GetLocal(this->LocalRow(i), this->LocalCol(j));
        mpi::Broadcast(value, owner, DistComm());
    }
    mpi::Broadcast(value, this->Root(), CrossComm());
    return value;
}

template <typename T, Device D>
Base<T>
DM::GetRealPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (!grid_->InGrid())
          LogicError("Get should only be called in-grid");
  )
    Base<T> value;
    if (CrossRank() == this->Root())
    {
        const int owner = this->Owner(i, j);
        if (owner == DistRank())
            value = GetLocalRealPart(this->LocalRow(i), this->LocalCol(j));
        mpi::Broadcast(value, owner, DistComm());
    }
    mpi::Broadcast(value, this->Root(), CrossComm());
    return value;
}

template <typename T, Device D>
Base<T>
DM::GetImagPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (!grid_->InGrid())
          LogicError("Get should only be called in-grid");
    )
    Base<T> value;
    if (IsComplex<T>::value)
    {
        if (CrossRank() == this->Root())
        {
            const int owner = this->Owner(i, j);
            if (owner == DistRank())
                value = GetLocalRealPart(this->LocalRow(i), this->LocalCol(j));
            mpi::Broadcast(value, owner, DistComm());
        }
        mpi::Broadcast(value, this->Root(), CrossComm());
    }
    else
        value = 0;
    return value;
}

template <typename T, Device D>
void
DM::Set(Int i, Int j, T value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        SetLocal(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void
DM::Set(const Entry<T>& entry)
EL_NO_RELEASE_EXCEPT
{ Set(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void
DM::SetRealPart(Int i, Int j, Base<T> value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        SetLocalRealPart(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void
DM::SetRealPart(const Entry<Base<T>>& entry)
EL_NO_RELEASE_EXCEPT
{ SetRealPart(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void DM::SetImagPart(Int i, Int j, Base<T> value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        SetLocalImagPart(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void DM::SetImagPart(const Entry<Base<T>>& entry)
EL_NO_RELEASE_EXCEPT
{ SetImagPart(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void
DM::Update(Int i, Int j, T value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        UpdateLocal(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void
DM::Update(const Entry<T>& entry)
EL_NO_RELEASE_EXCEPT
{ Update(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void
DM::UpdateRealPart(Int i, Int j, Base<T> value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        UpdateLocalRealPart(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void
DM::UpdateRealPart(const Entry<Base<T>>& entry)
EL_NO_RELEASE_EXCEPT
{ UpdateRealPart(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void DM::UpdateImagPart(Int i, Int j, Base<T> value)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        UpdateLocalImagPart(this->LocalRow(i), this->LocalCol(j), value);
}

template <typename T, Device D>
void DM::UpdateImagPart(const Entry<Base<T>>& entry)
EL_NO_RELEASE_EXCEPT
{ UpdateImagPart(entry.i, entry.j, entry.value); }

template <typename T, Device D>
void
DM::MakeReal(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        MakeLocalReal(this->LocalRow(i), this->LocalCol(j));
}

template <typename T, Device D>
void
DM::Conjugate(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    if (this->IsLocal(i,j))
        ConjugateLocal(this->LocalRow(i), this->LocalCol(j));
}

// Batch remote updates
// --------------------
template <typename T, Device D>
void DM::Reserve(Int numRemoteUpdates)
{
    EL_DEBUG_CSE
    const Int currSize = remoteUpdates_.size();
    remoteUpdates_.reserve(currSize+numRemoteUpdates);
}

template <typename T, Device D>
void DM::QueueUpdate(const Entry<T>& entry)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    // NOTE: We cannot always simply locally update since it can (and has)
    //       lead to the processors in the same redundant communicator having
    //       different results after ProcessQueues()
    if (RedundantSize() == 1 && this->IsLocal(entry.i,entry.j))
        UpdateLocal(this->LocalRow(entry.i), this->LocalCol(entry.j), entry.value);
    else
        remoteUpdates_.push_back(entry);
}

template <typename T, Device D>
void DM::QueueUpdate(Int i, Int j, T value)
EL_NO_RELEASE_EXCEPT
{ QueueUpdate(Entry<T>{i,j,value}); }

template <typename T, Device D>
void DM::ProcessQueues(bool includeViewers)
{
    EL_DEBUG_CSE
    const auto& grid = Grid();
    const Dist colDist = ColDist();
    const Dist rowDist = RowDist();
    const Int totalSend = remoteUpdates_.size();

    // We will first push to redundant rank 0
    const int redundantRoot = 0;

    // Compute the metadata
    // ====================
    mpi::Comm comm;
    vector<int> sendCounts, owners(totalSend);
    if (includeViewers)
    {
        comm = grid.ViewingComm();
        const int viewingSize = mpi::Size(grid.ViewingComm());
        sendCounts.resize(viewingSize,0);
        for(Int k=0; k<totalSend; ++k)
        {
            const Entry<T>& entry = remoteUpdates_[k];
            const int distOwner = this->Owner(entry.i,entry.j);
            const int vcOwner =
              grid.CoordsToVC(colDist,rowDist,distOwner,redundantRoot);
            owners[k] = grid.VCToViewing(vcOwner);
            ++sendCounts[owners[k]];
        }
    }
    else
    {
        if (!this->Participating())
            return;
        comm = grid.VCComm();
        const int vcSize = mpi::Size(grid.VCComm());
        sendCounts.resize(vcSize,0);
        for(Int k=0; k<totalSend; ++k)
        {
            const Entry<T>& entry = remoteUpdates_[k];
            const int distOwner = this->Owner(entry.i,entry.j);
            owners[k] =
              grid.CoordsToVC(colDist,rowDist,distOwner,redundantRoot);
            ++sendCounts[owners[k]];
        }
    }

    // Pack the data
    // =============
    vector<int> sendOffs;
    Scan(sendCounts, sendOffs);
    vector<Entry<T>> sendBuf(totalSend);
    auto offs = sendOffs;
    for(Int k=0; k<totalSend; ++k)
        sendBuf[offs[owners[k]]++] = remoteUpdates_[k];
    SwapClear(remoteUpdates_);

    // Exchange and unpack the data
    // ============================
    auto recvBuf = mpi::AllToAll(sendBuf, sendCounts, sendOffs, comm);
    Int recvBufSize = recvBuf.size();
    mpi::Broadcast(recvBufSize, redundantRoot, RedundantComm());
    recvBuf.resize(recvBufSize);
    mpi::Broadcast
    (recvBuf.data(), recvBufSize, redundantRoot, RedundantComm());
    // TODO: Make this loop faster
    for(const auto& entry : recvBuf)
        UpdateLocal(this->LocalRow(entry.i), this->LocalCol(entry.j), entry.value);
}

template <typename T, Device D>
void DM::ReservePulls(Int numPulls) const
{
    EL_DEBUG_CSE
    remotePulls_.reserve(numPulls);
}

template <typename T, Device D>
void DM::QueuePull(Int i, Int j) const EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    remotePulls_.push_back(ValueInt<Int>{i,j});
}

template <typename T, Device D>
void DM::ProcessPullQueue(T* pullBuf, bool includeViewers) const
{
    EL_DEBUG_CSE
    const auto& grid = Grid();
    const Dist colDist = ColDist();
    const Dist rowDist = RowDist();
    const int root = this->Root();
    const Int totalRecv = remotePulls_.size();

    // Compute the metadata
    // ====================
    mpi::Comm comm;
    int commSize;
    vector<int> recvCounts, owners(totalRecv);
    if (includeViewers)
    {
        comm = grid.ViewingComm();
        commSize = mpi::Size(comm);
        recvCounts.resize(commSize,0);
        for(Int k=0; k<totalRecv; ++k)
        {
            const auto& valueInt = remotePulls_[k];
            const Int i = valueInt.value;
            const Int j = valueInt.index;
            const int distOwner = this->Owner(i,j);
            const int vcOwner = grid.CoordsToVC(colDist,rowDist,distOwner,root);
            const int owner = grid.VCToViewing(vcOwner);
            owners[k] = owner;
            ++recvCounts[owner];
        }
    }
    else
    {
        if (!this->Participating())
            return;
        comm = grid.VCComm();
        commSize = mpi::Size(comm);
        recvCounts.resize(commSize,0);
        for(Int k=0; k<totalRecv; ++k)
        {
            const auto& valueInt = remotePulls_[k];
            const Int i = valueInt.value;
            const Int j = valueInt.index;
            const int distOwner = this->Owner(i,j);
            const int owner = grid.CoordsToVC(colDist,rowDist,distOwner,root);
            owners[k] = owner;
            ++recvCounts[owner];
        }
    }
    vector<int> recvOffs;
    Scan(recvCounts, recvOffs);
    vector<int> sendCounts(commSize);
    mpi::AllToAll(recvCounts.data(), 1, sendCounts.data(), 1, comm);
    vector<int> sendOffs;
    const int totalSend = Scan(sendCounts, sendOffs);

    auto offs = recvOffs;
    vector<ValueInt<Int>> recvCoords(totalRecv);
    for(Int k=0; k<totalRecv; ++k)
        recvCoords[offs[owners[k]]++] = remotePulls_[k];
    vector<ValueInt<Int>> sendCoords(totalSend);
    mpi::AllToAll
    (recvCoords.data(), recvCounts.data(), recvOffs.data(),
      sendCoords.data(), sendCounts.data(), sendOffs.data(), comm);

    // Pack the data
    // =============
    vector<T> sendBuf;
    FastResize(sendBuf, totalSend);
    for(Int k=0; k<totalSend; ++k)
    {
        const Int i = sendCoords[k].value;
        const Int j = sendCoords[k].index;
        sendBuf[k] = GetLocal(this->LocalRow(i), this->LocalCol(j));
    }

    // Exchange and unpack the data
    // ============================
    vector<T> recvBuf;
    FastResize(recvBuf, totalRecv);
    mpi::AllToAll
    (sendBuf.data(), sendCounts.data(), sendOffs.data(),
      recvBuf.data(), recvCounts.data(), recvOffs.data(), comm);
    offs = recvOffs;
    for(Int k=0; k<totalRecv; ++k)
        pullBuf[k] = recvBuf[offs[owners[k]]++];
    SwapClear(remotePulls_);
}

template <typename T, Device D>
void DM::ProcessPullQueue(vector<T>& pullVec, bool includeViewers) const
{
    EL_DEBUG_CSE
    pullVec.resize(remotePulls_.size());
    ProcessPullQueue(pullVec.data(), includeViewers);
}

// Local entry manipulation
// ------------------------

template <typename T, Device D>
T DM::GetLocal(Int iLoc, Int jLoc) const
EL_NO_RELEASE_EXCEPT
{ return matrix_.Get(iLoc,jLoc); }

template <typename T, Device D>
Base<T> DM::GetLocalRealPart(Int iLoc, Int jLoc) const
EL_NO_RELEASE_EXCEPT
{ return matrix_.GetRealPart(iLoc,jLoc); }

template <typename T, Device D>
Base<T> DM::GetLocalImagPart(Int iLoc, Int jLoc) const
EL_NO_RELEASE_EXCEPT
{ return matrix_.GetImagPart(iLoc,jLoc); }

template <typename T, Device D>
void DM::SetLocal(Int iLoc, Int jLoc, T alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.Set(iLoc,jLoc,alpha); }

template <typename T, Device D>
void DM::SetLocal(const Entry<T>& localEntry)
EL_NO_RELEASE_EXCEPT
{ SetLocal(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void
DM::SetLocalRealPart(Int iLoc, Int jLoc, Base<T> alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.SetRealPart(iLoc,jLoc,alpha); }

template <typename T, Device D>
void
DM::SetLocalRealPart(const Entry<Base<T>>& localEntry)
EL_NO_RELEASE_EXCEPT
{ SetLocalRealPart(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void DM::SetLocalImagPart
(Int iLoc, Int jLoc, Base<T> alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.SetImagPart(iLoc,jLoc,alpha); }

template <typename T, Device D>
void DM::SetLocalImagPart
(const Entry<Base<T>>& localEntry)
EL_NO_RELEASE_EXCEPT
{ SetLocalImagPart(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void
DM::UpdateLocal(Int iLoc, Int jLoc, T alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.Update(iLoc,jLoc,alpha); }

template <typename T, Device D>
void
DM::UpdateLocal(const Entry<T>& localEntry)
EL_NO_RELEASE_EXCEPT
{ UpdateLocal(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void
DM::UpdateLocalRealPart
(Int iLoc, Int jLoc, Base<T> alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.UpdateRealPart(iLoc,jLoc,alpha); }

template <typename T, Device D>
void
DM::UpdateLocalRealPart(const Entry<Base<T>>& localEntry)
EL_NO_RELEASE_EXCEPT
{ UpdateLocalRealPart(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void DM::UpdateLocalImagPart
(Int iLoc, Int jLoc, Base<T> alpha)
EL_NO_RELEASE_EXCEPT
{ matrix_.UpdateImagPart(iLoc,jLoc,alpha); }

template <typename T, Device D>
void DM::UpdateLocalImagPart
(const Entry<Base<T>>& localEntry)
EL_NO_RELEASE_EXCEPT
{ UpdateLocalImagPart(localEntry.i, localEntry.j, localEntry.value); }

template <typename T, Device D>
void
DM::MakeLocalReal(Int iLoc, Int jLoc)
EL_NO_RELEASE_EXCEPT
{ matrix_.MakeReal(iLoc, jLoc); }

template <typename T, Device D>
void
DM::ConjugateLocal(Int iLoc, Int jLoc)
EL_NO_RELEASE_EXCEPT
{ matrix_.Conjugate(iLoc, jLoc); }

template <typename T, Device D>
El::Matrix<T,D>&
DM::Matrix() EL_NO_EXCEPT
{
    return matrix_;
}

template <typename T, Device D>
El::Matrix<T,D> const&
DM::LockedMatrix() const EL_NO_EXCEPT
{
    return matrix_;
}

template <typename T, Device D>
Device DM::GetLocalDevice() const EL_NO_EXCEPT
{
    return D;
}

template <typename T, Device D>
void DM::do_empty_data_()
{
    SwapClear(remoteUpdates_);
}
} // namespace El

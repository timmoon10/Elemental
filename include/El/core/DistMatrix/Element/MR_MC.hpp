/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_DISTMATRIX_ELEMENTAL_MR_MC_HPP
#define EL_DISTMATRIX_ELEMENTAL_MR_MC_HPP

namespace El {

// Partial specialization to A[MR,MC].
//
// The columns of these distributed matrices will be distributed like
// "Matrix Rows" (MR), and the rows will be distributed like
// "Matrix Columns" (MC). Thus the columns will be distributed within
// rows of the process grid and the rows will be distributed within columns
// of the process grid.
template <typename Ring, Device Dev>
class DistMatrix<Ring,MR,MC,ELEMENT,Dev>
    : public ElementalMatrix<Ring>
{
public:
    // Typedefs
    // ========
    using absType = AbstractDistMatrix<Ring>;
    using elemType = ElementalMatrix<Ring>;
    using type = DistMatrix<Ring,MR,MC,ELEMENT,Dev>;
    using transType = DistMatrix<Ring,MC,MR,ELEMENT,Dev>;
    using diagType = DistMatrix<Ring,MD,STAR,ELEMENT,Dev>;
    using absLocalType = AbstractMatrix<Ring>;
    using localMatrixType = El::Matrix<Ring,Dev>;

    // Constructors and destructors
    // ============================

    // Create a 0 x 0 distributed matrix
    DistMatrix(const El::Grid& grid=Grid::Default(), int root=0);

    // Create a height x width distributed matrix
    DistMatrix
    (Int height, Int width, const El::Grid& grid=Grid::Default(), int root=0);

    // Create a copy of distributed matrix A
    DistMatrix(const type& A);
    DistMatrix(const absType& A);
    DistMatrix(const elemType& A);
    template<Dist colDist,Dist rowDist>
    DistMatrix(const DistMatrix<Ring,colDist,rowDist,ELEMENT,Dev>& A);
    template<Dist colDist,Dist rowDist>
    DistMatrix(const DistMatrix<Ring,colDist,rowDist,BLOCK,Dev>& A);

    // Copy from a different device
    template <Device Dev2,typename=typename std::enable_if<(Dev!=Dev2)&&(IsDeviceValidType<Ring,Dev>::value)&&(IsDeviceValidType<Ring,Dev2>::value)>::type>
    DistMatrix(DistMatrix<Ring,MR,MC,ELEMENT,Dev2> const& A);

    // Move constructor
    DistMatrix(type&& A) EL_NO_EXCEPT;

    // Destructor
    ~DistMatrix();

    type* Copy() const override;
    type* Construct
    (const El::Grid& grid, int root) const override;
    transType* ConstructTranspose
    (const El::Grid& grid, int root) const override;
    diagType* ConstructDiagonal
    (const El::Grid& grid, int root) const override;
    std::unique_ptr<absType> ConstructWithNewDevice(Device D2) const override;

    // Operator overloading
    // ====================

    // Return a view of a contiguous submatrix
    // ---------------------------------------
          type operator()(Range<Int> I, Range<Int> J);
    const type operator()(Range<Int> I, Range<Int> J) const;

    // Return a copy of a (generally non-contiguous) submatrix
    // -------------------------------------------------------
    type operator()(Range<Int> I, const vector<Int>& J) const;
    type operator()(const vector<Int>& I, Range<Int> J) const;
    type operator()(const vector<Int>& I, const vector<Int>& J) const;

    // Make a copy
    // -----------
    type& operator=(const absType& A);
    type& operator=(const elemType& A);
    type& operator=(const DistMatrix<Ring,MC,  MR  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,MC,  STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,MR  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,MD,  STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,MD  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,MR,  MC  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,MR,  STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,MC  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,VC,  STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,VC  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,VR,  STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,VR  ,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,STAR,STAR,ELEMENT,Dev>& A);
    type& operator=(const DistMatrix<Ring,CIRC,CIRC,ELEMENT,Dev>& A);
    template<Dist colDist,Dist rowDist>
    type& operator=(const DistMatrix<Ring,colDist,rowDist,BLOCK,Dev>& A);

    template <Device Dev2,typename=typename std::enable_if<(Dev!=Dev2)&&(IsDeviceValidType<Ring,Dev>::value)&&(IsDeviceValidType<Ring,Dev2>::value)>::type>
    type& operator=(DistMatrix<Ring,MR,MC,ELEMENT,Dev2> const& A);

    // Move assignment
    // ---------------
    type& operator=(type&& A);

    // Rescaling
    // ---------
    const type& operator*=(Ring alpha);

    // Addition/subtraction
    // --------------------
    const type& operator+=(const elemType& A);
    const type& operator+=(const absType& A);
    const type& operator-=(const elemType& A);
    const type& operator-=(const absType& A);

    // Basic queries
    // =============
    Dist ColDist()             const EL_NO_EXCEPT override;
    Dist RowDist()             const EL_NO_EXCEPT override;
    Dist PartialColDist()      const EL_NO_EXCEPT override;
    Dist PartialRowDist()      const EL_NO_EXCEPT override;
    Dist PartialUnionColDist() const EL_NO_EXCEPT override;
    Dist PartialUnionRowDist() const EL_NO_EXCEPT override;
    Dist CollectedColDist()    const EL_NO_EXCEPT override;
    Dist CollectedRowDist()    const EL_NO_EXCEPT override;

    mpi::Comm DistComm()            const EL_NO_EXCEPT override;
    mpi::Comm CrossComm()           const EL_NO_EXCEPT override;
    mpi::Comm RedundantComm()       const EL_NO_EXCEPT override;
    mpi::Comm ColComm()             const EL_NO_EXCEPT override;
    mpi::Comm RowComm()             const EL_NO_EXCEPT override;
    mpi::Comm PartialColComm()      const EL_NO_EXCEPT override;
    mpi::Comm PartialRowComm()      const EL_NO_EXCEPT override;
    mpi::Comm PartialUnionColComm() const EL_NO_EXCEPT override;
    mpi::Comm PartialUnionRowComm() const EL_NO_EXCEPT override;

    int DistSize()              const EL_NO_EXCEPT override;
    int CrossSize()             const EL_NO_EXCEPT override;
    int RedundantSize()         const EL_NO_EXCEPT override;
    int ColStride()             const EL_NO_EXCEPT override;
    int RowStride()             const EL_NO_EXCEPT override;
    int PartialColStride()      const EL_NO_EXCEPT override;
    int PartialRowStride()      const EL_NO_EXCEPT override;
    int PartialUnionColStride() const EL_NO_EXCEPT override;
    int PartialUnionRowStride() const EL_NO_EXCEPT override;

    int DistRank()            const EL_NO_EXCEPT override;
    int CrossRank()           const EL_NO_EXCEPT override;
    int RedundantRank()       const EL_NO_EXCEPT override;
    int ColRank()             const EL_NO_EXCEPT override;
    int RowRank()             const EL_NO_EXCEPT override;
    int PartialColRank()      const EL_NO_EXCEPT override;
    int PartialRowRank()      const EL_NO_EXCEPT override;
    int PartialUnionColRank() const EL_NO_EXCEPT override;
    int PartialUnionRowRank() const EL_NO_EXCEPT override;

    // Single-entry manipulation
    // =========================

    // Global entry manipulation
    // -------------------------
    // NOTE: Local entry manipulation is often much faster and should be
    //       preferred in most circumstances where performance matters.

    Ring Get(Int i, Int j) const EL_NO_RELEASE_EXCEPT override;

    Base<Ring> GetRealPart(Int i, Int j) const EL_NO_RELEASE_EXCEPT override;
    Base<Ring> GetImagPart(Int i, Int j) const EL_NO_RELEASE_EXCEPT override;

    void Set(Int i, Int j, Ring alpha) EL_NO_RELEASE_EXCEPT override;
    void Set(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT override;

    void SetRealPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT override;
    void SetImagPart (Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT override;

    void SetRealPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT override;
    void SetImagPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT override;

    void Update(Int i, Int j, Ring alpha) EL_NO_RELEASE_EXCEPT override;
    void Update(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT override;

    void UpdateRealPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT override;
    void UpdateImagPart(Int i, Int j, Base<Ring> alpha) EL_NO_RELEASE_EXCEPT override;

    void UpdateRealPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT override;
    void UpdateImagPart(const Entry<Base<Ring>>& entry) EL_NO_RELEASE_EXCEPT override;

    void MakeReal(Int i, Int j) EL_NO_RELEASE_EXCEPT override;
    void Conjugate(Int i, Int j) EL_NO_RELEASE_EXCEPT override;

    // Batch updating of remote entries
    // ---------------------------------
    void Reserve(Int numRemoteEntries) override;
    void QueueUpdate(const Entry<Ring>& entry) EL_NO_RELEASE_EXCEPT override;
    void QueueUpdate(Int i, Int j, Ring value) EL_NO_RELEASE_EXCEPT override;
    void ProcessQueues(bool includeViewers=true) override;

    // Batch extraction of remote entries
    // ----------------------------------
    void ReservePulls(Int numPulls) const override;
    void QueuePull(Int i, Int j) const EL_NO_RELEASE_EXCEPT override;
    void ProcessPullQueue(Ring* pullBuf, bool includeViewers=true) const override;
    void ProcessPullQueue(
        vector<Ring>& pullBuf, bool includeViewers=true) const override;

    // Local entry manipulation
    // ------------------------
    // NOTE: Clearly each of the following routines could instead be performed
    //       via composing [Locked]Matrix() with the corresponding local
    //       routine, but a large amount of code might need to change if
    //       these were removed.

    Ring GetLocal(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT override;
    Base<Ring> GetLocalRealPart(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT override;
    Base<Ring> GetLocalImagPart(Int iLoc, Int jLoc) const EL_NO_RELEASE_EXCEPT override;

    void SetLocal(Int iLoc, Int jLoc, Ring alpha) EL_NO_RELEASE_EXCEPT override;
    void SetLocal(Entry<Ring> const& localEntry) EL_NO_RELEASE_EXCEPT override;

    void SetLocalRealPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT override;
    void SetLocalImagPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT override;

    void SetLocalRealPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT override;
    void SetLocalImagPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT override;

    void UpdateLocal
    (Int iLoc, Int jLoc, Ring alpha) EL_NO_RELEASE_EXCEPT override;
    void UpdateLocal
    (const Entry<Ring>& localEntry) EL_NO_RELEASE_EXCEPT override;

    void UpdateLocalRealPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT override;
    void UpdateLocalImagPart(Int iLoc, Int jLoc, Base<Ring> alpha)
        EL_NO_RELEASE_EXCEPT override;

    void UpdateLocalRealPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT override;
    void UpdateLocalImagPart(const Entry<Base<Ring>>& localEntry)
        EL_NO_RELEASE_EXCEPT override;

    void MakeLocalReal(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT override;
    void ConjugateLocal(Int iLoc, Int jLoc) EL_NO_RELEASE_EXCEPT override;

    localMatrixType& Matrix() override;
    localMatrixType const& LockedMatrix() const override;

    Device GetLocalDevice() const EL_NO_EXCEPT override;

private:

    void do_empty_data_() override;

    template<typename S,Dist U,Dist V,DistWrap wrap,Device D>
    friend class DistMatrix;

private:

    // The node-local portion of the matrix
   localMatrixType matrix_ = localMatrixType{};

    // Remote updates
    // --------------
    // NOTE: Using ValueInt<Int> is somewhat of a hack; it would be nice to
    //       have a pair of integers as its own data structure that does not
    //       require separate MPI wrappers from ValueInt<Int>
    mutable vector<ValueInt<Int>> remotePulls_;

    // Remote updates
    // --------------
    vector<Entry<Ring>> remoteUpdates_;
};

} // namespace El

#endif // ifndef EL_DISTMATRIX_ELEMENTAL_MR_MC_HPP

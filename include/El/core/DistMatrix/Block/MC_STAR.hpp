/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_DISTMATRIX_BLOCKCYCLIC_MC_STAR_HPP
#define EL_DISTMATRIX_BLOCKCYCLIC_MC_STAR_HPP

namespace El {

// Partial specialization to A[MC,* ].
//
// The rows of these distributed matrices will be replicated on all
// processes (*), and the columns will be distributed like "Matrix Columns"
// (MC). Thus the columns will be distributed among columns of the process
// grid.
template <typename Ring, Device Dev>
class DistMatrix<Ring,MC,STAR,BLOCK,Dev>
    : public BlockMatrix<Ring>
{
public:
    using absType = AbstractDistMatrix<Ring>;
    using blockType = BlockMatrix<Ring>;
    using type = DistMatrix<Ring,MC,STAR,BLOCK,Dev>;
    using transType = DistMatrix<Ring,STAR,MC,BLOCK,Dev>;
    using diagType = DistMatrix<Ring,MC,STAR,BLOCK,Dev>;
    using absLocalType = AbstractMatrix<Ring>;
    using localMatrixType = El::Matrix<Ring,Dev>;

    // Constructors and destructors
    // ============================

    // Create a 0 x 0 distributed matrix with default (and unpinned) block size
    DistMatrix( const El::Grid& grid=Grid::Default(), int root=0 );

    // Create a 0 x 0 distributed matrix with fixed block size
    DistMatrix
    ( const El::Grid& grid, Int blockHeight, Int blockWidth, int root=0 );

    // Create a height x width distributed matrix with default block size
    DistMatrix
    ( Int height, Int width, const El::Grid& grid=Grid::Default(), int root=0 );

    // Create a height x width distributed matrix with fixed block size
    DistMatrix
    ( Int height, Int width, const El::Grid& grid,
      Int blockHeight, Int blockWidth, int root=0 );

    // Create a copy of distributed matrix A (redistributing if necessary)
    DistMatrix( const type& A );
    DistMatrix( const absType& A );
    DistMatrix( const blockType& A );
    template<Dist colDist,Dist rowDist>
    DistMatrix( const DistMatrix<Ring,colDist,rowDist,BLOCK,Dev>& A );
    template<Dist colDist,Dist rowDist>
    DistMatrix( const DistMatrix<Ring,colDist,rowDist,ELEMENT,Dev>& A );

    // Move constructor
    DistMatrix( type&& A ) EL_NO_EXCEPT;

    // Destructor
    ~DistMatrix();

    type* Copy() const override;
    type* Construct
    ( const El::Grid& grid, int root ) const override;
    transType* ConstructTranspose
    ( const El::Grid& grid, int root ) const override;
    diagType* ConstructDiagonal
    ( const El::Grid& grid, int root ) const override;

    // Operator overloading
    // ====================

    // Return a view of a contiguous submatrix
    // ---------------------------------------
          type operator()( Range<Int> I, Range<Int> J );
    const type operator()( Range<Int> I, Range<Int> J ) const;

    // Return a copy of a (generally non-contiguous) submatrix
    // -------------------------------------------------------
    type operator()( Range<Int> I, const vector<Int>& J ) const;
    type operator()( const vector<Int>& I, Range<Int> J ) const;
    type operator()( const vector<Int>& I, const vector<Int>& J ) const;

    // Make a copy
    // -----------
    type& operator=( const absType& A );
    type& operator=( const blockType& A );
    type& operator=( const DistMatrix<Ring,MC,  MR  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,MC,  STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,MR  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,MD,  STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,MD  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,MR,  MC  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,MR,  STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,MC  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,VC,  STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,VC  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,VR,  STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,VR  ,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,STAR,STAR,BLOCK,Dev>& A );
    type& operator=( const DistMatrix<Ring,CIRC,CIRC,BLOCK,Dev>& A );
    template<Dist colDist,Dist rowDist>
    type& operator=( const DistMatrix<Ring,colDist,rowDist,ELEMENT,Dev>& A );

    // Move assignment
    // ---------------
    type& operator=( type&& A );

    // Rescaling
    // ---------
    const type& operator*=( Ring alpha );

    // Addition/subtraction
    // --------------------
    const type& operator+=( const blockType& A );
    const type& operator+=( const absType& A );
    const type& operator-=( const blockType& A );
    const type& operator-=( const absType& A );

    // Basic queries
    // =============
    Dist ColDist()             const override EL_NO_EXCEPT;
    Dist RowDist()             const override EL_NO_EXCEPT;
    Dist PartialColDist()      const override EL_NO_EXCEPT;
    Dist PartialRowDist()      const override EL_NO_EXCEPT;
    Dist PartialUnionColDist() const override EL_NO_EXCEPT;
    Dist PartialUnionRowDist() const override EL_NO_EXCEPT;
    Dist CollectedColDist()    const override EL_NO_EXCEPT;
    Dist CollectedRowDist()    const override EL_NO_EXCEPT;

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

#endif // ifndef EL_DISTMATRIX_BLOCKCYCLIC_MC_STAR_HPP

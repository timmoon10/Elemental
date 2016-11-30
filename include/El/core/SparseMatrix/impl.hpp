/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2012 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin.
   All rights reserved.

   Copyright (c) 2013 Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright (c) 2014 Jack Poulson and The Georgia Institute of Technology.
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_SPARSEMATRIX_IMPL_HPP
#define EL_SPARSEMATRIX_IMPL_HPP

#include <El/blas_like/level1/Axpy.hpp>
#include <El/blas_like/level1/Scale.hpp>

namespace El {

// Constructors and destructors
// ============================

template<typename T>
SparseMatrix<T>::SparseMatrix()
: numRows_(0), numCols_(0),
  blockHeight_(0), blockWidth_(0)
{
    DEBUG_CSE
    blockOffsets_.resize( 1, 0 );
    rowOffsets_.resize( 1, 0 );
}

template<typename T>
SparseMatrix<T>::SparseMatrix
( Int height, Int width, Int numBlockRows, Int numBlockCols )
: numRows_(height), numCols_(width),
  numBlockRows_(numBlockRows), numBlockCols_(numBlockCols),
  blockHeight_((height+numBlockRows_-1)/numBlockRows_),
  blockWidth_((width+numBlockCols_-1)/numBlockCols_)
{
    DEBUG_CSE
    blockOffsets_.resize( numBlockRows_*numBlockCols_+1, 0 );
    rowOffsets_.resize( numBlockRows_*numBlockCols_*blockHeight_+1, 0 );
}

template<typename T>
SparseMatrix<T>::SparseMatrix( const SparseMatrix<T>& A )
{
    DEBUG_CSE
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct sparse matrix with itself");
}

template<typename T>
SparseMatrix<T>::SparseMatrix( const DistSparseMatrix<T>& A )
{
    // DEBUG_CSE
    // *this = A;
    // TODO: implement
    LogicError("Not yet implemented");
}

template<typename T>
SparseMatrix<T>::~SparseMatrix() { }

// Assignment and reconfiguration
// ==============================

// Change the size of the matrix
// -----------------------------
template<typename T>
void SparseMatrix<T>::Empty( bool clearMemory )
{
    DEBUG_CSE

    // Empty matrix has no rows, columns, or blocks
    numRows_ = 0;
    numCols_ = 0;    
    frozenSparsity_ = false;
    consistent_ = true;
    numBlockRows_ = 0;
    numBlockCols_ = 0;
    blockHeight_ = 0;
    blockWidth_ = 0;
    
    // Clear arrays
    if( clearMemory )
    {
        SwapClear( rows_ );
        SwapClear( cols_ );
        SwapClear( vals_ );
        SwapClear( blocks_ );
        SwapClear( blockOffsets_ );
        SwapClear( rowOffsets_ );
        SwapClear( markedForRemoval_ );
    }
    else
    {
        rows_.clear();
        cols_.clear();
        vals_.clear();
        blocks_.clear();
        markedForRemoval_.clear();
    }
    blockOffsets_.resize( 1, 0 );
    rowOffsets_.resize( 1, 0 );
}

template<typename T>
void SparseMatrix<T>::Resize
( Int height, Int width, Int numBlockRows, Int numBlockCols )
{
    DEBUG_CSE
    if( Height() == height && Width() == width )
        return;
    numRows_ = height;
    numCols_ = width;
    frozenSparsity_ = false;
    consistent_ = true;
    numBlockRows_ = numBlockRows;
    numBlockCols_ = numBlockCols;
    blockHeight_ = (height+numBlockRows_-1)/numBlockRows_;
    blockWidth_ = (width+numBlockCols_-1)/numBlockCols_;
    rows_.clear();
    cols_.clear();
    vals_.clear();
    blocks_.clear();
    blockOffsets_.resize( numBlockRows_*numBlockCols_+1, 0 );
    rowOffsets_.resize( numBlockRows_*numBlockCols_*blockHeight_+1, 0 );
    markedForRemoval_.clear();
}

// Assembly
// --------
template<typename T>
void SparseMatrix<T>::Reserve( Int numEntries )
{
    const Int currSize = vals_.size();
    rows_.reserve( currSize+numEntries );
    cols_.reserve( currSize+numEntries );
    vals_.reserve( currSize+numEntries );
    blocks_.reserve( currSize+numEntries );
}

template<typename T>
void SparseMatrix<T>::FreezeSparsity() EL_NO_EXCEPT
{ frozenSparsity_ = true; }
template<typename T>
void SparseMatrix<T>::UnfreezeSparsity() EL_NO_EXCEPT
{ frozenSparsity_ = false; }
template<typename T>
bool SparseMatrix<T>::FrozenSparsity() const EL_NO_EXCEPT
{ return frozenSparsity_; }

template<typename T>
void SparseMatrix<T>::Update( Int row, Int col, T value )
{
    DEBUG_CSE
    QueueUpdate( row, col, value );
    ProcessQueues();
}

template<typename T>
void SparseMatrix<T>::Update( const Entry<T>& entry )
{ Update( entry.i, entry.j, entry.value ); }

template<typename T>
void SparseMatrix<T>::Zero( Int row, Int col )
{
    DEBUG_CSE
    QueueZero( row, col );
    ProcessQueues();
}

template<typename T>
void SparseMatrix<T>::QueueUpdate( Int row, Int col, T value )
EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    if( row == END ) row = numRows_ - 1;
    if( col == END ) col = numCols_ - 1;

    // Check if entry is valid (debug)
    DEBUG_ONLY(
      if( row < 0 || row >= numRows_ )
          LogicError
          ("Row out of bounds: ",row," not in [0,",numRows_,")");
      if( col < 0 || col >= numCols_ )
          LogicError
          ("Column out of bounds: ",col," not in [0,",numCols_,")");
    )

    // Update entry if sparsity is frozen
    if( FrozenSparsity() )
    {
        const Int offset = Offset( row, col );
        DEBUG_ONLY(
          if( rows_[offset] != row || cols_[offset] != col )
              LogicError
              ("Cannot update entry: (",row,",",col,") is not in sparsity pattern");
        )
        vals_[offset] += value;
    }

    // Add entry to end of COO list if sparsity is not frozen
    else
    {
        DEBUG_ONLY(
          if( NumEdges() == Capacity() )
              cerr << "WARNING: Pushing back without first reserving space" << endl;
        )
        rows_.push_back( row );
        cols_.push_back( col );
        vals_.push_back( value );
        const Int blockIndex
          = (row/blockHeight_) + (col/blockWidth_)*numBlockRows_;
        blocks_.push_back( blockIndex );
        consistent_ = false;
    }
}

template<typename T>
void SparseMatrix<T>::QueueUpdate( const Entry<T>& entry )
EL_NO_RELEASE_EXCEPT
{ QueueUpdate( entry.i, entry.j, entry.value ); }

template<typename T>
void SparseMatrix<T>::QueueZero( Int row, Int col )
EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    if( row == END ) row = numRows_ - 1;
    if( col == END ) col = numCols_ - 1;

    // Check if entry is valid (debug)
    DEBUG_ONLY(
      if( row < 0 || row >= numRows_ )
          LogicError
          ("Row out of bounds: ",row," not in [0,",numRows_,")");
      if( col < 0 || col >= numCols_ )
          LogicError
          ("Column out of bounds: ",col," not in [0,",numCols_,")");
    )

    // Zero out entry if sparsity is frozen
    if( FrozenSparsity() )
    {
        const Int offset = Offset( row, col );
        DEBUG_ONLY(
          if( rows_[offset] != row || cols_[offset] != col )
              LogicError
              ("Cannot update entry: (",row,",",col,") is not in sparsity pattern");
        )
        vals_[offset] = 0;
    }

    // Mark entry for removal if sparsity is not frozen
    else
    {
        markedForRemoval_.insert( pair<Int,Int>(row,col) );
        consistent_ = false;
    }
}

// Operator overloading
// ====================

// Make a copy
// -----------
template<typename T>
const SparseMatrix<T>& SparseMatrix<T>::operator=( const SparseMatrix<T>& A )
{
    DEBUG_CSE
    // graph_ = A.graph_;
    // vals_ = A.vals_;
    // TODO: implement
    LogicError("Not yet implemented");
    return *this;
}

template<typename T>
const SparseMatrix<T>&
SparseMatrix<T>::operator=( const DistSparseMatrix<T>& A )
{
    DEBUG_CSE
    mpi::Comm comm = A.Comm();
    const int commSize = mpi::Size( comm );
    if( commSize != 1 )
        LogicError("Can not yet construct from distributed sparse matrix");

    // graph_ = A.distGraph_;
    // vals_ = A.vals_;
    // TODO: implement
    LogicError("Not yet implemented");
    return *this;
}

// Make a copy of a submatrix
// --------------------------
template<typename T>
SparseMatrix<T>
SparseMatrix<T>::operator()( Range<Int> I, Range<Int> J ) const
{
    DEBUG_CSE
    SparseMatrix<T> ASub;
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

template<typename T>
SparseMatrix<T>
SparseMatrix<T>::operator()( const vector<Int>& I, Range<Int> J ) const
{
    DEBUG_CSE
    SparseMatrix<T> ASub;
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

template<typename T>
SparseMatrix<T>
SparseMatrix<T>::operator()( Range<Int> I, const vector<Int>& J ) const
{
    DEBUG_CSE
    SparseMatrix<T> ASub;
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

template<typename T>
SparseMatrix<T>
SparseMatrix<T>::operator()( const vector<Int>& I, const vector<Int>& J ) const
{
    DEBUG_CSE
    SparseMatrix<T> ASub;
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

// Rescaling
// ---------
template<typename T>
const SparseMatrix<T>& SparseMatrix<T>::operator*=( T alpha )
{
    DEBUG_CSE
    Scale( alpha, *this );
    return *this;
}

// Addition/subtraction
// --------------------
template<typename T>
const SparseMatrix<T>& SparseMatrix<T>::operator+=( const SparseMatrix<T>& A )
{
    DEBUG_CSE
    Axpy( T(1), A, *this );
    return *this;
}

template<typename T>
const SparseMatrix<T>& SparseMatrix<T>::operator-=( const SparseMatrix<T>& A )
{
    DEBUG_CSE
    Axpy( T(-1), A, *this );
    return *this;
}

// Queries
// =======

// High-level information
// ----------------------
template<typename T>
Int SparseMatrix<T>::Height() const EL_NO_EXCEPT { return numRows_; }
template<typename T>
Int SparseMatrix<T>::Width() const EL_NO_EXCEPT { return numCols_; }

template<typename T>
Int SparseMatrix<T>::NumEntries() const EL_NO_EXCEPT 
{
    DEBUG_CSE
    return vals_.size();
}

template<typename T>
Int SparseMatrix<T>::Capacity() const EL_NO_EXCEPT
{
    DEBUG_CSE

    // Get smallest capacity out of rows_, cols_, vals_, and blocks_
    Int capacity = Min( rows_.capacity(), cols_.capacity() );
    capacity = Min( capacity, vals_.capacity() );
    capacity = Min( capacity, blocks_.capacity() );
    return capacity;
}

template<typename T>
bool SparseMatrix<T>::Consistent() const EL_NO_EXCEPT
{ return consistent_; }

template<typename T>
El::Graph& SparseMatrix<T>::Graph() EL_NO_EXCEPT
{
    // TODO: implement
    LogicError("Not yet implemented");
    El::Graph g;
    return g;
}
template<typename T>
const El::Graph& SparseMatrix<T>::LockedGraph() const EL_NO_EXCEPT
{
    // TODO: implement
    LogicError("Not yet implemented");
    const El::Graph g;
    return g;
}

// Entrywise information
// ---------------------
template<typename T>
Int SparseMatrix<T>::Row( Int index ) const EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( index < 0 || index >= (Int)rows_.size() )
          LogicError("Index number ",index," out of bounds");
    )
    return rows_[index];
}

template<typename T>
Int SparseMatrix<T>::Col( Int index ) const EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( index < 0 || index >= (Int)cols_.size() )
          LogicError("Index number ",index," out of bounds");
    )
    return cols_[index];
}

template<typename T>
Int SparseMatrix<T>::RowOffset( Int row, Int blockCol ) const
EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE    
    // return graph_.SourceOffset( row );
    // TODO: implement
    LogicError("Not yet implemented");
    return 0;
}

template<typename T>
Int SparseMatrix<T>::Offset( Int row, Int col ) const EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    // return graph_.Offset( row, col );
    // TODO: implement
    LogicError("Not yet implemented");
    return 0;
}

template<typename T>
Int SparseMatrix<T>::NumConnections( Int row ) const EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    // return graph_.NumConnections( row );
    // TODO: implement
    LogicError("Not yet implemented");
    return 0;
}

template<typename T>
T SparseMatrix<T>::Value( Int index ) const EL_NO_RELEASE_EXCEPT
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( index < 0 || index >= Int(vals_.size()) )
          LogicError("Index number ",index," out of bounds");
    )
    return vals_[index];
}

template< typename T>
T SparseMatrix<T>::Get( Int row, Int col) const EL_NO_RELEASE_EXCEPT
{
    if( row == END ) row = numRows_ - 1;
    if( col == END ) col = numCols_ - 1;
    Int index = Offset( row, col );
    if( Row(index) != row || Col(index) != col )
        return T(0);
    else
        return Value( index );
}

template< typename T>
void SparseMatrix<T>::Set( Int row, Int col, T val) EL_NO_RELEASE_EXCEPT
{
    if( row == END ) row = numRows_ - 1;
    if( col == END ) col = numCols_ - 1;
    Int index = Offset( row, col );
    if( Row(index) == row && Col(index) == col )
    {
        vals_[index] = val;
    }
    else
    {
        QueueUpdate( row, col, val );
        ProcessQueues();
    }
}

template<typename T>
Int* SparseMatrix<T>::RowBuffer() EL_NO_EXCEPT
{ return rows_.data(); }
template<typename T>
Int* SparseMatrix<T>::ColumnBuffer() EL_NO_EXCEPT
{ return cols_.data(); }
template<typename T>
Int* SparseMatrix<T>::RowOffsetBuffer() EL_NO_EXCEPT
{ return rowOffsets_.data(); }
template<typename T>
Int* SparseMatrix<T>::BlockOffsetBuffer() EL_NO_EXCEPT
{ return blockOffsets_.data(); }
template<typename T>
T* SparseMatrix<T>::ValueBuffer() EL_NO_EXCEPT
{ return vals_.data(); }

template<typename T>
const Int* SparseMatrix<T>::LockedRowBuffer() const EL_NO_EXCEPT
{ return rows_.data(); }
template<typename T>
const Int* SparseMatrix<T>::LockedColumnBuffer() const EL_NO_EXCEPT
{ return cols_.data(); }
template<typename T>
const Int* SparseMatrix<T>::LockedRowOffsetBuffer() const EL_NO_EXCEPT
{ return rowOffsets_.data(); }
template<typename T>
const Int* SparseMatrix<T>::LockedBlockOffsetBuffer() const EL_NO_EXCEPT
{ return blockOffsets_.data(); }
template<typename T>
const T* SparseMatrix<T>::LockedValueBuffer() const EL_NO_EXCEPT
{ return vals_.data(); }

template<typename T>
void SparseMatrix<T>::ForceNumEntries( Int numEntries )
{
    DEBUG_CSE
    // graph_.ForceNumEdges( numEntries );
    // vals_.resize( numEntries );
    // TODO: implement
    LogicError("Not yet implemented");
}

template<typename T>
void SparseMatrix<T>::ForceConsistency( bool consistent ) EL_NO_EXCEPT
{ consistent_ = consistent; }

// Auxiliary routines
// ==================

template<typename T>
void SparseMatrix<T>::ProcessQueues()
{
    DEBUG_CSE

    // TODO: implement
    LogicError("Not yet implemented");

#if 0
    DEBUG_ONLY(
      if( graph_.sources_.size() != graph_.targets_.size() ||
          graph_.targets_.size() != vals_.size() )
          LogicError("Inconsistent sparse matrix buffer sizes");
    )
    if( graph_.consistent_ )
        return;

    Int numRemoved = 0;
    const Int numEntries = vals_.size();
    vector<Entry<T>> entries( numEntries );
    if( graph_.markedForRemoval_.size() != 0 )
    {
        for( Int s=0; s<numEntries; ++s )
        {
            pair<Int,Int> candidate(graph_.sources_[s],graph_.targets_[s]);
            if( graph_.markedForRemoval_.find(candidate) ==
                graph_.markedForRemoval_.end() )
            {
                entries[s-numRemoved].i = graph_.sources_[s];
                entries[s-numRemoved].j = graph_.targets_[s];
                entries[s-numRemoved].value = vals_[s];
            }
            else
            {
                ++numRemoved;
            }
        }
        graph_.markedForRemoval_.clear();
        entries.resize( numEntries-numRemoved );
    }
    else
    {
        for( Int s=0; s<numEntries; ++s )
            entries[s] =
              Entry<T>{graph_.sources_[s],graph_.targets_[s],vals_[s]};
    }
    CompareEntriesFunctor comparer;
    std::sort( entries.begin(), entries.end(), comparer );
    const Int numSorted = entries.size();

    // Compress out duplicates
    Int lastUnique=0;
    for( Int s=1; s<numSorted; ++s )
    {
        if( entries[s].i != entries[lastUnique].i ||
            entries[s].j != entries[lastUnique].j )
            entries[++lastUnique] = entries[s];
        else
            entries[lastUnique].value += entries[s].value;
    }
    const Int numUnique = lastUnique+1;
    entries.resize( numUnique );

    graph_.sources_.resize( numUnique );
    graph_.targets_.resize( numUnique );
    vals_.resize( numUnique );
    for( Int s=0; s<numUnique; ++s )
    {
        graph_.sources_[s] = entries[s].i;
        graph_.targets_[s] = entries[s].j;
        vals_[s] = entries[s].value;
    }

    graph_.ComputeSourceOffsets();
    graph_.consistent_ = true;
#endif
}

template<typename T>
void SparseMatrix<T>::AssertConsistent() const
{
    if( !consistent_ )
        LogicError("Sparse matrix was not consistent; run ProcessQueues()");
}

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) EL_EXTERN template class SparseMatrix<T>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_SPARSEMATRIX_IMPL_HPP

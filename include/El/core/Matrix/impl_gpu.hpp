/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_IMPL_GPU_HPP_
#define EL_MATRIX_IMPL_GPU_HPP_

namespace El
{
//
// Public routines
//
// Constructors and destructors
//

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix() { }

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Int height, Int width)
    : height_(height), width_(width), leadingDimension_(Max(height,1))
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidDimensions(height, width));

    memory_.Require(leadingDimension_ * width);
    data_ = memory_.Buffer();
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Int height, Int width, Int leadingDimension)
    : height_(height), width_(width), leadingDimension_(leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidDimensions(height, width, leadingDimension));
    memory_.Require(leadingDimension*width);
    data_ = memory_.Buffer();
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix
(Int height, Int width, DevicePtr<const Ring> buffer, Int leadingDimension)
    : viewType_(LOCKED_VIEW),
      height_(height), width_(width), leadingDimension_(leadingDimension),
      data_(const_cast<Ring*>(buffer))
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidDimensions(height, width, leadingDimension));
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix
(Int height, Int width, DevicePtr<Ring> buffer, Int leadingDimension)
    : viewType_(VIEW),
      height_(height), width_(width), leadingDimension_(leadingDimension),
      data_(buffer)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidDimensions(height, width, leadingDimension));
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Matrix<Ring, Device::GPU> const& A)
{
    // FIXME (trb): This is idiomatically backward. Assignment in
    // terms of copy!
    EL_DEBUG_CSE;
    if (&A != this)
        *this = A;
    else
        LogicError("You just tried to construct a Matrix with itself!");
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Matrix<Ring, Device::CPU> const& A)
    : Matrix{A.height_, A.width_, A.leadingDimension_}
{
    EL_DEBUG_CSE;
    auto error = cudaMemcpy2D(data_, width_*sizeof(Ring),
                              A.LockedBuffer(), A.width_*sizeof(Ring),
                              width_*sizeof(Ring), leadingDimension_,//height_*sizeof(Ring),
                              cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "Error in copy to GPU: "
            << cudaGetErrorName(error) << "\ndescription: "
            << cudaGetErrorString(error) << "\n";
        RuntimeError(oss.str());
    }
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Matrix<Ring, Device::GPU>&& A) EL_NO_EXCEPT
    : viewType_(A.viewType_),
      height_(A.height_), width_(A.width_), leadingDimension_(A.leadingDimension_),
      memory_(std::move(A.memory_)), data_(nullptr)
{ std::swap(data_, A.data_); }

template<typename Ring>
Matrix<Ring, Device::GPU>::~Matrix() { }

#if 0
// Assignment and reconfiguration
// ==============================

template<typename Ring>
void Matrix<Ring, Device::GPU>::Empty(bool freeMemory)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (FixedSize())
            LogicError("Cannot empty a fixed-size matrix");
        )
        Empty_(freeMemory);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Resize(Int height, Int width)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidDimensions(height, width);
        if (FixedSize() && (height != height_ || width != width_))
        {
            LogicError
                ("Cannot resize this matrix from ",
                 height_," x ",width_," to ",height," x ",width);
        }
        if (Viewing() && (height > height_ || width > width_))
            LogicError("Cannot increase the size of this matrix");
        )
        Resize_(height, width);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Resize(
    Int height, Int width, Int leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidDimensions(height, width, leadingDimension);
        if (FixedSize() &&
            (height != height_ || width != width_ ||
             leadingDimension != leadingDimension_))
        {
            LogicError
                ("Cannot resize this matrix from ",
                 height_," x ",width_," (",leadingDimension_,") to ",
                 height," x ",width," (",leadingDimension,")");
        }
        if (Viewing() && (height > height_ || width > width_ ||
                          leadingDimension != leadingDimension_))
            LogicError("Cannot increase the size of this matrix");
        )
        Resize_(height, width, leadingDimension);
}

#endif // 0

template<typename Ring>
void Matrix<Ring, Device::GPU>::Attach
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (FixedSize())
            LogicError("Cannot attach a new buffer to a view with fixed size");
        )
        Attach_(height, width, buffer, leadingDimension);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::LockedAttach
(Int height, Int width, const Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (FixedSize())
            LogicError("Cannot attach a new buffer to a view with fixed size");
        )
        LockedAttach_(height, width, buffer, leadingDimension);
}

#if 0

template<typename Ring>
void Matrix<Ring, Device::GPU>::Control
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (FixedSize())
            LogicError("Cannot attach a new buffer to a view with fixed size");
        )
        Control_(height, width, buffer, leadingDimension);
}

#endif // 0

// Operator overloading
// ====================

// Return a view
// -------------
template<typename Ring>
Matrix<Ring, Device::GPU>
Matrix<Ring, Device::GPU>::operator()(Range<Int> I, Range<Int> J)
{
    EL_DEBUG_CSE;
    if (this->Locked())
        return LockedView(*this, I, J);
    else
        return View(*this, I, J);
}

template<typename Ring>
const Matrix<Ring, Device::GPU>
Matrix<Ring, Device::GPU>::operator()(Range<Int> I, Range<Int> J) const
{
    EL_DEBUG_CSE;
    return LockedView(*this, I, J);
}

#if 0

// Return a (potentially non-contiguous) subset of indices
// -------------------------------------------------------
template<typename Ring>
Matrix<Ring, Device::GPU> Matrix<Ring, Device::GPU>::operator()
    (Range<Int> I, vector<Int> const& J) const
{
    EL_DEBUG_CSE;
    Matrix<Ring, Device::GPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template<typename Ring>
Matrix<Ring, Device::GPU> Matrix<Ring, Device::GPU>::operator()
    (vector<Int> const& I, Range<Int> J) const
{
    EL_DEBUG_CSE;
    Matrix<Ring, Device::GPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template<typename Ring>
Matrix<Ring, Device::GPU> Matrix<Ring, Device::GPU>::operator()
    (vector<Int> const& I, vector<Int> const& J) const
{
    EL_DEBUG_CSE;
    Matrix<Ring, Device::GPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

// Make a copy
// -----------
template<typename Ring>
Matrix<Ring, Device::GPU> const&
Matrix<Ring, Device::GPU>::operator=(Matrix<Ring, Device::GPU> const& A)
{
    EL_DEBUG_CSE;
    Copy(A, *this);
    return *this;
}
#endif // 0

// Move assignment
// ---------------
template<typename Ring>
Matrix<Ring, Device::GPU>&
Matrix<Ring, Device::GPU>::operator=(Matrix<Ring, Device::GPU>&& A)
{
    EL_DEBUG_CSE;
    if (Viewing() || A.Viewing())
    {
        operator=((Matrix<Ring, Device::GPU> const&)A);
    }
    else
    {
        memory_.ShallowSwap(A.memory_);
        std::swap(data_, A.data_);
        viewType_ = A.viewType_;
        height_ = A.height_;
        width_ = A.width_;
        leadingDimension_ = A.leadingDimension_;
    }
    return *this;
}

#if 0

// Rescaling
// ---------
template<typename Ring>
Matrix<Ring, Device::GPU> const&
Matrix<Ring, Device::GPU>::operator*=(Ring const& alpha)
{
    EL_DEBUG_CSE;
    Scale(alpha, *this);
    return *this;
}

// Addition/subtraction
// --------------------
template<typename Ring>
Matrix<Ring, Device::GPU> const&
Matrix<Ring, Device::GPU>::operator+=(Matrix<Ring, Device::GPU> const& A)
{
    EL_DEBUG_CSE;
    Axpy(Ring(1), A, *this);
    return *this;
}

template<typename Ring>
Matrix<Ring, Device::GPU> const&
Matrix<Ring, Device::GPU>::operator-=(Matrix<Ring, Device::GPU> const& A)
{
    EL_DEBUG_CSE;
    Axpy(Ring(-1), A, *this);
    return *this;
}

#endif // 0

// Basic queries
// =============

template<typename Ring>
Int Matrix<Ring, Device::GPU>::Height() const EL_NO_EXCEPT { return height_; }

template<typename Ring>
Int Matrix<Ring, Device::GPU>::Width() const EL_NO_EXCEPT { return width_; }

template<typename Ring>
Int Matrix<Ring, Device::GPU>::LDim() const EL_NO_EXCEPT
{ return leadingDimension_; }

template<typename Ring>
Int Matrix<Ring, Device::GPU>::MemorySize() const EL_NO_EXCEPT
{ return memory_.Size(); }

#if 0

template<typename Ring>
Int Matrix<Ring, Device::GPU>::DiagonalLength(Int offset) const EL_NO_EXCEPT
{ return El::DiagonalLength(height_,width_,offset); }

#endif // 0

template<typename Ring>
Ring* Matrix<Ring, Device::GPU>::Buffer() EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (Locked())
            LogicError("Cannot return non-const buffer of locked Matrix");
        )
        return data_;
}

template<typename Ring>
Ring* Matrix<Ring, Device::GPU>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (Locked())
            LogicError("Cannot return non-const buffer of locked Matrix");
        )
        if (data_ == nullptr)
            return nullptr;
    if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    return &data_[i+j*leadingDimension_];
}

template<typename Ring>
const Ring* Matrix<Ring, Device::GPU>::LockedBuffer() const EL_NO_EXCEPT
{ return data_; }

template<typename Ring>
const Ring*
Matrix<Ring, Device::GPU>::LockedBuffer(Int i, Int j) const EL_NO_EXCEPT
{
    EL_DEBUG_CSE;
    if (data_ == nullptr)
        return nullptr;
    if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    return &data_[i+j*leadingDimension_];
}


template<typename Ring>
bool Matrix<Ring, Device::GPU>::Viewing() const EL_NO_EXCEPT
{ return IsViewing(viewType_); }

template<typename Ring>
bool Matrix<Ring, Device::GPU>::FixedSize() const EL_NO_EXCEPT
{ return IsFixedSize(viewType_); }

template<typename Ring>
bool Matrix<Ring, Device::GPU>::Locked() const EL_NO_EXCEPT
{ return IsLocked(viewType_); }


#if 0
template<typename Ring>
void Matrix<Ring, Device::GPU>::FixSize() EL_NO_EXCEPT
{
    // A view is marked as fixed if its second bit is nonzero
    // (and OWNER_FIXED is zero except in its second bit).
    viewType_ = static_cast<El::ViewType>(viewType_ | OWNER_FIXED);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetViewType(El::ViewType viewType) EL_NO_EXCEPT
{ viewType_ = viewType; }

template<typename Ring>
El::ViewType Matrix<Ring, Device::GPU>::ViewType() const EL_NO_EXCEPT
{ return viewType_; }

// Single-entry manipulation
// =========================

template<typename Ring>
Ring Matrix<Ring, Device::GPU>::Get(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidEntry(i, j))
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    return CRef(i, j);
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::GPU>::GetRealPart(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidEntry(i, j))
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    return El::RealPart(CRef(i, j));
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::GPU>::GetImagPart(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidEntry(i, j))
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    return El::ImagPart(CRef(i, j));
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Set(Int i, Int j, Ring const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    Ref(i, j) = alpha;
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Set(Entry<Ring> const& entry)
    EL_NO_RELEASE_EXCEPT
{ Set(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::SetRealPart(
    Int i, Int j, Base<Ring> const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    El::SetRealPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetRealPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ SetRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::SetImagPart(Int i, Int j, Base<Ring> const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    El::SetImagPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetImagPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ SetImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::GPU>::Update(Int i, Int j, Ring const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    Ref(i, j) += alpha;
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Update(Entry<Ring> const& entry)
    EL_NO_RELEASE_EXCEPT
{ Update(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::UpdateRealPart(Int i, Int j, Base<Ring> const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    El::UpdateRealPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::UpdateRealPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ UpdateRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::UpdateImagPart(Int i, Int j, Base<Ring> const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    El::UpdateImagPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::UpdateImagPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ UpdateImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::GPU>::MakeReal(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        Set(i, j, GetRealPart(i,j));
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Conjugate(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        Set(i, j, El::Conj(Get(i,j)));
}

// Private routines
// ################

// Exchange metadata with another matrix
// =====================================
template<typename Ring>
void Matrix<Ring, Device::GPU>::ShallowSwap(Matrix<Ring, Device::GPU>& A)
{
    memory_.ShallowSwap(A.memory_);
    std::swap(data_, A.data_);
    std::swap(viewType_, A.viewType_);
    std::swap(height_, A.height_);
    std::swap(width_, A.width_);
    std::swap(leadingDimension_, A.leadingDimension_);
}

// Reconfigure without error-checking
// ==================================

template<typename Ring>
void Matrix<Ring, Device::GPU>::Empty_(bool freeMemory)
{
    if (freeMemory)
        memory_.Empty();
    height_ = 0;
    width_ = 0;
    leadingDimension_ = 1;
    data_ = nullptr;
    viewType_ = static_cast<El::ViewType>(viewType_ & ~LOCKED_VIEW);
}

#endif // 0

template<typename Ring>
void Matrix<Ring, Device::GPU>::Attach_(
    Int height, Int width, Ring* buffer, Int leadingDimension)
{
    height_ = height;
    width_ = width;
    leadingDimension_ = leadingDimension;
    data_ = buffer;
    viewType_ = static_cast<El::ViewType>((viewType_ & ~LOCKED_OWNER) | VIEW);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::LockedAttach_(
    Int height, Int width, const Ring* buffer, Int leadingDimension)
{
    height_ = height;
    width_ = width;
    leadingDimension_ = leadingDimension;
    data_ = const_cast<Ring*>(buffer);
    viewType_ = static_cast<El::ViewType>(viewType_ | LOCKED_VIEW);
}

#if 0

template<typename Ring>
void Matrix<Ring, Device::GPU>::Control_
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    height_ = height;
    width_ = width;
    leadingDimension_ = leadingDimension;
    data_ = buffer;
    viewType_ = static_cast<El::ViewType>(viewType_ & ~LOCKED_VIEW);
}

// Return a reference to a single entry without error-checking
// ===========================================================
template<typename Ring>
Ring const& Matrix<Ring, Device::GPU>::CRef(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*leadingDimension_];
}

template<typename Ring>
Ring const& Matrix<Ring, Device::GPU>::operator()(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(AssertValidEntry(i, j))
        return data_[i+j*leadingDimension_];
}

template<typename Ring>
Ring& Matrix<Ring, Device::GPU>::Ref(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*leadingDimension_];
}

template<typename Ring>
Ring& Matrix<Ring, Device::GPU>::operator()(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        AssertValidEntry(i, j);
        if (Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        return data_[i+j*leadingDimension_];
}

// Assertions
// ==========

template<typename Ring>
void
Matrix<Ring, Device::GPU>::AssertValidDimensions(Int height, Int width) const
{
    EL_DEBUG_CSE;
    if (height < 0 || width < 0)
        LogicError("Height and width must be non-negative");
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::AssertValidDimensions
(Int height, Int width, Int leadingDimension) const
{
    EL_DEBUG_CSE;
    AssertValidDimensions(height, width);
    if (leadingDimension < height)
        LogicError("Leading dimension must be no less than height");
    if (leadingDimension == 0)
        LogicError("Leading dimension cannot be zero (for BLAS compatibility)");
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::AssertValidEntry(Int i, Int j) const
{
    EL_DEBUG_CSE;
    if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    if (i < 0 || j < 0)
        LogicError("Indices must be non-negative");
    if (i >= Height() || j >= Width())
        LogicError
            ("Out of bounds: (",i,",",j,") of ",Height()," x ",Width()," Matrix");
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Resize_(Int height, Int width)
{
    // Only change the leadingDimension when necessary.
    // Simply 'shrink' our view if possible.
    //
    // Note that the matrix is, by default, initialized as 0 x 0 with a
    // leading dimension of 1, so any resize to a nonzero number of entries
    // will trigger a reallocation if we use the following logic.
    //
    // TODO(poulson): Avoid reallocation if height*width == 0?
    const bool reallocate = height > leadingDimension_ || width > width_;
    height_ = height;
    width_ = width;
    if (reallocate)
    {
        leadingDimension_ = Max(height, 1);
        memory_.Require(leadingDimension_ * width);
        data_ = memory_.Buffer();
    }
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Resize_(
    Int height, Int width, Int leadingDimension)
{
    const bool reallocate =
        height > leadingDimension_ || width > width_ ||
        leadingDimension != leadingDimension_;
    height_ = height;
    width_ = width;
    if (reallocate)
    {
        leadingDimension_ = leadingDimension;
        memory_.Require(leadingDimension*width);
        data_ = memory_.Buffer();
    }
}

// For supporting duck typing
// ==========================
template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(El::Grid const& grid)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (grid != El::Grid::Trivial())
            LogicError("Tried to construct a Matrix with a nontrivial Grid");
        )
        }

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetGrid(El::Grid const& grid)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (grid != El::Grid::Trivial())
            LogicError("Tried to assign nontrivial Grid to Matrix");
        )
        }

template<typename Ring>
El::Grid const& Matrix<Ring, Device::GPU>::Grid() const
{
    EL_DEBUG_CSE;
    return El::Grid::Trivial();
}

template<typename Ring>
void
Matrix<Ring, Device::GPU>::Align(Int colAlign, Int rowAlign, bool constrain)
{
    EL_DEBUG_CSE;
        EL_DEBUG_ONLY(
            if (colAlign != 0 || rowAlign != 0)
                LogicError("Attempted to impose nontrivial alignment on Matrix");
            )
        }

template<typename Ring>
int Matrix<Ring, Device::GPU>::ColAlign() const EL_NO_EXCEPT { return 0; }
template<typename Ring>
int Matrix<Ring, Device::GPU>::RowAlign() const EL_NO_EXCEPT { return 0; }
#endif // 0

#if 0
#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(Ring) EL_EXTERN template class Matrix<Ring>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN
#endif // 0
} // namespace El

#endif // ifndef EL_MATRIX_IMPL_GPU_HPP_

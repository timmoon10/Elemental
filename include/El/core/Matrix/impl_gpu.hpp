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
    : AbstractMatrix<Ring>{height, width, Max(height,1)}
{
    memory_.Require(this->LDim() * width);
    data_ = memory_.Buffer();
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Int height, Int width, Int leadingDimension)
    : AbstractMatrix<Ring>{height, width, leadingDimension}
{
    memory_.Require(leadingDimension*width);
    data_ = memory_.Buffer();
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix
(Int height, Int width, DevicePtr<const Ring> buffer, Int leadingDimension)
    : AbstractMatrix<Ring>{LOCKED_VIEW,height,width,leadingDimension},
    data_{const_cast<Ring*>(buffer)}
{
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix
(Int height, Int width, DevicePtr<Ring> buffer, Int leadingDimension)
    : AbstractMatrix<Ring>{VIEW,height,width,leadingDimension},
      data_(buffer)
{
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Matrix<Ring, Device::GPU> const& A)
    : AbstractMatrix<Ring>{A}
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
    : Matrix{A.Height(), A.Width(), A.LDim()}
{
    EL_DEBUG_CSE;
    auto stream = GPUManager::Stream();
    EL_CHECK_CUDA(cudaMemcpy2DAsync(data_, this->LDim()*sizeof(Ring),
                                    A.LockedBuffer(), A.LDim()*sizeof(Ring),
                                    A.Height()*sizeof(Ring), A.Width(),
                                    cudaMemcpyHostToDevice,
                                    stream));
    EL_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Ring>
Matrix<Ring, Device::GPU>&
Matrix<Ring, Device::GPU>::operator=(Matrix<Ring, Device::CPU> const& A)
{
    auto A_new = Matrix<Ring, Device::GPU>(A);
    *this = std::move(A_new);
    return *this;
}

template<typename Ring>
Matrix<Ring, Device::GPU>::Matrix(Matrix<Ring, Device::GPU>&& A) EL_NO_EXCEPT
    : AbstractMatrix<Ring>{std::move(A)},
      memory_{std::move(A.memory_)}, data_{nullptr}
{ std::swap(data_, A.data_); }

template<typename Ring>
Matrix<Ring, Device::GPU>::~Matrix() { }

template<typename Ring>
void Matrix<Ring, Device::GPU>::Attach
(Int height, Int width, DevicePtr<Ring> buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->FixedSize())
        LogicError("Cannot attach a new buffer to a view with fixed size");
#endif // !EL_RELEASE
    Attach_(height, width, buffer, leadingDimension);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::LockedAttach
(Int height, Int width, DevicePtr<const Ring> buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (this->FixedSize())
        LogicError("Cannot attach a new buffer to a view with fixed size");
#endif // !EL_RELEASE

    LockedAttach_(height, width, buffer, leadingDimension);
}

#if 0

template<typename Ring>
void Matrix<Ring, Device::GPU>::Control
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (this->FixedSize())
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

#endif // 0

// Make a copy
// -----------
template<typename Ring>
Matrix<Ring, Device::GPU>&
Matrix<Ring, Device::GPU>::operator=(Matrix<Ring, Device::GPU> const& A)
{
    EL_DEBUG_CSE;
    Copy(A, *this);
    return *this;
}

// Move assignment
// ---------------
template<typename Ring>
Matrix<Ring, Device::GPU>&
Matrix<Ring, Device::GPU>::operator=(Matrix<Ring, Device::GPU>&& A)
{
    EL_DEBUG_CSE;
    if (this->Viewing() || A.Viewing())
    {
        operator=((Matrix<Ring, Device::GPU> const&)A);
    }
    else
    {
        AbstractMatrix<Ring>::operator=(A);
        memory_.ShallowSwap(A.memory_);
        std::swap(data_, A.data_);
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
// -------------

template<typename Ring>
Int Matrix<Ring, Device::GPU>::do_get_memory_size_() const EL_NO_EXCEPT
{ return memory_.Size(); }

template <typename Ring>
Device Matrix<Ring, Device::GPU>::do_get_device_() const EL_NO_EXCEPT
{
    return Device::GPU;
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::do_empty_(bool freeMemory)
{
    if (freeMemory)
        memory_.Empty();
    data_ = nullptr;
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::do_resize_()
{
    data_ = memory_.Require(this->LDim() * this->Width());
}

template<typename Ring>
Ring* Matrix<Ring, Device::GPU>::Buffer() EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (this->Locked())
            LogicError("Cannot return non-const buffer of locked Matrix");
        )
        return data_;
}

template<typename Ring>
Ring* Matrix<Ring, Device::GPU>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        if (this->Locked())
            LogicError("Cannot return non-const buffer of locked Matrix");
        )
        if (data_ == nullptr)
            return nullptr;
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
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
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
}

// Advanced functions
// ------------------

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetMemoryMode(unsigned int mode)
{ memory_.SetMode(mode); }

template<typename Ring>
unsigned int Matrix<Ring, Device::GPU>::MemoryMode() const EL_NO_EXCEPT
{ return memory_.Mode(); }

// Single-entry manipulation
// =========================

template<typename Ring>
Ring Matrix<Ring, Device::GPU>::Get(Int i, Int j) const
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    Ring val = Ring(0);
    EL_CHECK_CUDA(cudaMemcpy( &val, &data_[i+j*this->LDim()],
                              sizeof(Ring), cudaMemcpyDeviceToHost ));
    return val;
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::GPU>::GetRealPart(Int i, Int j) const
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    return El::RealPart(Get(i, j));
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::GPU>::GetImagPart(Int i, Int j) const
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    return El::ImagPart(Get(i, j));
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Set(Int i, Int j, Ring const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    EL_CHECK_CUDA(cudaMemcpy( &data_[i+j*this->LDim()], &alpha,
                              sizeof(Ring), cudaMemcpyHostToDevice ));
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Set(Entry<Ring> const& entry)
    EL_NO_RELEASE_EXCEPT
{ Set(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::SetRealPart(
    Int i, Int j, Base<Ring> const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    Ring val = Get(i, j);
    El::SetRealPart(val, alpha);
    Set(i, j, val);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetRealPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ SetRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::SetImagPart(Int i, Int j, Base<Ring> const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    Ring val = Get(i, j);
    El::SetImagPart(val, alpha);
    Set(i, j, val);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::SetImagPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ SetImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::GPU>::Update(Int i, Int j, Ring const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    Ring val = Get(i, j);
    val += alpha;
    Set(i, j, val);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Update(Entry<Ring> const& entry)
    EL_NO_RELEASE_EXCEPT
{ Update(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::UpdateRealPart(Int i, Int j, Base<Ring> const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    Ring val = Get(i, j);
    El::UpdateRealPart(val, alpha);
    Set(i, j, val);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::UpdateRealPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ UpdateRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::GPU>::UpdateImagPart(Int i, Int j, Base<Ring> const& alpha)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
    Ring val = Get(i, j);
    El::UpdateImagPart(val, alpha);
    Set(i, j, val);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::UpdateImagPart(Entry<Base<Ring>> const& entry)
    EL_NO_RELEASE_EXCEPT
{ UpdateImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::GPU>::MakeReal(Int i, Int j)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        Set(i, j, GetRealPart(i,j));
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Conjugate(Int i, Int j)
{
    EL_DEBUG_CSE;
    EL_DEBUG_ONLY(
        this->AssertValidEntry(i, j);
        if (this->Locked())
            LogicError("Cannot modify data of locked matrices");
        )
        Set(i, j, El::Conj(Get(i,j)));
}

#if 0
// Private routines
// ################


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

// Exchange metadata with another matrix
// =====================================
template<typename Ring>
void Matrix<Ring, Device::GPU>::ShallowSwap(Matrix<Ring, Device::GPU>& A)
{
    AbstractMatrix<Ring>::ShallowSwap(A);
    memory_.ShallowSwap(A.memory_);
    std::swap(data_, A.data_);
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::Attach_(
    Int height, Int width, Ring* buffer, Int leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>((this->ViewType() & ~LOCKED_OWNER) | VIEW));
    this->SetSize_(height, width, leadingDimension);

    data_ = buffer;
}

template<typename Ring>
void Matrix<Ring, Device::GPU>::LockedAttach_(
    Int height, Int width, const Ring* buffer, Int leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>(this->ViewType() | LOCKED_VIEW));
    this->SetSize_(height, width, leadingDimension);

    data_ = const_cast<Ring*>(buffer);
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
#endif //0

// Return a reference to a single entry without error-checking
// ===========================================================
template<typename Ring>
Ring const& Matrix<Ring, Device::GPU>::CRef(Int i, Int j) const
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template<typename Ring>
Ring const& Matrix<Ring, Device::GPU>::operator()(Int i, Int j) const
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template<typename Ring>
Ring& Matrix<Ring, Device::GPU>::Ref(Int i, Int j)
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

template<typename Ring>
Ring& Matrix<Ring, Device::GPU>::operator()(Int i, Int j)
{
    LogicError("Attempted to get reference to entry of a GPU matrix");
    return data_[0];
}

#if 0
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


#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(Ring) EL_EXTERN template class Matrix<Ring,Device::GPU>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El

#endif // ifndef EL_MATRIX_IMPL_GPU_HPP_

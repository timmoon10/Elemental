/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_IMPL_CPU_HPP_
#define EL_MATRIX_IMPL_CPU_HPP_

namespace El
{

// Public routines
// ###############

// Constructors and destructors
// ============================

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix() { }

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(Int height, Int width)
    : AbstractMatrix<Ring>{height,width,Max(height,1)}
{
    memory_.Require(Max(height,1) * width);
    data_ = memory_.Buffer();
    // TODO(poulson): Consider explicitly zeroing
}

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(Int height, Int width, Int leadingDimension)
    : AbstractMatrix<Ring>{height, width, leadingDimension}
{
    memory_.Require(leadingDimension*width);
    data_ = memory_.Buffer();
}

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix
(Int height, Int width, const Ring* buffer, Int leadingDimension)
    : AbstractMatrix<Ring>{LOCKED_VIEW,height,width,leadingDimension},
    data_(const_cast<Ring*>(buffer))
{
}

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix
(Int height, Int width, Ring* buffer, Int leadingDimension)
    : AbstractMatrix<Ring>{VIEW,height,width,leadingDimension},
    data_(buffer)
{
}

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(Matrix<Ring, Device::CPU> const& A)
    : AbstractMatrix<Ring>{A}
{
    EL_DEBUG_CSE
    if (&A != this)
        *this = A;
    else
        LogicError("You just tried to construct a Matrix with itself!");
}

#ifdef HYDROGEN_HAVE_CUDA
template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(Matrix<Ring, Device::GPU> const& A)
    : Matrix{A.Height(), A.Width(), A.LDim()}
{
    EL_DEBUG_CSE;
    auto stream = GPUManager::Stream();
    EL_CHECK_CUDA(cudaMemcpy2DAsync(data_, this->LDim()*sizeof(Ring),
                                    A.LockedBuffer(), A.LDim()*sizeof(Ring),
                                    A.Height()*sizeof(Ring), A.Width(),
                                    cudaMemcpyDeviceToHost,
                                    stream));
    EL_CHECK_CUDA(cudaStreamSynchronize(stream));
}
#endif

template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(Matrix<Ring, Device::CPU>&& A) EL_NO_EXCEPT
    : AbstractMatrix<Ring>{A},
      memory_(std::move(A.memory_)), data_(nullptr)
{
    std::swap(data_, A.data_);
}

template<typename Ring>
Matrix<Ring, Device::CPU>::~Matrix() { }

// Assignment and reconfiguration
// ==============================


template<typename Ring>
void Matrix<Ring, Device::CPU>::Attach
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (this->FixedSize())
          LogicError("Cannot attach a new buffer to a view with fixed size");
   )
    Attach_(height, width, buffer, leadingDimension);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::LockedAttach
(Int height, Int width, const Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (this->FixedSize())
          LogicError("Cannot attach a new buffer to a view with fixed size");
   )
    LockedAttach_(height, width, buffer, leadingDimension);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Control
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (this->FixedSize())
          LogicError("Cannot attach a new buffer to a view with fixed size");
   )
    Control_(height, width, buffer, leadingDimension);
}

// Operator overloading
// ====================

// Return a view
// -------------
template<typename Ring>
Matrix<Ring, Device::CPU>
Matrix<Ring, Device::CPU>::operator()(Range<Int> I, Range<Int> J)
{
    EL_DEBUG_CSE
    if (this->Locked())
        return LockedView(*this, I, J);
    else
        return View(*this, I, J);
}

template<typename Ring>
const Matrix<Ring, Device::CPU>
Matrix<Ring, Device::CPU>::operator()(Range<Int> I, Range<Int> J) const
{
    EL_DEBUG_CSE
    return LockedView(*this, I, J);
}

// Return a (potentially non-contiguous) subset of indices
// -------------------------------------------------------
template<typename Ring>
Matrix<Ring, Device::CPU> Matrix<Ring, Device::CPU>::operator()
(Range<Int> I, vector<Int> const& J) const
{
    EL_DEBUG_CSE
    Matrix<Ring, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template<typename Ring>
Matrix<Ring, Device::CPU> Matrix<Ring, Device::CPU>::operator()
(vector<Int> const& I, Range<Int> J) const
{
    EL_DEBUG_CSE
    Matrix<Ring, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

template<typename Ring>
Matrix<Ring, Device::CPU> Matrix<Ring, Device::CPU>::operator()
(vector<Int> const& I, vector<Int> const& J) const
{
    EL_DEBUG_CSE
    Matrix<Ring, Device::CPU> ASub;
    GetSubmatrix(*this, I, J, ASub);
    return ASub;
}

// Make a copy
// -----------
template<typename Ring>
Matrix<Ring, Device::CPU>&
Matrix<Ring, Device::CPU>::operator=(Matrix<Ring, Device::CPU> const& A)
{
    EL_DEBUG_CSE
    Copy(A, *this);
    return *this;
}

// Move assignment
// ---------------
template<typename Ring>
Matrix<Ring, Device::CPU>&
Matrix<Ring, Device::CPU>::operator=(Matrix<Ring, Device::CPU>&& A)
{
    EL_DEBUG_CSE
    if (this->Viewing() || A.Viewing())
    {
        operator=(static_cast<Matrix<Ring, Device::CPU> const&>(A));
    }
    else
    {
        AbstractMatrix<Ring>::operator=(A);
        memory_.ShallowSwap(A.memory_);
        std::swap(data_, A.data_);
    }
    return *this;
}

// Rescaling
// ---------
template<typename Ring>
Matrix<Ring, Device::CPU> const&
Matrix<Ring, Device::CPU>::operator*=(Ring const& alpha)
{
    EL_DEBUG_CSE
    Scale(alpha, *this);
    return *this;
}

// Addition/subtraction
// --------------------
template<typename Ring>
Matrix<Ring, Device::CPU> const&
Matrix<Ring, Device::CPU>::operator+=(Matrix<Ring, Device::CPU> const& A)
{
    EL_DEBUG_CSE
    Axpy(Ring(1), A, *this);
    return *this;
}

template<typename Ring>
Matrix<Ring, Device::CPU> const&
Matrix<Ring, Device::CPU>::operator-=(Matrix<Ring, Device::CPU> const& A)
{
    EL_DEBUG_CSE
    Axpy(Ring(-1), A, *this);
    return *this;
}

// Basic queries
// =============

template<typename Ring>
Int Matrix<Ring, Device::CPU>::do_get_memory_size_() const EL_NO_EXCEPT
{ return memory_.Size(); }

template <typename Ring>
Device Matrix<Ring, Device::CPU>::do_get_device_() const EL_NO_EXCEPT
{
    return Device::CPU;
}

template<typename Ring>
Ring* Matrix<Ring, Device::CPU>::Buffer() EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (this->Locked())
          LogicError("Cannot return non-const buffer of locked Matrix");
   )
    return data_;
}

template<typename Ring>
Ring* Matrix<Ring, Device::CPU>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
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
const Ring* Matrix<Ring, Device::CPU>::LockedBuffer() const EL_NO_EXCEPT
{ return data_; }

template<typename Ring>
const Ring*
Matrix<Ring, Device::CPU>::LockedBuffer(Int i, Int j) const EL_NO_EXCEPT
{
    EL_DEBUG_CSE
    if (data_ == nullptr)
        return nullptr;
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return &data_[i+j*this->LDim()];
}

// Advanced functions
// ==================

template<typename Ring>
void Matrix<Ring, Device::CPU>::SetMemoryMode(unsigned int mode)
{ memory_.SetMode(mode); }

template<typename Ring>
unsigned int Matrix<Ring, Device::CPU>::MemoryMode() const EL_NO_EXCEPT
{ return memory_.Mode(); }

// Single-entry manipulation
// =========================

template<typename Ring>
Ring Matrix<Ring, Device::CPU>::Get(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return CRef(i, j);
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::CPU>::GetRealPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return El::RealPart(CRef(i, j));
}

template<typename Ring>
Base<Ring> Matrix<Ring, Device::CPU>::GetImagPart(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    return El::ImagPart(CRef(i, j));
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Set(Int i, Int j, Ring const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    Ref(i, j) = alpha;
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Set(Entry<Ring> const& entry)
EL_NO_RELEASE_EXCEPT
{ Set(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::CPU>::SetRealPart(
    Int i, Int j, Base<Ring> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::SetRealPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::SetRealPart(Entry<Base<Ring>> const& entry)
EL_NO_RELEASE_EXCEPT
{ SetRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::CPU>::SetImagPart(Int i, Int j, Base<Ring> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::SetImagPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::SetImagPart(Entry<Base<Ring>> const& entry)
EL_NO_RELEASE_EXCEPT
{ SetImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::CPU>::Update(Int i, Int j, Ring const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    Ref(i, j) += alpha;
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Update(Entry<Ring> const& entry)
EL_NO_RELEASE_EXCEPT
{ Update(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::CPU>::UpdateRealPart(Int i, Int j, Base<Ring> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::UpdateRealPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::UpdateRealPart(Entry<Base<Ring>> const& entry)
EL_NO_RELEASE_EXCEPT
{ UpdateRealPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void
Matrix<Ring, Device::CPU>::UpdateImagPart(Int i, Int j, Base<Ring> const& alpha)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    if (i == END) i = this->Height() - 1;
    if (j == END) j = this->Width() - 1;
    El::UpdateImagPart(Ref(i, j), alpha);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::UpdateImagPart(Entry<Base<Ring>> const& entry)
EL_NO_RELEASE_EXCEPT
{ UpdateImagPart(entry.i, entry.j, entry.value); }

template<typename Ring>
void Matrix<Ring, Device::CPU>::MakeReal(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    Set(i, j, GetRealPart(i,j));
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Conjugate(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    Set(i, j, El::Conj(Get(i,j)));
}

// Private routines
// ################

// Exchange metadata with another matrix
// =====================================
template<typename Ring>
void Matrix<Ring, Device::CPU>::ShallowSwap(Matrix<Ring, Device::CPU>& A)
{
    AbstractMatrix<Ring>::ShallowSwap(A);
    memory_.ShallowSwap(A.memory_);
    std::swap(data_, A.data_);
}

// Reconfigure without error-checking
// ==================================

template<typename Ring>
void Matrix<Ring, Device::CPU>::do_empty_(bool freeMemory)
{
    if (freeMemory)
        memory_.Empty();
    data_ = nullptr;
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Attach_
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    this->SetViewType(static_cast<El::ViewType>((this->ViewType() & ~LOCKED_OWNER) | VIEW));
    this->SetSize_(height, width, leadingDimension);
    data_ = buffer;
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::LockedAttach_
(Int height, Int width, const Ring* buffer, Int leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>(this->ViewType() | LOCKED_VIEW));
    this->SetSize_(height,width,leadingDimension);
    data_ = const_cast<Ring*>(buffer);
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::Control_
(Int height, Int width, Ring* buffer, Int leadingDimension)
{
    this->SetViewType(
        static_cast<El::ViewType>(this->ViewType() & ~LOCKED_VIEW));
    this->SetSize_(height,width,leadingDimension);
    data_ = buffer;
}

// Return a reference to a single entry without error-checking
// ===========================================================
template<typename Ring>
Ring const& Matrix<Ring, Device::CPU>::CRef(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*this->LDim()];
}

template<typename Ring>
Ring const& Matrix<Ring, Device::CPU>::operator()(Int i, Int j) const
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(this->AssertValidEntry(i, j))
    return data_[i+j*this->LDim()];
}

template<typename Ring>
Ring& Matrix<Ring, Device::CPU>::Ref(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    return data_[i+j*this->LDim()];
}

template<typename Ring>
Ring& Matrix<Ring, Device::CPU>::operator()(Int i, Int j)
EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      this->AssertValidEntry(i, j);
      if (this->Locked())
          LogicError("Cannot modify data of locked matrices");
   )
    return data_[i+j*this->LDim()];
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::do_resize_()
{
    data_ = memory_.Require(this->LDim() * this->Width());
}

// For supporting duck typing
// ==========================
template<typename Ring>
Matrix<Ring, Device::CPU>::Matrix(El::Grid const& grid)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (grid != El::Grid::Trivial())
          LogicError("Tried to construct a Matrix with a nontrivial Grid");
   )
}

template<typename Ring>
void Matrix<Ring, Device::CPU>::SetGrid(El::Grid const& grid)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (grid != El::Grid::Trivial())
          LogicError("Tried to assign nontrivial Grid to Matrix");
   )
}

template<typename Ring>
El::Grid const& Matrix<Ring, Device::CPU>::Grid() const
{
    EL_DEBUG_CSE
    return El::Grid::Trivial();
}

template<typename Ring>
void
Matrix<Ring, Device::CPU>::Align(Int colAlign, Int rowAlign, bool constrain)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if (colAlign != 0 || rowAlign != 0)
          LogicError("Attempted to impose nontrivial alignment on Matrix");
   )
}

template<typename Ring>
int Matrix<Ring, Device::CPU>::ColAlign() const EL_NO_EXCEPT { return 0; }
template<typename Ring>
int Matrix<Ring, Device::CPU>::RowAlign() const EL_NO_EXCEPT { return 0; }

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
} // namespace El

#endif // ifndef EL_MATRIX_IMPL_CPU_HPP_

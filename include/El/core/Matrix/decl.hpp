/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MATRIX_DECL_HPP
#define EL_MATRIX_DECL_HPP

#include <El/core/Device.hpp>
#include <El/core/Grid.hpp>
#include <El/core/Memory.hpp>

namespace El
{

// Matrix base for arbitrary rings
template<typename Ring, Device Dev>
class Matrix;

// Specialization for CPU
template<typename Ring>
class Matrix<Ring, Device::CPU>
{
public:
    // Constructors and destructors
    // ============================

    // Create a 0x0 matrix
    Matrix();

    // Create a matrix with the specified dimensions
    Matrix(Int height, Int width);

    // Create a matrix with the specified dimensions and leading dimension
    Matrix(Int height, Int width, Int leadingDimension);

    // Construct a matrix around an existing (possibly immutable) buffer
    Matrix(Int height, Int width, const Ring* buffer, Int leadingDimension);
    Matrix(Int height, Int width, Ring* buffer, Int leadingDimension);

    // Create a copy of a matrix
    Matrix(Matrix<Ring, Device::CPU> const& A);

    // Create a copy of a matrix from a GPU matrix
    Matrix(Matrix<Ring, Device::GPU> const& A);

    // Move the metadata from a given matrix
    Matrix(Matrix<Ring, Device::CPU>&& A) EL_NO_EXCEPT;

    // Destructor
    ~Matrix();

    // Copy assignment
    Matrix<Ring, Device::CPU> const& operator=(
        Matrix<Ring, Device::CPU> const& A);

    // Assign by copying data from a GPU
    Matrix<Ring, Device::CPU> const& operator=(
        Matrix<Ring, Device::GPU> const& A);

    // Move assignment
    Matrix<Ring, Device::CPU>& operator=(Matrix<Ring, Device::CPU>&& A);

    //
    // Assignment and reconfiguration
    //

    void Empty(bool freeMemory=true);
    void Resize(Int height, Int width);
    void Resize(Int height, Int width, Int leadingDimension);

    // Reconfigure around the given buffer, but do not assume ownership
    void Attach
    (Int height, Int width, Ring* buffer, Int leadingDimension);
    void LockedAttach
    (Int height, Int width, const Ring* buffer, Int leadingDimension);

    // Reconfigure around the given buffer and assume ownership
    void Control
    (Int height, Int width, Ring* buffer, Int leadingDimension);

    // Force the size to remain constant (but allow the entries to be modified).
    void FixSize() EL_NO_EXCEPT;

    //
    // Operator overloading
    //

    // Return a view
    Matrix<Ring, Device::CPU> operator()(Range<Int> I, Range<Int> J);
    const Matrix<Ring, Device::CPU> operator()(
        Range<Int> I, Range<Int> J) const;

    // Return a copy of (potentially non-contiguous) subset of indices
    Matrix<Ring, Device::CPU> operator()(
        Range<Int> I, vector<Int> const& J) const;
    Matrix<Ring, Device::CPU> operator()(
        vector<Int> const& I, Range<Int> J) const;
    Matrix<Ring, Device::CPU> operator()(
        vector<Int> const& I, vector<Int> const& J) const;


    // Rescaling
    Matrix<Ring, Device::CPU> const& operator*=(Ring const& alpha);

    // Addition/substraction
    Matrix<Ring, Device::CPU> const&
    operator+=(Matrix<Ring, Device::CPU> const& A);

    Matrix<Ring, Device::CPU> const&
    operator-=(Matrix<Ring, Device::CPU> const& A);

    //
    // Basic queries
    //

    Int Height() const EL_NO_EXCEPT;
    Int Width() const EL_NO_EXCEPT;
    Int LDim() const EL_NO_EXCEPT;
    Int MemorySize() const EL_NO_EXCEPT;
    Int DiagonalLength(Int offset=0) const EL_NO_EXCEPT;

    Ring* Buffer() EL_NO_RELEASE_EXCEPT;
    Ring* Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT;
    const Ring* LockedBuffer() const EL_NO_EXCEPT;
    const Ring* LockedBuffer(Int i, Int j) const EL_NO_EXCEPT;

    bool Viewing()   const EL_NO_EXCEPT;
    bool FixedSize() const EL_NO_EXCEPT;
    bool Locked()    const EL_NO_EXCEPT;
    // Advanced
    // --------
    void SetViewType(El::ViewType viewType) EL_NO_EXCEPT;
    El::ViewType ViewType() const EL_NO_EXCEPT;

    // Single-entry manipulation
    // =========================
    Ring Get(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;
    Base<Ring> GetRealPart(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;
    Base<Ring> GetImagPart(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;

    void Set(Int i, Int j, Ring const& alpha) EL_NO_RELEASE_EXCEPT;
    void Set(Entry<Ring> const& entry) EL_NO_RELEASE_EXCEPT;

    void SetRealPart
    (Int i, Int j, Base<Ring> const& alpha) EL_NO_RELEASE_EXCEPT;
    void SetImagPart
    (Int i, Int j, Base<Ring> const& alpha) EL_NO_RELEASE_EXCEPT;

    void SetRealPart
    (Entry<Base<Ring>> const& entry) EL_NO_RELEASE_EXCEPT;
    void SetImagPart
    (Entry<Base<Ring>> const& entry) EL_NO_RELEASE_EXCEPT;

    void Update(Int i, Int j, Ring const& alpha) EL_NO_RELEASE_EXCEPT;
    void Update(Entry<Ring> const& entry) EL_NO_RELEASE_EXCEPT;

    void UpdateRealPart
    (Int i, Int j, Base<Ring> const& alpha) EL_NO_RELEASE_EXCEPT;
    void UpdateImagPart
    (Int i, Int j, Base<Ring> const& alpha) EL_NO_RELEASE_EXCEPT;

    void UpdateRealPart
    (Entry<Base<Ring>> const& entry) EL_NO_RELEASE_EXCEPT;
    void UpdateImagPart
    (Entry<Base<Ring>> const& entry) EL_NO_RELEASE_EXCEPT;

    void MakeReal(Int i, Int j) EL_NO_RELEASE_EXCEPT;
    void Conjugate(Int i, Int j) EL_NO_RELEASE_EXCEPT;

    // Return a reference to a single entry without error-checking
    // -----------------------------------------------------------
    inline Ring const& CRef(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;
    inline Ring const& operator()(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;

    inline Ring& Ref(Int i, Int j=0) EL_NO_RELEASE_EXCEPT;
    inline Ring& operator()(Int i, Int j=0) EL_NO_RELEASE_EXCEPT;

private:
    // Member variables
    // ================
    El::ViewType viewType_=OWNER;
    Int height_=0, width_=0, leadingDimension_=1;

    Memory<Ring,Device::CPU> memory_;
    // Const-correctness is internally managed to avoid the need for storing
    // two separate pointers with different 'const' attributes
    Ring* data_=nullptr;

    // Exchange metadata with another matrix
    // =====================================
    void ShallowSwap(Matrix<Ring, Device::CPU>& A);

    // Reconfigure without error-checking
    // ==================================
    void Empty_(bool freeMemory=true);
    void Resize_(Int height, Int width);
    void Resize_(Int height, Int width, Int leadingDimension);

    void Control_
    (Int height, Int width, Ring* buffer, Int leadingDimension);
    void Attach_
    (Int height, Int width, Ring* buffer, Int leadingDimension);
    void LockedAttach_
    (Int height, Int width, const Ring* buffer, Int leadingDimension);

    // Assertions
    // ==========
    void AssertValidDimensions(Int height, Int width) const;
    void AssertValidDimensions
    (Int height, Int width, Int leadingDimension) const;
    void AssertValidEntry(Int i, Int j) const;

    // Friend declarations
    // ===================
    template<typename S, Device D> friend class Matrix;
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

    // For supporting duck typing
    // ==========================
    // The following are provided in order to aid duck-typing over
    // {Matrix, DistMatrix, etc.}.

    // This is equivalent to the trivial constructor in functionality
    // (though an error is thrown if 'grid' is not equal to 'Grid::Trivial()').
    explicit Matrix(El::Grid const& grid);

    // This is a no-op
    // (though an error is thrown if 'grid' is not equal to 'Grid::Trivial()').
    void SetGrid(El::Grid const& grid);

    // This always returns 'Grid::Trivial()'.
    El::Grid const& Grid() const;

    // This is a no-op
    // (though an error is thrown if 'colAlign' or 'rowAlign' is not zero).
    void Align(Int colAlign, Int rowAlign, bool constrain=true);

    // These always return 0.
    int ColAlign() const EL_NO_EXCEPT;
    int RowAlign() const EL_NO_EXCEPT;
};

// GPU version
template<typename Ring>
class Matrix<Ring, Device::GPU>
{
public:
    // Constructors and destructors
    // ============================

    // Create a 0x0 matrix
    Matrix();

    // Create a matrix with the specified dimensions
    Matrix(Int height, Int width);

    // Create a matrix with the specified dimensions and leading dimension
    Matrix(Int height, Int width, Int leadingDimension);

    // Construct a matrix around an existing (possibly immutable) buffer
    Matrix(Int height, Int width, const Ring* buffer, Int leadingDimension);
    Matrix(Int height, Int width, Ring* buffer, Int leadingDimension);

    // Create a copy of a matrix
    Matrix(Matrix<Ring, Device::GPU> const& A);

    // Create a copy of a matrix from a CPU matrix
    Matrix(Matrix<Ring, Device::CPU> const& A);

    // Move the metadata from a given matrix
    Matrix(Matrix<Ring, Device::GPU>&& A) EL_NO_EXCEPT;

    // Destructor
    ~Matrix();

    // Copy assignment
    Matrix<Ring, Device::GPU> const& operator=(
        Matrix<Ring, Device::GPU> const& A);

    // Assign by copying data from a CPU matrix
    Matrix<Ring, Device::GPU> const& operator=(
        Matrix<Ring, Device::CPU> const& A);

    // Move assignment
    Matrix<Ring, Device::GPU>& operator=(Matrix<Ring, Device::GPU>&& A);

    DevicePtr<const Ring> Data() { return data_; }

    DevicePtr<Ring> Buffer() EL_NO_RELEASE_EXCEPT;
    DevicePtr<Ring> Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT;
    DevicePtr<const Ring> LockedBuffer() const EL_NO_EXCEPT;
    DevicePtr<const Ring> LockedBuffer(Int i, Int j) const EL_NO_EXCEPT;


    // Reconfigure around the given buffer, but do not assume ownership
    void Attach(Int height, Int width, Ring* buffer, Int leadingDimension);
    void LockedAttach(
        Int height, Int width, const Ring* buffer, Int leadingDimension);

    // Return a view
    Matrix<Ring, Device::GPU> operator()(Range<Int> I, Range<Int> J);

    // Return a locked view
    const Matrix<Ring, Device::GPU>
    operator()(Range<Int> I, Range<Int> J) const;

    Int Height() const EL_NO_EXCEPT;
    Int Width() const EL_NO_EXCEPT;
    Int LDim() const EL_NO_EXCEPT;
    Int MemorySize() const EL_NO_EXCEPT;

    bool Viewing() const EL_NO_EXCEPT;
    bool FixedSize() const EL_NO_EXCEPT;
    bool Locked()    const EL_NO_EXCEPT;

private:

    void Attach_(Int height, Int width, Ring* buffer, Int leadingDimension);
    void LockedAttach_(
        Int height, Int width, const Ring* buffer, Int leadingDimension);

    template<typename S, Device D> friend class Matrix;
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

private:

    El::ViewType viewType_=OWNER;

    Int height_=0, width_=0, leadingDimension_=1;

    Memory<Ring,Device::GPU> memory_;

    DevicePtr<Ring> data_=nullptr;

};// class Matrix<Ring,Device::GPU>

} // namespace El

#endif // ifndef EL_MATRIX_DECL_HPP

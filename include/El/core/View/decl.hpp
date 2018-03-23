/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_VIEW_DECL_HPP
#define EL_VIEW_DECL_HPP

namespace El
{

// View an entire matrix
// =====================

// (Sequential) matrix
// -------------------

template <typename T, Device D>
void View(Matrix<T,D>& A, Matrix<T,D>& B);
template <typename T, Device D>
void LockedView(Matrix<T,D>& A, const Matrix<T,D>& B);

template <typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B);
template <typename T, Device D>
Matrix<T,D> LockedView(const Matrix<T,D>& B);

// Abstract Sequential Matrix
// --------------------------

template<typename T>
void View(AbstractMatrix<T>& A, AbstractMatrix<T>& B);
template<typename T>
void LockedView(AbstractMatrix<T>& A, const AbstractMatrix<T>& B);

// ElementalMatrix
// ---------------

template <typename T>
void View(ElementalMatrix<T>& A, ElementalMatrix<T>& B);
template <typename T>
void LockedView(ElementalMatrix<T>& A, const ElementalMatrix<T>& B);

// Return by value
// ^^^^^^^^^^^^^^^
template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> View(DistMatrix<T,U,V,wrapType,D>& B);
template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> LockedView(const DistMatrix<T,U,V,wrapType,D>& B);

// BlockMatrix
// -----------
template <typename T>
void View(BlockMatrix<T>& A, BlockMatrix<T>& B);
template <typename T>
void LockedView(BlockMatrix<T>& A, const BlockMatrix<T>& B);

// Mixed
// -----
template <typename T>
void View (BlockMatrix<T>& A, ElementalMatrix<T>& B);
template <typename T>
void LockedView (BlockMatrix<T>& A, const ElementalMatrix<T>& B);

template <typename T>
void View (ElementalMatrix<T>& A, BlockMatrix<T>& B);
template <typename T>
void LockedView (ElementalMatrix<T>& A, const BlockMatrix<T>& B);

// AbstractDistMatrix
// ------------------
template <typename T>
void View(AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B);
template <typename T>
void LockedView(AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B);

// View a contiguous submatrix
// ===========================

// (Sequential) Matrix
// -------------------

template <typename T, Device D>
void View
(Matrix<T,D>& A,
  Matrix<T,D>& B,
  Int i, Int j,
  Int height, Int width);
template <typename T, Device D>
void LockedView
(      Matrix<T,D>& A,
  const Matrix<T,D>& B,
  Int i, Int j,
  Int height, Int width);

template <typename T, Device D>
void View
(Matrix<T,D>& A,
  Matrix<T,D>& B,
  Range<Int> I, Range<Int> J);
template <typename T, Device D>
void LockedView
(      Matrix<T,D>& A,
  const Matrix<T,D>& B,
  Range<Int> I, Range<Int> J);

// Return by value
// ^^^^^^^^^^^^^^^

template <typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B, Int i, Int j, Int height, Int width);
template <typename T, Device D>
Matrix<T,D> LockedView(const Matrix<T,D>& B, Int i, Int j, Int height, Int width);

template <typename T, Device D>
Matrix<T,D> View(Matrix<T,D>& B, Range<Int> I, Range<Int> J);
template <typename T, Device D>
Matrix<T,D> LockedView(const Matrix<T,D>& B, Range<Int> I, Range<Int> J);

// Abstract Sequential Matrix
// --------------------------

template <typename T>
void View
(AbstractMatrix<T>& A,
  AbstractMatrix<T>& B,
  Range<Int> I, Range<Int> J);
template <typename T>
void LockedView
(AbstractMatrix<T>& A,
  const AbstractMatrix<T>& B,
  Range<Int> I, Range<Int> J);

// ElementalMatrix
// ---------------

template <typename T>
void View
(ElementalMatrix<T>& A,
  ElementalMatrix<T>& B,
  Int i, Int j, Int height, Int width);
template <typename T>
void LockedView
(      ElementalMatrix<T>& A,
  const ElementalMatrix<T>& B,
  Int i, Int j, Int height, Int width);

template <typename T>
void View
(ElementalMatrix<T>& A,
  ElementalMatrix<T>& B,
  Range<Int> I, Range<Int> J);
template <typename T>
void LockedView
(      ElementalMatrix<T>& A,
  const ElementalMatrix<T>& B,
  Range<Int> I, Range<Int> J);

// Return by value
// ^^^^^^^^^^^^^^^

template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> View
(DistMatrix<T,U,V,wrapType,D>& B, Int i, Int j, Int height, Int width);
template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> LockedView
(const DistMatrix<T,U,V,wrapType,D>& B, Int i, Int j, Int height, Int width);

template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> View
(DistMatrix<T,U,V,wrapType,D>& B, Range<Int> I, Range<Int> J);
template <typename T,Dist U,Dist V,DistWrap wrapType, Device D>
DistMatrix<T,U,V,wrapType,D> LockedView
(const DistMatrix<T,U,V,wrapType,D>& B, Range<Int> I, Range<Int> J);

// BlockMatrix
// -----------

template <typename T>
void View
(BlockMatrix<T>& A,
  BlockMatrix<T>& B,
  Int i,
  Int j,
  Int height,
  Int width);
template <typename T>
void LockedView
(      BlockMatrix<T>& A,
  const BlockMatrix<T>& B,
  Int i, Int j, Int height, Int width);

template <typename T>
void View
(BlockMatrix<T>& A,
  BlockMatrix<T>& B,
  Range<Int> I, Range<Int> J);
template <typename T>
void LockedView
(      BlockMatrix<T>& A,
  const BlockMatrix<T>& B,
  Range<Int> I, Range<Int> J);

// AbstractDistMatrix
// ------------------
template <typename T>
void View
(AbstractDistMatrix<T>& A,
  AbstractDistMatrix<T>& B,
  Int i, Int j, Int height, Int width);
template <typename T>
void LockedView
(      AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
  Int i, Int j, Int height, Int width);

template <typename T>
void View
(AbstractDistMatrix<T>& A,
  AbstractDistMatrix<T>& B,
  Range<Int> I, Range<Int> J);
template <typename T>
void LockedView
(      AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
  Range<Int> I, Range<Int> J);

} // namespace El

#endif // ifndef EL_VIEW_DECL_HPP

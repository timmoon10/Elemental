/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_UTIL_HPP
#define EL_UTIL_HPP

namespace El {

// Median
// ======
template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
ValueInt<Real> Median( const Matrix<Real>& x );
template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
ValueInt<Real> Median( const AbstractDistMatrix<Real>& x );

// Sort
// ====
template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
void Sort
( Matrix<Real>& X, SortType sort=ASCENDING, bool stable=false );
template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
void Sort
( AbstractDistMatrix<Real>& X, SortType sort=ASCENDING, bool stable=false );

template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
vector<ValueInt<Real>>
TaggedSort
( const Matrix<Real>& x, SortType sort=ASCENDING, bool stable=false );
template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
vector<ValueInt<Real>>
TaggedSort
( const AbstractDistMatrix<Real>& x, SortType sort=ASCENDING,
  bool stable=false );

template<typename Real,typename Field>
void ApplyTaggedSortToEachRow
( const vector<ValueInt<Real>>& sortPairs,
        Matrix<Field>& Z );
template<typename Real,typename Field>
void ApplyTaggedSortToEachColumn
( const vector<ValueInt<Real>>& sortPairs,
        Matrix<Field>& Z );

template<typename Real,typename Field>
void ApplyTaggedSortToEachRow
( const vector<ValueInt<Real>>& sortPairs,
        AbstractDistMatrix<Field>& Z );
template<typename Real,typename Field>
void ApplyTaggedSortToEachColumn
( const vector<ValueInt<Real>>& sortPairs,
        AbstractDistMatrix<Field>& Z );

template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
void SortingPermutation
( const Matrix<Real>& x, Permutation& sortPerm, SortType sort=ASCENDING,
  bool stable=false );

template<typename Real,
         typename=DisableIf<IsComplex<Real>>>
void MergeSortingPermutation
( Int n0, Int n1, const Matrix<Real>& x, Permutation& sortPerm,
  SortType sort=ASCENDING );

} // namespace El

#endif // ifndef EL_UTIL_HPP

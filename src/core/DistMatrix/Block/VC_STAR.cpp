/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like.hpp>

#define COLDIST VC
#define ROWDIST STAR

#include "./setup.hpp"

namespace El {

// Public section
// ##############

// Assignment and reconfiguration
// ==============================

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,MC,MR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    copy::ColAllToAllDemote( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,MC,STAR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    copy::PartialColFilter( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,MR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,MC,MR,BLOCK,D> A_MC_MR( A );
    *this = A_MC_MR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,MD,STAR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,MD,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,MR,MC,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,VR,STAR,BLOCK,D> A_VR_STAR( A );
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,MR,STAR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,VR,STAR,BLOCK,D> A_VR_STAR( A );
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,MC,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,BLOCK,D> A_MR_MC( A );
    DistMatrix<T,VR,STAR,BLOCK,D> A_VR_STAR( A_MR_MC );
    A_MR_MC.Empty();
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const BDM& A )
{
    EL_DEBUG_CSE
    copy::Translate( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,VC,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,MR,MC,BLOCK,D> A_MR_MC( A );
    DistMatrix<T,VR,STAR,BLOCK,D> A_VR_STAR( A_MR_MC );
    A_MR_MC.Empty();
    *this = A_VR_STAR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,VR,STAR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,VR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    DistMatrix<T,MC,MR,BLOCK,D> A_MC_MR( A );
    *this = A_MC_MR;
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,STAR,STAR,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    copy::ColFilter( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const DistMatrix<T,CIRC,CIRC,BLOCK,D>& A )
{
    EL_DEBUG_CSE
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template <typename T, Device D>
BDM& BDM::operator=( const BlockMatrix<T>& A )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST && \
      BLOCK == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& ACast = \
        static_cast<const DistMatrix<T,CDIST,RDIST,BLOCK,D>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
    return *this;
}

// Basic queries
// =============
template <typename T, Device D>
mpi::Comm BDM::DistComm() const EL_NO_EXCEPT
{ return this->Grid().VCComm(); }
template <typename T, Device D>
mpi::Comm BDM::CrossComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }
template <typename T, Device D>
mpi::Comm BDM::RedundantComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template <typename T, Device D>
mpi::Comm BDM::ColComm() const EL_NO_EXCEPT
{ return this->Grid().VCComm(); }
template <typename T, Device D>
mpi::Comm BDM::RowComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template <typename T, Device D>
mpi::Comm BDM::PartialColComm() const EL_NO_EXCEPT
{ return this->Grid().MCComm(); }
template <typename T, Device D>
mpi::Comm BDM::PartialUnionColComm() const EL_NO_EXCEPT
{ return this->Grid().MRComm(); }

template <typename T, Device D>
mpi::Comm BDM::PartialRowComm() const EL_NO_EXCEPT
{ return this->RowComm(); }
template <typename T, Device D>
mpi::Comm BDM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template <typename T, Device D>
int BDM::DistSize() const EL_NO_EXCEPT
{ return this->Grid().VCSize(); }
template <typename T, Device D>
int BDM::CrossSize() const EL_NO_EXCEPT
{ return 1; }
template <typename T, Device D>
int BDM::RedundantSize() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int BDM::ColStride() const EL_NO_EXCEPT
{ return this->Grid().VCSize(); }
template <typename T, Device D>
int BDM::RowStride() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int BDM::PartialColStride() const EL_NO_EXCEPT
{ return this->Grid().MCSize(); }
template <typename T, Device D>
int BDM::PartialUnionColStride() const EL_NO_EXCEPT
{ return this->Grid().MRSize(); }

template <typename T, Device D>
int BDM::PartialRowStride() const EL_NO_EXCEPT
{ return this->RowStride(); }
template <typename T, Device D>
int BDM::PartialUnionRowStride() const EL_NO_EXCEPT
{ return 1; }

template <typename T, Device D>
int BDM::DistRank() const EL_NO_EXCEPT
{ return this->Grid().VCRank(); }
template <typename T, Device D>
int BDM::CrossRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }
template <typename T, Device D>
int BDM::RedundantRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }

template <typename T, Device D>
int BDM::ColRank() const EL_NO_EXCEPT
{ return this->Grid().VCRank(); }
template <typename T, Device D>
int BDM::RowRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }

template <typename T, Device D>
int BDM::PartialColRank() const EL_NO_EXCEPT
{ return this->Grid().MCRank(); }
template <typename T, Device D>
int BDM::PartialUnionColRank() const EL_NO_EXCEPT
{ return this->Grid().MRRank(); }

template <typename T, Device D>
int BDM::PartialRowRank() const EL_NO_EXCEPT
{ return this->RowRank(); }
template <typename T, Device D>
int BDM::PartialUnionRowRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK,Device::CPU>::DistMatrix \
  ( const DistMatrix<T,U,V,BLOCK,Device::CPU>& A );
#define OTHER(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK,Device::CPU>::DistMatrix \
  ( const DistMatrix<T,U,V>& A ); \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK,Device::CPU>& \
           DistMatrix<T,COLDIST,ROWDIST,BLOCK,Device::CPU>::operator= \
           ( const DistMatrix<T,U,V>& A )
#define BOTH(T,U,V) \
  SELF(T,U,V) \
  OTHER(T,U,V)
#define PROTO(T) \
  template class DistMatrix<T,COLDIST,ROWDIST,BLOCK,Device::CPU>; \
  BOTH( T,CIRC,CIRC); \
  BOTH( T,MC,  MR  ); \
  BOTH( T,MC,  STAR); \
  BOTH( T,MD,  STAR); \
  BOTH( T,MR,  MC  ); \
  BOTH( T,MR,  STAR); \
  BOTH( T,STAR,MC  ); \
  BOTH( T,STAR,MD  ); \
  BOTH( T,STAR,MR  ); \
  BOTH( T,STAR,STAR); \
  BOTH( T,STAR,VC  ); \
  BOTH( T,STAR,VR  ); \
  OTHER(T,VC,  STAR); \
  BOTH( T,VR,  STAR);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

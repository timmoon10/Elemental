/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_INVERSE_HPD_CHOLESKYUVAR2_HPP
#define EL_INVERSE_HPD_CHOLESKYUVAR2_HPP

namespace El {
namespace hpd_inv {

// This approach is based upon the reordered Variant 2 algorithm from Fig. 9 in 
// Bientinesi et al.'s "Families of Algorithms Related to the Inversion of 
// a Symmetric Positive Definite Matrix".

template<typename F> 
inline void
CholeskyUVar2( Matrix<F>& A )
{
    DEBUG_ONLY(
        CSE cse("hpd_inv::CholeskyUVar2");
        if( A.Height() != A.Width() )
            LogicError("Nonsquare matrices cannot be triangular");
    )

    const Int n = A.Height();
    const Int bsize = Blocksize();
    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(n-k,bsize);
        const Range<Int> ind0( 0, k ), ind1( k, k+nb ), ind2( k+nb, n );

        auto A00 = A( ind0, ind0 );
        auto A01 = A( ind0, ind1 );
        auto A02 = A( ind0, ind2 );
        auto A11 = A( ind1, ind1 );
        auto A12 = A( ind1, ind2 );
        auto A22 = A( ind2, ind2 );

        Cholesky( UPPER, A11 );
        Trsm( RIGHT, UPPER, NORMAL, NON_UNIT, F(1), A11, A01 );
        Trsm( LEFT, UPPER, ADJOINT, NON_UNIT, F(1), A11, A12 );
        Herk( UPPER, NORMAL, Base<F>(1), A01, Base<F>(1), A00 );
        Gemm( NORMAL, NORMAL, F(-1), A01, A12, F(1), A02 );
        Herk( UPPER, ADJOINT, Base<F>(-1), A12, Base<F>(1), A22 );
        Trsm( RIGHT, UPPER, ADJOINT, NON_UNIT, F(1), A11, A01 );
        Trsm( LEFT, UPPER, NORMAL, NON_UNIT, F(-1), A11, A12 );
        TriangularInverse( UPPER, NON_UNIT, A11 );
        Trtrmm( UPPER, A11, true );
    }
}

template<typename F> 
inline void
CholeskyUVar2( ElementalMatrix<F>& APre )
{
    DEBUG_ONLY(
        CSE cse("hpd_inv::CholeskyUVar2");
        if( APre.Height() != APre.Width() )
            LogicError("Nonsquare matrices cannot be triangular");
    )

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    const Grid& g = A.Grid();
    DistMatrix<F,STAR,STAR> A11_STAR_STAR(g);
    DistMatrix<F,VC,  STAR> A01_VC_STAR(g);
    DistMatrix<F,VR,  STAR> A01_VR_STAR(g);
    DistMatrix<F,STAR,VR  > A12_STAR_VR(g);
    DistMatrix<F,STAR,MC  > A01Trans_STAR_MC(g);
    DistMatrix<F,MR,  STAR> A01_MR_STAR(g);
    DistMatrix<F,STAR,MR  > A01Adj_STAR_MR(g);
    DistMatrix<F,STAR,MR  > A12_STAR_MR(g);
    DistMatrix<F,STAR,MC  > A12_STAR_MC(g);

    const Int n = A.Height();
    const Int bsize = Blocksize();
    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(n-k,bsize);
        const Range<Int> ind0( 0, k ), ind1( k, k+nb ), ind2( k+nb, n );

        auto A00 = A( ind0, ind0 );
        auto A01 = A( ind0, ind1 );
        auto A02 = A( ind0, ind2 );
        auto A11 = A( ind1, ind1 );
        auto A12 = A( ind1, ind2 );
        auto A22 = A( ind2, ind2 );

        A11_STAR_STAR = A11;
        Cholesky( UPPER, A11_STAR_STAR );

        A01_VC_STAR.AlignWith( A00 );
        A01_VC_STAR = A01;
        LocalTrsm
        ( RIGHT, UPPER, NORMAL, NON_UNIT, F(1), A11_STAR_STAR, A01_VC_STAR );

        A12_STAR_VR.AlignWith( A02 );
        A12_STAR_VR = A12;
        LocalTrsm
        ( LEFT, UPPER, ADJOINT, NON_UNIT, F(1), A11_STAR_STAR, A12_STAR_VR );

        A01Trans_STAR_MC.AlignWith( A00 );
        Transpose( A01_VC_STAR, A01Trans_STAR_MC );
        A01_VR_STAR.AlignWith( A00 );
        A01_VR_STAR = A01_VC_STAR;
        A01Adj_STAR_MR.AlignWith( A00 );
        Adjoint( A01_VR_STAR, A01Adj_STAR_MR );
        LocalTrrk
        ( UPPER, TRANSPOSE,
          F(1), A01Trans_STAR_MC, A01Adj_STAR_MR, F(1), A00 );

        A12_STAR_MR.AlignWith( A02 );
        A12_STAR_MR = A12_STAR_VR;
        LocalGemm
        ( TRANSPOSE, NORMAL, F(-1), A01Trans_STAR_MC, A12_STAR_MR, F(1), A02 );

        A12_STAR_MC.AlignWith( A22 );
        A12_STAR_MC = A12_STAR_VR;
        LocalTrrk
        ( UPPER, ADJOINT,
          F(-1), A12_STAR_MC, A12_STAR_MR, F(1), A22 );

        LocalTrsm
        ( RIGHT, UPPER, ADJOINT, NON_UNIT, F(1), A11_STAR_STAR, A01_VC_STAR );
        A01 = A01_VC_STAR;

        LocalTrsm
        ( LEFT, UPPER, NORMAL, NON_UNIT, F(-1), A11_STAR_STAR, A12_STAR_VR );
        A12 = A12_STAR_VR;

        LocalTriangularInverse( UPPER, NON_UNIT, A11_STAR_STAR );

        Trtrmm( UPPER, A11_STAR_STAR, true );
        A11 = A11_STAR_STAR;
    }
}

} // namespace hpd_inv
} // namespace El

#endif // ifndef EL_INVERSE_HPD_CHOLESKYUVAR2_HPP

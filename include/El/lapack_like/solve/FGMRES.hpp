/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_SOLVE_FGMRES_HPP
#define EL_SOLVE_FGMRES_HPP

// The pseudocode for Flexible GMRES can be found in "Algorithm 2.2" in
//   Yousef Saad
//   "A flexible inner-outer preconditioned GMRES algorithm"
//   SIAM J. Sci. Comput., Vol. 14, No. 2, pp. 461--469, 1993.

// Add support for promotion?

namespace El {

namespace fgmres {

// In what follows, 'applyA' should be a function of the form
//
//   void applyA
//   ( Field alpha, const Matrix<Field>& x, Field beta, Matrix<Field>& y )
//
// and overwrite y := alpha A x + beta y. However, 'precond' should have the
// form
//
//   void precond( Matrix<Field>& b )
//
// and overwrite b with an approximation of inv(A) b.
//

// TODO(poulson): Add support for an initial guess
template<typename Field,class ApplyAType,class PrecondType>
Int Single
( const ApplyAType& applyA,
  const PrecondType& precond,
        Matrix<Field>& b,
        Base<Field> relTol,
        Int restart,
        Int maxIts,
        bool progress )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( b.Width() != 1 )
          LogicError("Expected a single right-hand side");
    )

    // A z_j and A x_0
    const bool saveProducts = true;
    const bool time = false;

    typedef Base<Field> Real;
    const Int n = b.Height();
    Timer iterTimer;

    // x := 0
    // ======
    Matrix<Field> x;
    Zeros( x, n, 1 );

    Matrix<Field> Ax0;
    if( saveProducts )
    {
        // A x_0 := 0
        // ==========
        Zeros( Ax0, n, 1 );
    }

    // w := b (= b - A x_0)
    // ====================
    auto w = b;
    const Real origResidNorm = Nrm2( w );
    if( progress )
        Output("origResidNorm: ",origResidNorm);
    if( origResidNorm == Real(0) )
        return 0;

    // TODO: Constrain the maximum number of iterations

    Int iter=0;
    bool converged = false;
    Matrix<Real> cs;
    Matrix<Field> sn, H, t;
    Matrix<Field> x0, V, Z, AZ, q;
    while( !converged )
    {
        if( progress )
            Output("Starting FGMRES iteration ",iter);
        const Int indent = PushIndent();

        Zeros( cs, restart, 1 );
        Zeros( sn, restart, 1 );
        Zeros( H,  restart, restart );
        Zeros( V, n, restart );
        Zeros( Z, n, restart );
        if( saveProducts )
            Zeros( AZ, n, restart );

        // x0 := x
        // =======
        x0 = x;
        if( saveProducts && iter != 0 )
            Ax0 = q;

        // NOTE: w = b - A x already

        // beta := || w ||_2
        // =================
        const Real beta = Nrm2( w );

        // v0 := w / beta
        // ==============
        auto v0 = V( ALL, IR(0) );
        v0 = w;
        v0 *= 1/beta;

        // t := beta e_0
        // =============
        Zeros( t, restart+1, 1 );
        t(0) = beta;

        // Run one round of GMRES(restart)
        // ===============================
        for( Int j=0; j<restart; ++j )
        {
            if( progress )
                Output("Starting inner FGMRES iteration ",j);
            if( time )
                iterTimer.Start();
            const Int innerIndent = PushIndent();

            // z_j := inv(M) v_j
            // =================
            auto vj = V( ALL, IR(j) );
            auto zj = Z( ALL, IR(j) );
            zj = vj;
            precond( zj );

            // w := A z_j
            // ----------
            applyA( Field(1), zj, Field(0), w );
            if( saveProducts )
            {
                auto Azj = AZ( ALL, IR(j) );
                Azj = w;
            }

            // Run the j'th step of Arnoldi
            // ----------------------------
            for( Int i=0; i<=j; ++i )
            {
                // H(i,j) := v_i' w
                // ^^^^^^^^^^^^^^^^
                auto vi = V( ALL, IR(i) );
                H(i,j) = Dot(vi,w);

                // w := w - H(i,j) v_i
                // ^^^^^^^^^^^^^^^^^^^
                Axpy( -H(i,j), vi, w );
            }
            const Real delta = Nrm2( w );
            if( !limits::IsFinite(delta) )
                RuntimeError("Arnoldi step produced a non-finite number");
            if( delta == Real(0) )
                restart = j+1;
            if( j+1 != restart )
            {
                // v_{j+1} := w / delta
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^
                auto vjp1 = V( ALL, IR(j+1) );
                vjp1 = w;
                vjp1 *= 1/delta;
            }

            // Apply existing rotations to the new column of H
            // -----------------------------------------------
            for( Int i=0; i<j; ++i )
            {
                const Real& c = cs(i);
                const Field& s = sn(i);
                const Field sConj = Conj(s);
                const Field eta_i_j = H(i,j);
                const Field eta_ip1_j = H(i+1,j);
                H(i,  j) =  c    *eta_i_j + s*eta_ip1_j;
                H(i+1,j) = -sConj*eta_i_j + c*eta_ip1_j;
            }

            // Generate and apply a new rotation to both H and the rotated
            // beta*e_0 vector, t, then solve the minimum residual problem
            // -----------------------------------------------------------
            const Field eta_j_j = H(j,j);
            const Field eta_jp1_j = delta;
            if( !limits::IsFinite(RealPart(eta_j_j))   ||
                !limits::IsFinite(ImagPart(eta_j_j))   ||
                !limits::IsFinite(RealPart(eta_jp1_j)) ||
                !limits::IsFinite(ImagPart(eta_jp1_j)) )
                RuntimeError("Either H(j,j) or H(j+1,j) was not finite");
            Real c;
            Field s;
            Field rho = Givens( eta_j_j, eta_jp1_j, c, s );
            if( !limits::IsFinite(c) ||
                !limits::IsFinite(RealPart(s)) ||
                !limits::IsFinite(ImagPart(s)) ||
                !limits::IsFinite(RealPart(rho)) ||
                !limits::IsFinite(ImagPart(rho)) )
                RuntimeError("Givens rotation produced a non-finite number");
            H(j,j) = rho;
            cs(j) = c;
            sn(j) = s;
            // Apply the rotation to the rotated beta*e_0 vector
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            const Field sConj = Conj(s);
            const Field tau_j = t(j);
            const Field tau_jp1 = t(j+1);
            t(j)   =  c    *tau_j + s*tau_jp1;
            t(j+1) = -sConj*tau_j + c*tau_jp1;
            // Minimize the residual
            // ^^^^^^^^^^^^^^^^^^^^^
            auto tT = t( IR(0,j+1), ALL );
            auto HTL = H( IR(0,j+1), IR(0,j+1) );
            auto y = tT;
            Trsv( UPPER, NORMAL, NON_UNIT, HTL, y );
            // x := x0 + Zj y
            // ^^^^^^^^^^^^^^
            x = x0;
            auto Zj = Z( ALL, IR(0,j+1) );
            auto yj = y( IR(0,j+1), ALL );
            Gemv( NORMAL, Field(1), Zj, yj, Field(1), x );

            // w := b - A x
            // ------------
            w = b;
            if( saveProducts )
            {
                // q := Ax = Ax0 + A Z_j y_j
                // ^^^^^^^^^^^^^^^^^^^^^^^^^
                q = Ax0;
                auto AZj = AZ( ALL, IR(0,j+1) );
                Gemv( NORMAL, Field(1), AZj, yj, Field(1), q );

                // w := b - A x
                // ^^^^^^^^^^^^
                w -= q;
            }
            else
            {
                applyA( Field(-1), x, Field(1), w );
            }

            if( time )
                Output("iter took ",iterTimer.Stop()," secs");

            // Residual checks
            // ---------------
            const Real residNorm = Nrm2( w );
            if( !limits::IsFinite(residNorm) )
                RuntimeError("Residual norm was not finite");
            const Real relResidNorm = residNorm/origResidNorm;
            if( relResidNorm < relTol )
            {
                if( progress )
                    Output("converged with relative tolerance: ",relResidNorm);
                converged = true;
                ++iter;
                break;
            }
            else
            {
                if( progress )
                    Output
                    ("finished iteration ",iter," with relResidNorm=",
                     relResidNorm);
            }
            ++iter;
            if( iter == maxIts )
                RuntimeError("FGMRES did not converge");
            SetIndent( innerIndent );
        }
        SetIndent( indent );
    }
    b = x;
    return iter;
}

} // namespace fgmres

// TODO(poulson): Add support for an initial guess
template<typename Field,class ApplyAType,class PrecondType>
Int FGMRES
( const ApplyAType& applyA,
  const PrecondType& precond,
        Matrix<Field>& B,
        Base<Field> relTol,
        Int restart,
        Int maxIts,
        bool progress )
{
    EL_DEBUG_CSE
    Int mostIts = 0;
    const Int width = B.Width();
    for( Int j=0; j<width; ++j )
    {
        auto b = B( ALL, IR(j) );
        const Int its =
          fgmres::Single
          ( applyA, precond, b, relTol, restart, maxIts, progress );
        mostIts = Max(mostIts,its);
    }
    return mostIts;
}


} // namespace El

#endif // ifndef EL_SOLVE_FGMRES_HPP

/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_UPDATE_MOMENTUM_PEREZ_ELASTIC_H_
#define WARPX_PARTICLES_COLLISION_UPDATE_MOMENTUM_PEREZ_ELASTIC_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_Random.H>

#include <cmath>  // isnan() isinf()
#include <limits> // numeric_limits<float>::min()

/* \brief Update particle velocities according to
 *        F. Perez et al., Phys.Plasmas.19.083104 (2012),
 *        which is based on Nanbu's method, PhysRevE.55.4642 (1997).
 *        @param[in] bmax is max(Debye length, minimal interparticle distance).
 *        @param[in] L is the Coulomb log. A fixed L will be used if L > 0,
 *        otherwise L will be calculated based on the algorithm.
 *        @param[in] n12 = max(w1,w2)*min(N1,N2)/dV is the effective density used for s12
 *        @param[in] sigma_max is the maximum cross section based on mfp = atomic spacing
 *        used for the normalized scattering length s12 (see Sec. II.C of Perez et al.)
 *        To see if there are nan or inf updated velocities,
 *        compile with USE_ASSERTION=TRUE.
 *
 * Updates and corrections to the original publication are documented in
 * https://github.com/ECP-WarpX/WarpX/issues/429
 * https://github.com/ECP-WarpX/WarpX/files/3799803/main.pdf
 */

template <typename T_PR, typename T_R>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumPerezElastic (
    T_PR& u1x, T_PR& u1y, T_PR& u1z, T_PR& u2x, T_PR& u2y, T_PR& u2z,
    T_PR const q1, T_PR const m1, T_PR const w1,
    T_PR const q2, T_PR const m2, T_PR const w2,
    T_PR const n12, T_PR const sigma_max,
    T_PR const L, T_PR const bmax,
    T_R const dt, amrex::RandomEngine const& engine )
{

    T_PR constexpr inv_c2 = T_PR(1.0) / ( PhysConst::c * PhysConst::c );

    // Compute Lorentz factor gamma
    T_PR const gb1sq = (u1x*u1x + u1y*u1y + u1z*u1z)*inv_c2;
    T_PR const gb2sq = (u2x*u2x + u2y*u2y + u2z*u2z)*inv_c2;
    T_PR const g1 = std::sqrt( T_PR(1.0) + gb1sq );
    T_PR const g2 = std::sqrt( T_PR(1.0) + gb2sq );

    T_PR const diffx = u1x-u2x;
    T_PR const diffy = u1y-u2y;
    T_PR const diffz = u1z-u2z;
    T_PR const diffm = std::sqrt((diffx*diffx+diffy*diffy+diffz*diffz)*inv_c2);
    T_PR const summm = std::sqrt(gb1sq) + std::sqrt(gb2sq);
    // If g = u1 - u2 = 0, do not collide.
    // Or if the relative difference is less than 1.0e-10.
    if ( diffm < std::numeric_limits<T_PR>::min() || diffm/summm < 1.0e-10 ) { return; }

    // Compute momenta
    T_PR const p1x = u1x * m1;
    T_PR const p1y = u1y * m1;
    T_PR const p1z = u1z * m1;
    T_PR const p2x = u2x * m2;
    T_PR const p2y = u2y * m2;
    T_PR const p2z = u2z * m2;

    // Compute center-of-momentum (COM) velocity and gamma
    T_PR const mass_g = m1 * g1 + m2 * g2;
    T_PR const vcx    = (p1x+p2x) / mass_g;
    T_PR const vcy    = (p1y+p2y) / mass_g;
    T_PR const vcz    = (p1z+p2z) / mass_g;
    T_PR const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;
    T_PR const gc     = T_PR(1.0) / std::sqrt( T_PR(1.0) - vcms*inv_c2 );

    // Compute vc dot v1 and v2
    T_PR const vcDv1 = (vcx*u1x + vcy*u1y + vcz*u1z) / g1;
    T_PR const vcDv2 = (vcx*u2x + vcy*u2y + vcz*u2z) / g2;

    // Compute p1 star
    // lorentz_transform_factor = ( (gc-1.0)/vcms*vcDv1 - gc )*m1*g1
    // Rewrite by multiplying and dividing first term by (gc+1) to
    // avoid loss of precision when gc is close to 1
    T_PR const lorentz_transform_factor =
            ( gc*gc*vcDv1*inv_c2/(T_PR(1.0) + gc) - gc )*m1*g1;
    T_PR const p1sx = p1x + vcx*lorentz_transform_factor;
    T_PR const p1sy = p1y + vcy*lorentz_transform_factor;
    T_PR const p1sz = p1z + vcz*lorentz_transform_factor;
    T_PR const p1sm = std::sqrt( p1sx*p1sx + p1sy*p1sy + p1sz*p1sz );

    // Compute gamma star
    T_PR const g1s = ( T_PR(1.0) - vcDv1*inv_c2 )*gc*g1;
    T_PR const g2s = ( T_PR(1.0) - vcDv2*inv_c2 )*gc*g2;

    // Compute relative velocity in center-of-momentum frame
    // (not a Lorentz invariant quantity)
    T_PR const muRst = g1s*m1*g2s*m2/(g1s*m1 + g2s*m2);
    T_PR const vrelst = p1sm/muRst; // |v1s - v2s|

    // Compute invariant relative velocity in center-of-momentum frame
    // (Lorentz invariant quantity)
    T_PR const denom = T_PR(1.0) + p1sm*p1sm/(m1*g1s*m2*g2s)*inv_c2; // (1.0 - v1s*v2s/c^2)
    T_PR const vrelst_invar = vrelst/denom; // |v1s - v2s|/(1.0 - v1s*v2s/c^2)

    // Compute s12
    T_PR s12 = 0;
    if (p1sm > std::numeric_limits<T_PR>::min()) {

        // Writing b0 in a form that is directly analagous to the well-known non-relativistic form.
        // See Eq. 3.3.2 in Principles of Plasma Discharges and Material Processing by
        // M. A. Lieberman and A. J. Lichtenberg.
        // Note that b0 on Eq. 22 of Perez POP 19 (2012) is bmin = b0/2,
        // Note: there is a typo in Eq 22 of Perez, the last square is incorrect!
        // See the SMILEI documentation: https://smileipic.github.io/Smilei/Understand/collisions.html
        // and https://github.com/ECP-WarpX/WarpX/files/3799803/main.pdf from GitHub #429
        T_PR const b0 = amrex::Math::abs(q1*q2) /
                        (T_PR(2.0)*MathConst::pi*PhysConst::ep0*muRst*vrelst*vrelst_invar);


        // Compute the Coulomb log lnLmd first
        T_PR lnLmd;
        if ( L > T_PR(0.0) ) { lnLmd = L; }
        else
        {

            // Compute the minimum impact parameter from quantum: bqm = lDB/(4*pi) = hbar/(2*p1sm)
            // See NRL formulary. Also see "An introduction to the physics of the Coulomb logarithm,
            // with emphasis on quantum-mechanical effects", J. Plasma Phys. vol. 85 (2019). by J.A. Krommes.
            // Note: The formula in Perez 2012 and in Lee and More 1984 uses h rather than
            // hbar for bqm. If this is used, then the transition energy where bmin goes from classical
            // to quantum is only 2.5 eV for electrons; compared to 100 eV when using hbar.
            const T_PR bmin_qm = static_cast<T_PR>(PhysConst::hbar*0.5/p1sm);

            // Set the minimum impact parameter
            const T_PR bmin = amrex::max(bmin_qm, T_PR(0.5)*b0);

            // Compute the Coulomb log lnLmd
            lnLmd = amrex::max( T_PR(2.0),
                    T_PR(0.5)*std::log(T_PR(1.0) + bmax*bmax/(bmin*bmin)) );
        }

        // Compute s12 with sigma limited by sigma_max where mfp = atomic spacing
        // See https://github.com/user-attachments/files/16555064/CoulombScattering_s12.pdf
        // for a proof that this expression for s12 is the same as Eq. 9 of Perez 2012 when
        // sigma_eff = pi*b0^2*lnLmd
        const T_PR sigma_eff = amrex::min(T_PR(MathConst::pi)*b0*b0*lnLmd,sigma_max);
        s12 = sigma_eff * n12 * dt * vrelst * g1s*g2s/(g1*g2);

    }

    // Only modify momenta if s12 is non-zero
    if (s12 > std::numeric_limits<T_PR>::min()) {

        // Get random numbers
        T_PR r = amrex::Random(engine);

        // Compute scattering angle
        T_PR cosXs;
        T_PR sinXs;
        if ( s12 <= T_PR(0.1) )
        {
            while ( true )
            {
                cosXs = T_PR(1.0) + s12 * std::log(r);
                // Avoid the bug when r is too small such that cosXs < -1
                if ( cosXs >= T_PR(-1.0) ) { break; }
                r = amrex::Random(engine);
            }
        }
        else if ( s12 > T_PR(0.1) && s12 <= T_PR(3.0) )
        {
            T_PR const Ainv = static_cast<T_PR>(
                0.0056958 + 0.9560202*s12 - 0.508139*s12*s12 +
                0.47913906*s12*s12*s12 - 0.12788975*s12*s12*s12*s12 + 0.02389567*s12*s12*s12*s12*s12);
            cosXs = Ainv * std::log( std::exp(T_PR(-1.0)/Ainv) +
                    T_PR(2.0) * r * std::sinh(T_PR(1.0)/Ainv) );
        }
        else if ( s12 > T_PR(3.0) && s12 <= T_PR(6.0) )
        {
            T_PR const A = T_PR(3.0) * std::exp(-s12);
            cosXs = T_PR(1.0)/A * std::log( std::exp(-A) +
                    T_PR(2.0) * r * std::sinh(A) );
        }
        else
        {
            cosXs = T_PR(2.0) * r - T_PR(1.0);
        }
        sinXs = std::sqrt(T_PR(1.0) - cosXs*cosXs);

        // Get random azimuthal angle
        T_PR const phis = amrex::Random(engine) * T_PR(2.0) * MathConst::pi;
        T_PR const cosphis = std::cos(phis);
        T_PR const sinphis = std::sin(phis);

        // Compute post-collision momenta pfs in COM
        T_PR p1fsx;
        T_PR p1fsy;
        T_PR p1fsz;
        // p1sp is the p1s perpendicular
        T_PR p1sp = std::sqrt( p1sx*p1sx + p1sy*p1sy );
        // Make sure p1sp is not almost zero
        if ( p1sp > std::numeric_limits<T_PR>::min() )
        {
            p1fsx = ( p1sx*p1sz/p1sp ) * sinXs*cosphis +
                    ( p1sy*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sx           ) * cosXs;
            p1fsy = ( p1sy*p1sz/p1sp ) * sinXs*cosphis +
                    (-p1sx*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sy           ) * cosXs;
            p1fsz = (-p1sp           ) * sinXs*cosphis +
                    ( T_PR(0.0)      ) * sinXs*sinphis +
                    ( p1sz           ) * cosXs;
            // Note a negative sign is different from
            // Eq. (12) in Perez's paper,
            // but they are the same due to the random nature of phis.
        }
        else
        {
            // If the previous p1sp is almost zero
            // x->y  y->z  z->x
            // This set is equivalent to the one in Nanbu's paper
            p1sp = std::sqrt( p1sy*p1sy + p1sz*p1sz );
            p1fsy = ( p1sy*p1sx/p1sp ) * sinXs*cosphis +
                    ( p1sz*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sy           ) * cosXs;
            p1fsz = ( p1sz*p1sx/p1sp ) * sinXs*cosphis +
                    (-p1sy*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sz           ) * cosXs;
            p1fsx = (-p1sp           ) * sinXs*cosphis +
                    ( T_PR(0.0)      ) * sinXs*sinphis +
                    ( p1sx           ) * cosXs;
        }

        T_PR const p2fsx = -p1fsx;
        T_PR const p2fsy = -p1fsy;
        T_PR const p2fsz = -p1fsz;

        // Transform from COM to lab frame
        T_PR const vcDp1fs = vcx*p1fsx + vcy*p1fsy + vcz*p1fsz;
        T_PR const vcDp2fs = vcx*p2fsx + vcy*p2fsy + vcz*p2fsz;
        /* factor = (gc-1.0)/vcms; Rewrite to avoid subtraction losing precision when gc is close to 1 */
        T_PR const factor = gc*gc*inv_c2/(gc+T_PR(1.0));
        T_PR const factor1 = factor*vcDp1fs + m1*g1s*gc;
        T_PR const factor2 = factor*vcDp2fs + m2*g2s*gc;
        T_PR const p1fx = p1fsx + vcx * factor1;
        T_PR const p1fy = p1fsy + vcy * factor1;
        T_PR const p1fz = p1fsz + vcz * factor1;
        T_PR const p2fx = p2fsx + vcx * factor2;
        T_PR const p2fy = p2fsy + vcy * factor2;
        T_PR const p2fz = p2fsz + vcz * factor2;

        // Rejection method
        r = amrex::Random(engine);
        if ( w2 > r*amrex::max(w1, w2) )
        {
            u1x  = p1fx / m1;
            u1y  = p1fy / m1;
            u1z  = p1fz / m1;
        }
        r = amrex::Random(engine);
        if ( w1 > r*amrex::max(w1, w2) )
        {
            u2x  = p2fx / m2;
            u2y  = p2fy / m2;
            u2z  = p2fz / m2;
        }
#ifndef AMREX_USE_DPCPP
        AMREX_ASSERT(!std::isnan(u1x+u1y+u1z+u2x+u2y+u2z));
        AMREX_ASSERT(!std::isinf(u1x+u1y+u1z+u2x+u2y+u2z));
#endif

    } // if s > std::numeric_limits<T_PR>::min()

}

#endif // WARPX_PARTICLES_COLLISION_UPDATE_MOMENTUM_PEREZ_ELASTIC_H_

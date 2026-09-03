/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef LINEAR_BREIT_WHEELER_UTIL_H
#define LINEAR_BREIT_WHEELER_UTIL_H

#include "Particles/Collision/BinaryCollision/LinearBreitWheeler/LinearBreitWheelerCrossSection.H"

#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace {
    /**
     * \brief Given the momenta of two colliding macrophotons in a two-photon collision,
     * this function computes the momenta of the two product macroparticles (electron and positron).
     *
     * This is done by using the conservation of energy and momentum.
     * The polar angle of the products in the center-of-momentum frame is sampled
     * from the Breit-Wheeler differential cross section d\sigma/d(cos\theta) using
     * Newton-Raphson inversion of the CDF. The azimuthal angle is uniform in [0, 2\pi).
     *
     * @param[in] u1x_in normalized momentum of the first colliding macrophoton along x (in m.s^-1)
     * @param[in] u1y_in normalized momentum of the first colliding macrophoton along y (in m.s^-1)
     * @param[in] u1z_in normalized momentum of the first colliding macrophoton along z (in m.s^-1)
     * @param[in] u2x_in normalized momentum of the second colliding macrophoton along x (in m.s^-1)
     * @param[in] u2y_in normalized momentum of the second colliding macrophoton along y (in m.s^-1)
     * @param[in] u2z_in normalized momentum of the second colliding macrophoton along z (in m.s^-1)
     * @param[out] u1x_out normalized momentum of the first product macroparticle along x (in m.s^-1)
     * @param[out] u1y_out normalized momentum of the first product macroparticle along y (in m.s^-1)
     * @param[out] u1z_out normalized momentum of the first product macroparticle along z (in m.s^-1)
     * @param[out] u2x_out normalized momentum of the second product macroparticle along x (in m.s^-1)
     * @param[out] u2y_out normalized momentum of the second product macroparticle along y (in m.s^-1)
     * @param[out] u2z_out normalized momentum of the second product macroparticle along z (in m.s^-1)
     * @param[in] engine the random engine (used to calculate the angle of emission of the products)
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void LinearBreitWheelerComputeProductMomenta (
                            const amrex::ParticleReal& u1x_in,
                            const amrex::ParticleReal& u1y_in,
                            const amrex::ParticleReal& u1z_in,
                            const amrex::ParticleReal& u2x_in,
                            const amrex::ParticleReal& u2y_in,
                            const amrex::ParticleReal& u2z_in,
                            amrex::ParticleReal& u1x_out,
                            amrex::ParticleReal& u1y_out,
                            amrex::ParticleReal& u1z_out,
                            amrex::ParticleReal& u2x_out,
                            amrex::ParticleReal& u2y_out,
                            amrex::ParticleReal& u2z_out,
                            const amrex::RandomEngine& engine )
    {
        using namespace amrex::literals;
        using namespace amrex::Math;

        constexpr auto c2 = PhysConst::c2_v<amrex::ParticleReal>;
        constexpr auto inv_c2 = PhysConst::inv_c2_v<amrex::ParticleReal>;
        constexpr amrex::ParticleReal me = PhysConst::m_e;

        // Compute momenta
        const amrex::ParticleReal p1x_in = u1x_in * me;
        const amrex::ParticleReal p1y_in = u1y_in * me;
        const amrex::ParticleReal p1z_in = u1z_in * me;
        const amrex::ParticleReal p2x_in = u2x_in * me;
        const amrex::ParticleReal p2y_in = u2y_in * me;
        const amrex::ParticleReal p2z_in = u2z_in * me;
        const amrex::ParticleReal p1_in = std::sqrt(powi<2>(p1x_in)+powi<2>(p1y_in)+powi<2>(p1z_in));
        const amrex::ParticleReal p2_in = std::sqrt(powi<2>(p2x_in)+powi<2>(p2y_in)+powi<2>(p2z_in));

        // Compute cosine of the angle between the two photon momenta in the lab frame
        const amrex::ParticleReal cos_ang = (p1x_in*p2x_in+p1y_in*p2y_in+p1z_in*p2z_in)/(p1_in*p2_in);

        // Energy squared of each of the two colliding photons in the center of momentum frame,
        // calculated using the Lorentz invariance of the total four-momentum norm
        const amrex::ParticleReal E_star_sq = 0.5_prt*c2*p1_in*p2_in*(1._prt - cos_ang);

        // Square of the norm of the momentum of the products in the center of mass frame
        // Formula obtained by inverting E^2 = p^2*c^2 + m^2*c^4 in the COM frame for each particle:
        // p_star_sq = E_star_sq/c2 - me*me*c2;
        // The expression below is specifically written in a form that avoids returning
        // small negative numbers due to machine precision errors, for low-energy particles
        const amrex::ParticleReal E_ratio = std::sqrt(E_star_sq)/(2._prt*me*c2);
        const amrex::ParticleReal p_star_sq = me*me*c2 * ( powi<2>(2._prt*E_ratio) - 1._prt );

        // Compute momentum of first product in the center of mass frame.
        // The polar angle is sampled from the BW differential cross section
        // via Newton-Raphson inversion of the CDF.
        amrex::ParticleReal px_star, py_star, pz_star;

        // s = (E_CM / (2*me*c^2))^2 = (E_star / (me*c^2))^2
        const amrex::ParticleReal s = amrex::Math::powi<2>(2._prt * E_ratio);
        const amrex::ParticleReal one_minus_beta2 = 1._prt / s;
        const amrex::ParticleReal gamma_star = std::sqrt(s);
        const amrex::ParticleReal max_angular_rapidity = std::acosh(gamma_star);
        const amrex::ParticleReal beta = std::tanh(max_angular_rapidity);
        const amrex::ParticleReal u = amrex::Random(engine);
        amrex::ParticleReal angular_rapidity =
            (2._prt * u - 1._prt) * max_angular_rapidity;
        amrex::ParticleReal lower_bound = -max_angular_rapidity;
        amrex::ParticleReal upper_bound = max_angular_rapidity;
        const amrex::ParticleReal I1 = LBWIntegralOfDifferentialCrossSectionTransformed(
            beta, one_minus_beta2, max_angular_rapidity);

        // Inverse transform sampling: draw u ~ Uniform[0,1] and solve CDF(x) = u
        // for x = cos(theta). In order to resolve the forward peaks at high energy,
        // solve in the transformed coordinate y = atanh(beta*x). Let I(y) be the
        // indefinite integral of dsigma/dy.
        // Since dsigma/dy is even, I(y) is odd. Thus, at y_max = atanh(beta),
        // CDF(y) = 0.5 + 0.5*I(y)/I(y_max). Define
        // G(y) = 0.5 + 0.5*I(y)/I(y_max) - u = 0.
        // Use a safeguarded Newton iteration, falling back to bisection if the Newton
        // update leaves the current bracket.
        for (int iter = 0; iter < 20; ++iter)
        {
            const amrex::ParticleReal I = LBWIntegralOfDifferentialCrossSectionTransformed(
                beta, one_minus_beta2, angular_rapidity);
            const amrex::ParticleReal G = 0.5_prt + 0.5_prt * (I / I1) - u;
            if (amrex::Math::abs(G)
                <= 32._prt * std::numeric_limits<amrex::ParticleReal>::epsilon())
            {
                break;
            }

            if (G < 0._prt)
            {
                lower_bound = angular_rapidity;
            }
            else
            {
                upper_bound = angular_rapidity;
            }

            const amrex::ParticleReal G_prime =
                LBWDifferentialCrossSectionTransformed(
                    beta, one_minus_beta2, angular_rapidity) / (2._prt * I1);
            const amrex::ParticleReal newton_candidate = angular_rapidity - G / G_prime;
            const amrex::ParticleReal newton_step =
                amrex::Math::abs(newton_candidate - angular_rapidity);
            const amrex::ParticleReal newton_tolerance =
                std::numeric_limits<amrex::ParticleReal>::epsilon()
                * amrex::max(1._prt, amrex::Math::abs(angular_rapidity));
            if (newton_step <= newton_tolerance)
            {
                // The Newton update has converged to machine precision.
                break;
            }
            if (newton_candidate > lower_bound && newton_candidate < upper_bound)
            {
                angular_rapidity = newton_candidate;
            }
            else
            {
                angular_rapidity = 0.5_prt * (lower_bound + upper_bound);
            }
        }
        // Compute beta*cos(theta) and beta*sin(theta) directly from the angular
        // rapidity. In particular, forming sin(theta) as sqrt(1-cos(theta)^2)
        // loses all transverse momentum in single precision near the forward peaks.
        const amrex::ParticleReal beta_costh = std::tanh(angular_rapidity);
        const amrex::ParticleReal beta_sinth = std::sqrt(
            std::sinh(max_angular_rapidity - angular_rapidity)
            * std::sinh(max_angular_rapidity + angular_rapidity))
            / (std::cosh(max_angular_rapidity) * std::cosh(angular_rapidity));

        // Sample azimuthal angle uniformly in [0, 2pi)
        const amrex::ParticleReal phi = 2._prt * MathConst::pi * amrex::Random(engine);

        // Compute velocity of the center of momentum frame:
        // v_CM = (p1 + p2) / (|p1| + |p2|) * c
        const amrex::ParticleReal vcx = (p1x_in+p2x_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vcy = (p1y_in+p2y_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vcz = (p1z_in+p2z_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vc_sq = vcx*vcx + vcy*vcy + vcz*vcz;

        // Collision axis = n = direction of photon 1 in the CM frame.
        // Momentum of the first photon in the CM frame:
        // p1_CM = p1_lab + v_CM * [(gc-1)/v^2 * (v_CM \dot p1) − gc*|p1|/c]
        // n = p1_CM / |p1_CM|
        // When v_CM ~ 0, photon 1's direction is unchanged.
        // Otherwise, Lorentz-boost photon 1 to the CM frame.
        amrex::ParticleReal nx = p1x_in;
        amrex::ParticleReal ny = p1y_in;
        amrex::ParticleReal nz = p1z_in;
        amrex::ParticleReal gc = 1._prt;
        if (vc_sq > std::numeric_limits<amrex::ParticleReal>::min())
        {
            // Lorentz factor of the CM frame boost
            gc = 1._prt / std::sqrt(1._prt - vc_sq * inv_c2);
            const amrex::ParticleReal vc_dot_p1 = vcx*p1x_in + vcy*p1y_in + vcz*p1z_in;
            const amrex::ParticleReal bfact = (gc - 1._prt) / vc_sq * vc_dot_p1 - gc * p1_in / PhysConst::c;
            nx += vcx * bfact;
            ny += vcy * bfact;
            nz += vcz * bfact;
        }
        const amrex::ParticleReal n_mag = std::sqrt(nx*nx + ny*ny + nz*nz);
        nx /= n_mag;
        ny /= n_mag;
        nz /= n_mag;
        // Build orthonormal basis (e1,e2,n) with n along the collision axis:
        // if n is not near z, n \times (0,0,1) = (ny,-nx,0); otherwise, n \times (1,0,0) = (0,nz,-ny)
        amrex::ParticleReal e1x, e1y, e1z;
        if (amrex::Math::abs(nz) < 0.9_prt)
        {
            e1x =  ny;
            e1y = -nx;
            e1z =  0._prt;
        }
        else
        {
            e1x = 0._prt;
            e1y = nz;
            e1z = -ny;
        }
        const amrex::ParticleReal e1_mag = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
        e1x /= e1_mag;
        e1y /= e1_mag;
        e1z /= e1_mag;
        const amrex::ParticleReal e2x = ny*e1z - nz*e1y;
        const amrex::ParticleReal e2y = nz*e1x - nx*e1z;
        const amrex::ParticleReal e2z = nx*e1y - ny*e1x;

        // Momentum of the first product in the CM frame, with the BW
        // polar angle measured from the collision axis
        const amrex::ParticleReal p_star_scale = me * PhysConst::c * gamma_star;
        const amrex::ParticleReal p_star_perp1 =
            p_star_scale * beta_sinth * std::cos(phi);
        const amrex::ParticleReal p_star_perp2 =
            p_star_scale * beta_sinth * std::sin(phi);
        const amrex::ParticleReal p_star_par = p_star_scale * beta_costh;
        px_star = p_star_perp1*e1x + p_star_perp2*e2x + p_star_par*nx;
        py_star = p_star_perp1*e1y + p_star_perp2*e2y + p_star_par*ny;
        pz_star = p_star_perp1*e1z + p_star_perp2*e2z + p_star_par*nz;

        amrex::ParticleReal p1x_out, p1y_out, p1z_out;
        if ( vc_sq > std::numeric_limits<amrex::ParticleReal>::min() )
        {   // Convert momentum of first product to lab frame, using equation (13) of F. Perez et al.,
            // Phys.Plasmas.19.083104 (2012) (which is a regular Lorentz transformation)
            // p1_lab = p1_CM + v_CM * [(gc-1)/v^2 * (v_CM \dot p1_CM) + me*g_star*gc]
            // Lorentz factor of the product (electron/positron) in the CM frame
            const amrex::ParticleReal g_star = std::sqrt(1._prt + p_star_sq / (me*me*c2));
            const amrex::ParticleReal vc_dot_ps = vcx*px_star + vcy*py_star + vcz*pz_star;
            const amrex::ParticleReal factor0 = (gc-1._prt)/vc_sq;
            const amrex::ParticleReal factor = factor0*vc_dot_ps + me*g_star*gc;
            p1x_out = px_star + vcx * factor;
            p1y_out = py_star + vcy * factor;
            p1z_out = pz_star + vcz * factor;
        }
        else // If center of mass velocity is zero, we are already in the lab frame
        {
            p1x_out = px_star;
            p1y_out = py_star;
            p1z_out = pz_star;
        }

        // Compute momentum of electron/positron in lab frame, using total momentum conservation
        const amrex::ParticleReal p2x_out = p1x_in + p2x_in - p1x_out;
        const amrex::ParticleReal p2y_out = p1y_in + p2y_in - p1y_out;
        const amrex::ParticleReal p2z_out = p1z_in + p2z_in - p1z_out;

        // Compute the momentum of the product macroparticles
        u1x_out = p1x_out/me;
        u1y_out = p1y_out/me;
        u1z_out = p1z_out/me;
        u2x_out = p2x_out/me;
        u2y_out = p2y_out/me;
        u2z_out = p2z_out/me;
    }
}

#endif // LINEAR_BREIT_WHEELER_UTIL_H

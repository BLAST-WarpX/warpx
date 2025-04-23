/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef TWO_PRODUCT_UTIL_H
#define TWO_PRODUCT_UTIL_H

#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace {
    /**
     * \brief Given the momenta of two colliding macrophotons in a two-photon collision,
     * this function computes the momenta of the two product macroparticles (electron and positron).
     *
     * This is done by using the conservation of energy and momentum,
     * and by assuming isotropic emission of the products in the center-of-momentum frame
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
    void TwoProductComputeProductMomenta (
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

        constexpr amrex::ParticleReal c_sq = PhysConst::c * PhysConst::c;
        constexpr amrex::ParticleReal inv_csq = 1._prt / ( c_sq );
        constexpr amrex::ParticleReal me = PhysConst::m_e;
        auto constexpr pow2 = [](double const x) { return x*x; };
        constexpr auto one_half_pr = amrex::ParticleReal(1./2.);
        constexpr auto one_pr = amrex::ParticleReal(1.);

        // Compute momenta
        const amrex::ParticleReal p1x_in = u1x_in * me;
        const amrex::ParticleReal p1y_in = u1y_in * me;
        const amrex::ParticleReal p1z_in = u1z_in * me;
        const amrex::ParticleReal p2x_in = u2x_in * me;
        const amrex::ParticleReal p2y_in = u2y_in * me;
        const amrex::ParticleReal p2z_in = u2z_in * me;
        const amrex::ParticleReal p1_in = std::sqrt(pow2(p1x_in)+pow2(p1y_in)+pow2(p1z_in));
        const amrex::ParticleReal p2_in = std::sqrt(pow2(p2x_in)+pow2(p2y_in)+pow2(p2z_in));

        // Compute cosine of the angle between the two photon momenta in the lab frame
        const amrex::ParticleReal cos_ang = (p1x_in*p2x_in+p1y_in*p2y_in+p1z_in*p2z_in)/(p1_in*p2_in);

        // Energy squared of each of the two colliding photons in the center of momentum frame,
        // calculated using the Lorentz invariance of the total four-momentum norm
        const amrex::ParticleReal E_star_sq = one_half_pr*c_sq*p1_in*p2_in*(one_pr - cos_ang);

        // Square of the norm of the momentum of the products in the center of mass frame
        // Formula obtained by inverting E^2 = p^2*c^2 + m^2*c^4 in the COM frame for each particle:
        // p_star_sq = E_star_sq/c_sq - me*me*c_sq;
        // The expression below is specifically written in a form that avoids returning
        // small negative numbers due to machine precision errors, for low-energy particles
        const amrex::ParticleReal E_ratio = std::sqrt(E_star_sq)/(2._prt*me*c_sq);
        const amrex::ParticleReal p_star_sq = me*me*c_sq * ( pow2(2._prt*E_ratio) - 1._prt );

        // Compute momentum of first product in the center of mass frame, assuming isotropic
        // distribution
        amrex::ParticleReal px_star, py_star, pz_star;
        ParticleUtils::RandomizeVelocity(px_star, py_star, pz_star, std::sqrt(p_star_sq),
                                         engine);

        // Next step is to convert momenta to lab frame
        amrex::ParticleReal p1x_out, p1y_out, p1z_out;
        // Preliminary calculation: compute velocity of the center of momentum frame:
        // v = (p1 + p2) / | p1 + p2 | * c
        const amrex::ParticleReal vcx    = (p1x_in+p2x_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vcy    = (p1y_in+p2y_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vcz    = (p1z_in+p2z_in) * PhysConst::c / (p1_in + p2_in);
        const amrex::ParticleReal vc_sq  = vcx*vcx + vcy*vcy + vcz*vcz;

        // Convert momentum of first product to lab frame, using equation (13) of F. Perez et al.,
        // Phys.Plasmas.19.083104 (2012) (which is a regular Lorentz transformation)
        if ( vc_sq > std::numeric_limits<amrex::ParticleReal>::min() )
        {
            const amrex::ParticleReal gc = 1._prt / std::sqrt( 1._prt - vc_sq*inv_csq );
            const amrex::ParticleReal g_star = std::sqrt(1._prt + p_star_sq / (me*me*c_sq));
            const amrex::ParticleReal vcDps = vcx*px_star + vcy*py_star + vcz*pz_star;
            const amrex::ParticleReal factor0 = (gc-1._prt)/vc_sq;
            const amrex::ParticleReal factor = factor0*vcDps + me*g_star*gc;
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

#endif // TWO_PRODUCT_UTIL_H

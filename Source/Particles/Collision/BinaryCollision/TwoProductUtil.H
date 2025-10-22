/* Copyright 2022 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TWO_PRODUCT_UTIL_H
#define WARPX_TWO_PRODUCT_UTIL_H

#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace {
    /**
     * \brief Given the momenta of two colliding macroparticles in a two-product reaction,
     * this function computes the momenta of the two product macroparticles.
     *
     * This is done by using the conservation of energy and momentum.
     *
     * @param[in] u1x_in normalized momentum of the first colliding macroparticles along x (in m.s^-1)
     * @param[in] u1y_in normalized momentum of the first colliding macroparticles along y (in m.s^-1)
     * @param[in] u1z_in normalized momentum of the first colliding macroparticles along z (in m.s^-1)
     * @param[in] m1_in mass of the first colliding macroparticles
     * @param[in] u2x_in normalized momentum of the second colliding macroparticles along x (in m.s^-1)
     * @param[in] u2y_in normalized momentum of the second colliding macroparticles along y (in m.s^-1)
     * @param[in] u2z_in normalized momentum of the second colliding macroparticles along z (in m.s^-1)
     * @param[in] m2_in mass of the second colliding macroparticles
     * @param[out] u1x_out normalized momentum of the first product macroparticles along x (in m.s^-1)
     * @param[out] u1y_out normalized momentum of the first product macroparticles along y (in m.s^-1)
     * @param[out] u1z_out normalized momentum of the first product macroparticles along z (in m.s^-1)
     * @param[in] m1_out mass of the first product macroparticles
     * @param[out] u2x_out normalized momentum of the second product macroparticles along x (in m.s^-1)
     * @param[out] u2y_out normalized momentum of the second product macroparticles along y (in m.s^-1)
     * @param[out] u2z_out normalized momentum of the second product macroparticles along z (in m.s^-1)
     * @param[in] m2_out mass of the second product macroparticles
     * @param[in] E_reaction energy liberated during the reaction
     * @param[in] isotropic_scattering flag to indicate whether the particles are scattered isotropically.
     * If true, the particles are scattered isotropically in the center-of-mass frame.
     * If false, the particles momentum do not change direction, in the center-of-mass frame.
     * (but the magnitude of the momentum may change, esp. if E_reaction is non-zero)
     * @param[in] engine the random engine (used to calculate the angle of emission of the products)
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void TwoProductComputeProductMomenta (
                            const amrex::ParticleReal& u1x_in,
                            const amrex::ParticleReal& u1y_in,
                            const amrex::ParticleReal& u1z_in,
                            const amrex::ParticleReal& m1_in,
                            const amrex::ParticleReal& u2x_in,
                            const amrex::ParticleReal& u2y_in,
                            const amrex::ParticleReal& u2z_in,
                            const amrex::ParticleReal& m2_in,
                            amrex::ParticleReal& u1x_out,
                            amrex::ParticleReal& u1y_out,
                            amrex::ParticleReal& u1z_out,
                            const amrex::ParticleReal& m1_out,
                            amrex::ParticleReal& u2x_out,
                            amrex::ParticleReal& u2y_out,
                            amrex::ParticleReal& u2z_out,
                            const amrex::ParticleReal& m2_out,
                            const amrex::ParticleReal& E_reaction,
                            const bool isotropic_scattering,
                            const amrex::RandomEngine& engine )
    {
        using namespace amrex::literals;
        using namespace amrex::Math;

        constexpr amrex::ParticleReal c_sq = PhysConst::c * PhysConst::c;
        constexpr amrex::ParticleReal inv_csq = 1._prt / ( c_sq );
        // Rest energy of incident particles
        const amrex::ParticleReal E_rest_in = (m1_in + m2_in)*c_sq;
        // Rest energy of products
        const amrex::ParticleReal E_rest_out = (m1_out + m2_out)*c_sq;

        // Compute Lorentz factor gamma in the lab frame
        const amrex::ParticleReal g1_in = std::sqrt( 1._prt
            + (u1x_in*u1x_in + u1y_in*u1y_in + u1z_in*u1z_in)*inv_csq );
        const amrex::ParticleReal g2_in = std::sqrt( 1._prt
            + (u2x_in*u2x_in + u2y_in*u2y_in + u2z_in*u2z_in)*inv_csq );

        // Compute momenta
        const amrex::ParticleReal p1x_in = u1x_in * m1_in;
        const amrex::ParticleReal p1y_in = u1y_in * m1_in;
        const amrex::ParticleReal p1z_in = u1z_in * m1_in;
        const amrex::ParticleReal p2x_in = u2x_in * m2_in;
        const amrex::ParticleReal p2y_in = u2y_in * m2_in;
        const amrex::ParticleReal p2z_in = u2z_in * m2_in;
        // Square norm of the total (sum between the two particles) momenta in the lab frame
        const amrex::ParticleReal p_total_sq =  powi<2>(p1x_in+p2x_in) +
                                                powi<2>(p1y_in+p2y_in) +
                                                powi<2>(p1z_in+p2z_in);

        // Total energy of incident macroparticles in the lab frame
        const amrex::ParticleReal E_lab = (m1_in * g1_in + m2_in * g2_in) * c_sq;
        // Total energy squared of the reactants in the center of mass frame, calculated using the
        // Lorentz invariance of the four-momentum norm
        const amrex::ParticleReal E_star_sq = E_lab*E_lab - c_sq*p_total_sq;
        // Total energy squared of the products in the center of mass frame
        // In principle, the term - E_rest_in + E_rest_out + E_reaction is not needed and equal to
        // zero (i.e. the energy liberated during the reaction is equal to the mass difference). However,
        // due to possible inconsistencies in how the mass is defined in the code, it is
        // probably more robust to subtract the rest masses and to add the reaction energy to the
        // total kinetic energy.
        const amrex::ParticleReal E_star_f_sq = powi<2>(std::sqrt(E_star_sq)
                                                         - E_rest_in + E_rest_out + E_reaction);

        // Square of the norm of the momentum of the products in the center of mass frame
        // Formula obtained by inverting E^2 = p^2*c^2 + m^2*c^4 in the COM frame for each particle
        // The expression below is specifically written in a form that avoids returning
        // small negative numbers due to machine precision errors, for low-energy particles
        const amrex::ParticleReal E_ratio = std::sqrt(E_star_f_sq)/((m1_out + m2_out)*c_sq);
        const amrex::ParticleReal p_star_f_sq = m1_out*m2_out*c_sq * ( powi<2>(E_ratio) - 1._prt )
                + powi<2>(m1_out - m2_out)*c_sq*0.25_prt * powi<2>( E_ratio - 1._prt/E_ratio );

        // Preliminary calculation: compute center of mass velocity vc
        const amrex::ParticleReal mass_g = m1_in * g1_in + m2_in * g2_in;
        const amrex::ParticleReal vcx    = (p1x_in+p2x_in) / mass_g;
        const amrex::ParticleReal vcy    = (p1y_in+p2y_in) / mass_g;
        const amrex::ParticleReal vcz    = (p1z_in+p2z_in) / mass_g;
        const amrex::ParticleReal vc_sq   = vcx*vcx + vcy*vcy + vcz*vcz;
        const amrex::ParticleReal gc = 1._prt / std::sqrt( 1._prt - vc_sq*inv_csq );

        // Compute momentum of first product in the center of mass frame
        amrex::ParticleReal px_star_out, py_star_out, pz_star_out;
        if (isotropic_scattering) {
            // Assume isotropic scattering in the center-of-mass frame
            ParticleUtils::RandomizeVelocity(px_star_out, py_star_out, pz_star_out, std::sqrt(p_star_f_sq),
                                             engine);
        }
        else {
            // Assume that the momentum of the products has the same direction
            // as that of the incident particle in the center of mass frame

            amrex::ParticleReal p1x_star_in, p1y_star_in, p1z_star_in;
            if ( vc_sq > std::numeric_limits<amrex::ParticleReal>::min() )
            {
                // Convert momentum of first incident particle to lab frame, using equation (2)
                // of F. Perez et al., Phys.Plasmas.19.083104 (2012)
                const amrex::ParticleReal vcDps = vcx*p1x_in + vcy*p1y_in + vcz*p1z_in;
                const amrex::ParticleReal factor0 = (gc-1._prt)/vc_sq;
                const amrex::ParticleReal factor = factor0*vcDps - m1_in*g1_in*gc;
                p1x_star_in = p1x_in + vcx * factor;
                p1y_star_in = p1y_in + vcy * factor;
                p1z_star_in = p1z_in + vcz * factor;
            }
            else // If center of mass velocity is zero, we are already in the lab frame
            {
                p1x_star_in = p1x_in;
                p1y_star_in = p1y_in;
                p1z_star_in = p1z_in;
            }
            amrex::ParticleReal p1_star_in = std::sqrt(p1x_star_in*p1x_star_in + p1y_star_in*p1y_star_in + p1z_star_in*p1z_star_in);

            // Momentum of the first product in the center of mass frame:
            // same direction as the incident, but scaled to have the magnitude sqrt(p_star_f_sq)
            amrex::ParticleReal scaling = std::sqrt(p_star_f_sq)/p1_star_in;
            px_star_out = p1x_star_in * scaling;
            py_star_out = p1y_star_in * scaling;
            pz_star_out = p1z_star_in * scaling;
        }

        // Next step is to convert momenta to lab frame
        amrex::ParticleReal p1x_out, p1y_out, p1z_out;
        // Convert momentum of first product to lab frame, using equation (13)
        // of F. Perez et al., Phys.Plasmas.19.083104 (2012)
        if ( vc_sq > std::numeric_limits<amrex::ParticleReal>::min() )
        {
            const amrex::ParticleReal g_star = std::sqrt(1._prt + p_star_f_sq / (m1_out*m1_out*c_sq));
            const amrex::ParticleReal vcDps = vcx*px_star_out + vcy*py_star_out + vcz*pz_star_out;
            const amrex::ParticleReal factor0 = (gc-1._prt)/vc_sq;
            const amrex::ParticleReal factor = factor0*vcDps + m1_out*g_star*gc;
            p1x_out = px_star_out + vcx * factor;
            p1y_out = py_star_out + vcy * factor;
            p1z_out = pz_star_out + vcz * factor;
        }
        else // If center of mass velocity is zero, we are already in the lab frame
        {
            p1x_out = px_star_out;
            p1y_out = py_star_out;
            p1z_out = pz_star_out;
        }

        // Compute momentum of the second product in lab frame, using total momentum conservation
        const amrex::ParticleReal p2x_out = p1x_in + p2x_in - p1x_out;
        const amrex::ParticleReal p2y_out = p1y_in + p2y_in - p1y_out;
        const amrex::ParticleReal p2z_out = p1z_in + p2z_in - p1z_out;

        // Compute the momentum of the product macroparticles
        u1x_out = p1x_out/m1_out;
        u1y_out = p1y_out/m1_out;
        u1z_out = p1z_out/m1_out;
        u2x_out = p2x_out/m2_out;
        u2y_out = p2y_out/m2_out;
        u2z_out = p2z_out/m2_out;
    }
}

#endif // WARPX_TWO_PRODUCT_UTIL_H

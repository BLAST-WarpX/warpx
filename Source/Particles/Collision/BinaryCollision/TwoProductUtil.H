/* Copyright 2022 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TWO_PRODUCT_UTIL_H
#define WARPX_TWO_PRODUCT_UTIL_H

#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <ablastr/warn_manager/WarnManager.H>

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
     * @param[in] scattering_angle_model the scattering angle distribution to use
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
                            ScatteringAngleModel scattering_angle_model,
                            const amrex::RandomEngine& engine )
    {
        using namespace amrex::literals;
        using namespace amrex::Math;

        // Squared SI momenta and energies underflow in single precision. Keep the
        // dimensionful kinematic intermediates in double precision.
        constexpr double c_d = PhysConst::c_v<double>;
        constexpr double c2_d = PhysConst::c2_v<double>;
        constexpr double inv_c2_d = PhysConst::inv_c2_v<double>;

        // Compute momenta
        // Note: for massless particles, the momentum is normalized to the electron mass
        constexpr amrex::ParticleReal me = PhysConst::m_e;
        constexpr auto me_d = static_cast<double>(me);
        const amrex::ParticleReal p1x_in = (m1_in == 0) ? me*u1x_in : m1_in*u1x_in;
        const amrex::ParticleReal p1y_in = (m1_in == 0) ? me*u1y_in : m1_in*u1y_in;
        const amrex::ParticleReal p1z_in = (m1_in == 0) ? me*u1z_in : m1_in*u1z_in;
        const amrex::ParticleReal p2x_in = (m2_in == 0) ? me*u2x_in : m2_in*u2x_in;
        const amrex::ParticleReal p2y_in = (m2_in == 0) ? me*u2y_in : m2_in*u2y_in;
        const amrex::ParticleReal p2z_in = (m2_in == 0) ? me*u2z_in : m2_in*u2z_in;

        // Keep dimensionful kinematic intermediates in double precision.
        const auto m1_in_d = static_cast<double>(m1_in);
        const auto m2_in_d = static_cast<double>(m2_in);
        const auto m1_out_d = static_cast<double>(m1_out);
        const auto m2_out_d = static_cast<double>(m2_out);
        const auto p1x_in_d = static_cast<double>(p1x_in);
        const auto p1y_in_d = static_cast<double>(p1y_in);
        const auto p1z_in_d = static_cast<double>(p1z_in);
        const auto p2x_in_d = static_cast<double>(p2x_in);
        const auto p2y_in_d = static_cast<double>(p2y_in);
        const auto p2z_in_d = static_cast<double>(p2z_in);
        const auto E_reaction_d = static_cast<double>(E_reaction);

        // Compute 0th component of the 4-momentum of the incident particles
        const double p1t_in = std::sqrt(
            m1_in_d*m1_in_d*c2_d + p1x_in_d*p1x_in_d + p1y_in_d*p1y_in_d + p1z_in_d*p1z_in_d);
        const double p2t_in = std::sqrt(
            m2_in_d*m2_in_d*c2_d + p2x_in_d*p2x_in_d + p2y_in_d*p2y_in_d + p2z_in_d*p2z_in_d);

        // Square norm of the total (sum between the two particles) momenta in the lab frame
        const double p_total_sq_d =  powi<2>(p1x_in_d+p2x_in_d) +
                                     powi<2>(p1y_in_d+p2y_in_d) +
                                     powi<2>(p1z_in_d+p2z_in_d);

        // Total energy of incident macroparticles in the lab frame
        const double E_lab_d = (p1t_in + p2t_in) * c_d;
        // Total energy squared of the reactants in the center of mass frame, calculated using the
        // Lorentz invariance of the four-momentum norm
        const double E_star_sq_d = E_lab_d*E_lab_d - c2_d*p_total_sq_d;
        const double E_star_sq_pos_d = amrex::max(E_star_sq_d, 0.0);
        // Total energy squared of the products in the center of mass frame
        // In principle, the term - E_rest_in + E_rest_out + E_reaction is not needed and equal to
        // zero (i.e. the energy liberated during the reaction is equal to the mass difference). However,
        // due to possible inconsistencies in how the mass is defined in the code, it is
        // probably more robust to subtract the rest masses and to add the reaction energy to the
        // total kinetic energy.
        const double E_rest_in_d = (m1_in_d + m2_in_d)*c2_d;
        const double E_rest_out_d = (m1_out_d + m2_out_d)*c2_d;
        const double E_star_f_d = amrex::max(std::sqrt(E_star_sq_pos_d) - E_rest_in_d + E_rest_out_d + E_reaction_d, 0.0);

        // Square of the norm of the momentum of the products in the center of mass frame
        // Formula obtained by inverting E^2 = p^2*c^2 + m^2*c^4 in the COM frame for each particle
        // The expression below is specifically written in a form that avoids returning
        // small negative numbers due to machine precision errors, for low-energy particles
        const double E_ratio_d = E_star_f_d/E_rest_out_d;
        double p_star_f_sq_d = 0.0;
        // E_ratio <= 1 means the COM energy does not reach the combined rest mass of
        // the products, i.e. this pair is below the reaction threshold.
        if (E_ratio_d <= 1.0 && E_star_f_d > 0.0) {
            AMREX_IF_ON_HOST((
                ablastr::warn_manager::WMRecordWarning("BinaryCollision::TwoProductComputeProductMomenta",
                    "COM energy is below the product rest mass (pair selected below the reaction threshold)",
                    ablastr::warn_manager::WarnPriority::high);
            ))
        }
        if (E_ratio_d > 1.0)
        {
            p_star_f_sq_d = m1_out_d*m2_out_d*c2_d * (powi<2>(E_ratio_d) - 1.0) + powi<2>(m1_out_d - m2_out_d)*c2_d*0.25 * powi<2>(E_ratio_d - 1.0/E_ratio_d);
        }
        const double p_star_f_sq_clamped_d = amrex::max(p_star_f_sq_d, 0.0);
        const double p_star_f_d = std::sqrt(p_star_f_sq_clamped_d);

        // Preliminary calculation: compute center of mass velocity
        const double pt_d = p1t_in + p2t_in;
        double vcx_d = 0.0;
        double vcy_d = 0.0;
        double vcz_d = 0.0;
        if (pt_d > std::numeric_limits<double>::min())
        {
            vcx_d = (p1x_in_d + p2x_in_d) * c_d / pt_d;
            vcy_d = (p1y_in_d + p2y_in_d) * c_d / pt_d;
            vcz_d = (p1z_in_d + p2z_in_d) * c_d / pt_d;
        }
        const double vc_sq_d = vcx_d*vcx_d + vcy_d*vcy_d + vcz_d*vcz_d;
        const double one_minus_beta_sq_d = amrex::max(1.0 - vc_sq_d*inv_c2_d, std::numeric_limits<double>::epsilon());
        const double gc_d = 1.0 / std::sqrt(one_minus_beta_sq_d);

        // Compute momentum of first product in the center of mass frame
        amrex::ParticleReal px_star_out = 0.0_prt;
        amrex::ParticleReal py_star_out = 0.0_prt;
        amrex::ParticleReal pz_star_out = 0.0_prt;
        if (scattering_angle_model == ScatteringAngleModel::Isotropic) {
            // Isotropic scattering: the angle of emission of the products in the center of mass frame is random
            ParticleUtils::RandomizeVelocity(px_star_out, py_star_out, pz_star_out, static_cast<amrex::ParticleReal>(p_star_f_d),
                                             engine);
        }
        else if (scattering_angle_model == ScatteringAngleModel::Forward ||
                 scattering_angle_model == ScatteringAngleModel::Backward) {
            // Forward scattering: the products have the same direction as the incident particle in the center of mass frame
            // Backward scattering: the products have the opposite direction of the incident particle in the center of mass frame
            double p1x_star_in_d;
            double p1y_star_in_d;
            double p1z_star_in_d;
            if ( vc_sq_d > std::numeric_limits<double>::min() )
            {
                // Convert momentum of first incident particle to lab frame, using equation (2)
                // of F. Perez et al., Phys.Plasmas.19.083104 (2012)
                const double vcDps_d = vcx_d*p1x_in_d + vcy_d*p1y_in_d + vcz_d*p1z_in_d;
                const double factor0_d = (gc_d-1.0)/vc_sq_d;
                const double factor_d = factor0_d*vcDps_d - p1t_in*gc_d/c_d;
                p1x_star_in_d = p1x_in_d + vcx_d * factor_d;
                p1y_star_in_d = p1y_in_d + vcy_d * factor_d;
                p1z_star_in_d = p1z_in_d + vcz_d * factor_d;
            }
            else // If center of mass velocity is zero, we are already in the lab frame
            {
                p1x_star_in_d = p1x_in_d;
                p1y_star_in_d = p1y_in_d;
                p1z_star_in_d = p1z_in_d;
            }
            const double p1_star_in_d = std::sqrt(p1x_star_in_d*p1x_star_in_d + p1y_star_in_d*p1y_star_in_d + p1z_star_in_d*p1z_star_in_d);

            // Momentum of the first product in the center of mass frame:
            // same (forward) or opposite (backward) direction as the incident, scaled to have
            // the magnitude sqrt(p_star_f_sq)
            const double sign_d = (scattering_angle_model == ScatteringAngleModel::Backward) ? -1.0 : 1.0;
            if (p1_star_in_d > std::numeric_limits<double>::min())
            {
                const double scaling_d = sign_d*p_star_f_d/p1_star_in_d;
                px_star_out = static_cast<amrex::ParticleReal>(p1x_star_in_d * scaling_d);
                py_star_out = static_cast<amrex::ParticleReal>(p1y_star_in_d * scaling_d);
                pz_star_out = static_cast<amrex::ParticleReal>(p1z_star_in_d * scaling_d);
            }
            else
            {
                // Incident direction is undefined in the COM frame; fall back to an
                // isotropic direction while conserving the outgoing COM momentum magnitude.
                ParticleUtils::RandomizeVelocity(px_star_out, py_star_out, pz_star_out,
                    static_cast<amrex::ParticleReal>(p_star_f_d), engine);
            }
        }

        const auto px_star_out_d = static_cast<double>(px_star_out);
        const auto py_star_out_d = static_cast<double>(py_star_out);
        const auto pz_star_out_d = static_cast<double>(pz_star_out);

        // Next step is to convert momenta to lab frame
        double p1x_out_d;
        double p1y_out_d;
        double p1z_out_d;
        // Convert momentum of first product to lab frame, using equation (13)
        // of F. Perez et al., Phys.Plasmas.19.083104 (2012)
        if ( vc_sq_d > std::numeric_limits<double>::min() )
        {
            const double p1t_out_d = std::sqrt(m1_out_d*m1_out_d*c2_d + p_star_f_sq_clamped_d);
            const double vcDps_d = vcx_d*px_star_out_d + vcy_d*py_star_out_d + vcz_d*pz_star_out_d;
            const double factor0_d = (gc_d-1.0)/vc_sq_d;
            const double factor_d = factor0_d*vcDps_d + p1t_out_d*gc_d/c_d;
            p1x_out_d = px_star_out_d + vcx_d * factor_d;
            p1y_out_d = py_star_out_d + vcy_d * factor_d;
            p1z_out_d = pz_star_out_d + vcz_d * factor_d;
        }
        else // If center of mass velocity is zero, we are already in the lab frame
        {
            p1x_out_d = px_star_out_d;
            p1y_out_d = py_star_out_d;
            p1z_out_d = pz_star_out_d;
        }

        // Compute momentum of the second product in lab frame, using total momentum conservation
        const double p2x_out_d = p1x_in_d + p2x_in_d - p1x_out_d;
        const double p2y_out_d = p1y_in_d + p2y_in_d - p1y_out_d;
        const double p2z_out_d = p1z_in_d + p2z_in_d - p1z_out_d;

        // Compute the momentum of the product macroparticles
        // Note: for massless particles, the momentum is normalized to the electron mass
        const double m1_norm_d = (m1_out == 0) ? me_d : m1_out_d;
        const double m2_norm_d = (m2_out == 0) ? me_d : m2_out_d;
        u1x_out = static_cast<amrex::ParticleReal>(p1x_out_d/m1_norm_d);
        u1y_out = static_cast<amrex::ParticleReal>(p1y_out_d/m1_norm_d);
        u1z_out = static_cast<amrex::ParticleReal>(p1z_out_d/m1_norm_d);
        u2x_out = static_cast<amrex::ParticleReal>(p2x_out_d/m2_norm_d);
        u2y_out = static_cast<amrex::ParticleReal>(p2y_out_d/m2_norm_d);
        u2z_out = static_cast<amrex::ParticleReal>(p2z_out_d/m2_norm_d);
    }
}

#endif // WARPX_TWO_PRODUCT_UTIL_H

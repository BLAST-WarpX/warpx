/* Copyright 2026 Luca Fedeli Neil Zaim
 *
 * This file is part of WarpX.
 * Based on PICSAR QED.
 *
 * License: BSD-3-Clause-LBNL
 */

//This file contains functions to compute the quantum parameter
//'chi' for photons or electrons and positrons.

#ifndef ABLASTR_QED_CHI_FUNCTIONS
#define ABLASTR_QED_CHI_FUNCTIONS

#include "ablastr/constant.H"
#include "ablastr/sfqed/UnitConversion.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace ablastr::qed
{

    namespace detail
    {
        /**
        * This function returns the chi parameter for a photon
        *
        * @tparam unit_system unit system to be used for input&output
        * @param[in] tpx the x component of the momentum of the photon
        * @param[in] tpy the y component of the momentum of the photon
        * @param[in] tpz the z component of the momentum of the photon
        * @param[in] tex the x component of the electric field
        * @param[in] tey the y component of the electric field
        * @param[in] tez the z component of the electric field
        * @param[in] tbx the x component of the magnetic field
        * @param[in] tby the y component of the magnetic field
        * @param[in] tbz the z component of the magnetic field
        * @return the chi parameter
        */
        template<unit_system UnitSystem>
        AMREX_GPU_DEVICE
        AMREX_FORCE_INLINE
        amrex::ParticleReal compute_chi_photon(
            const amrex::ParticleReal tpx, const amrex::ParticleReal tpy, const amrex::ParticleReal tpz,
            const amrex::ParticleReal tex, const amrex::ParticleReal tey, const amrex::ParticleReal tez,
            const amrex::ParticleReal tbx, const amrex::ParticleReal tby, const amrex::ParticleReal tbz)
        {
            using namespace amrex;

            constexpr auto mom_conv = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(qq);};
            constexpr auto E_conv   = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(qq);};
            constexpr auto B_conv   = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(qq);};

            const auto px = mom_conv(tpx);
            const auto py = mom_conv(tpy);
            const auto pz = mom_conv(tpz);
            const auto ex = E_conv(tex);
            const auto ey = E_conv(tey);
            const auto ez = E_conv(tez);
            const auto bx = B_conv(tbx);
            const auto by = B_conv(tby);
            const auto bz = B_conv(tbz);

            const auto p2 = px*px + py*py + pz*pz;
            if(p2 == 0.0_prt){
                return 0.0_prt;
            }

            const auto norm_p = std::sqrt(p2);
            const auto one_over_p = 1.0_prt / norm_p;

            const auto nx = px * one_over_p;
            const auto ny = py * one_over_p;
            const auto nz = pz * one_over_p;

            constexpr auto one_over_m = static_cast<ParticleReal>(1.0/
                heaviside_lorentz_electron_rest_energy_v<double>);

            const auto gamma_phot = norm_p*one_over_m;

            const auto dotp = nx*ex + ny*ey + nz*ez;
            const auto em_eperp_x = ex - dotp*nx;
            const auto em_eperp_y = ey - dotp*ny;
            const auto em_eperp_z = ez - dotp*nz;

            const auto field_x = em_eperp_x + ny*bz - nz*by;
            const auto field_y = em_eperp_y + nz*bx - nx*bz;
            const auto field_z = em_eperp_z + nx*by - ny*bx;

            const auto field =
                std::sqrt(field_x*field_x + field_y*field_y + field_z*field_z);

            constexpr auto one_over_schwinger = static_cast<ParticleReal>(1.0/
                heaviside_lorentz_schwinger_field_v<double>);

            return gamma_phot*field*one_over_schwinger;
        }

        /**
        * This function returns the chi parameter for either an electron or a positron
        *
        * @tparam unit_system unit system to be used for input&output
        * @param[in] tpx the x component of the momentum of the electron (or the positron)
        * @param[in] tpy the y component of the momentum of the electron (or the positron)
        * @param[in] tpz the z component of the momentum of the electron (or the positron)
        * @param[in] tex the x component of the electric field
        * @param[in] tey the y component of the electric field
        * @param[in] tez the z component of the electric field
        * @param[in] tbx the x component of the magnetic field
        * @param[in] tby the y component of the magnetic field
        * @param[in] tbz the z component of the magnetic field
        * @return the chi parameter
        */
        template<unit_system UnitSystem>
        AMREX_GPU_DEVICE
        AMREX_FORCE_INLINE
        amrex::ParticleReal compute_chi_ele_pos(
            const amrex::ParticleReal tpx, const amrex::ParticleReal tpy, const amrex::ParticleReal tpz,
            const amrex::ParticleReal tex, const amrex::ParticleReal tey, const amrex::ParticleReal tez,
            const amrex::ParticleReal tbx, const amrex::ParticleReal tby, const amrex::ParticleReal tbz)
        {
            using namespace amrex;

            constexpr auto mom_conv = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::momentum>(qq);};
            constexpr auto E_conv   = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::E>(qq);};
            constexpr auto B_conv   = [](const auto qq){
                return convert_from_to<UnitSystem, unit_system::heaviside_lorentz, quantity::B>(qq);};

            const auto px = mom_conv(tpx);
            const auto py = mom_conv(tpy);
            const auto pz = mom_conv(tpz);
            const auto ex = E_conv(tex);
            const auto ey = E_conv(tey);
            const auto ez = E_conv(tez);
            const auto bx = B_conv(tbx);
            const auto by = B_conv(tby);
            const auto bz = B_conv(tbz);

            const auto p2 = px*px + py*py + pz*pz;
            if(p2 == 0.0_prt){
                return 0.0_prt;
            }

            const auto norm_p = std::sqrt(p2);
            const auto one_over_p = 1.0_prt / norm_p;

            const auto nx = px * one_over_p;
            const auto ny = py * one_over_p;
            const auto nz = pz * one_over_p;

            constexpr auto one_over_m2 = static_cast<ParticleReal>(1.0/
                heaviside_lorentz_electron_rest_energy_v<double>/
                heaviside_lorentz_electron_rest_energy_v<double>);

            const auto gamma_2 = 1.0_prt + p2*one_over_m2;
            const auto gamma = std::sqrt(gamma_2);
            const auto beta  = std::sqrt((gamma_2 - 1.0_prt)/gamma_2);
            const auto betax = beta*nx;
            const auto betay = beta*ny;
            const auto betaz = beta*nz;

            const auto beta_dot_e = betax*ex + betay*ey + betaz*ez;
            const auto beta_dot_e_2 = beta_dot_e*beta_dot_e;

            const auto e_plus_beta_cross_b_x = ex + betay*bz - betaz*by;
            const auto e_plus_beta_cross_b_y = ey + betaz*bx - betax*bz;
            const auto e_plus_beta_cross_b_z = ez + betax*by - betay*bx;
            const auto e_plus_beta_cross_b_2 =
                e_plus_beta_cross_b_x*e_plus_beta_cross_b_x +
                e_plus_beta_cross_b_y*e_plus_beta_cross_b_y +
                e_plus_beta_cross_b_z*e_plus_beta_cross_b_z;

            const auto field = std::sqrt(std::fabs(beta_dot_e_2-e_plus_beta_cross_b_2));

            constexpr auto one_over_schwinger = static_cast<ParticleReal>(1.0/
                heaviside_lorentz_schwinger_field_v<double>);

            return gamma*field*one_over_schwinger;
        }
    }

    /**
    * Computes the 'chi' parameter for photons.
    *
    * @param[in] px,py,pz components of momentum (SI units)
    * @param[in] ex,ey,ez components of electric field (SI units)
    * @param[in] bx,by,bz components of magnetic field (SI units)
    * @return chi parameter
    */
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real chi_photon(
        const amrex::ParticleReal px, const amrex::ParticleReal py,
        const amrex::ParticleReal pz, const amrex::ParticleReal ex,
        const amrex::ParticleReal ey, const amrex::ParticleReal ez,
        const amrex::ParticleReal bx, const amrex::ParticleReal by,
        const amrex::ParticleReal bz)
    {

        return detail::compute_chi_photon<ablastr::qed::unit_system::SI>(
            px, py, pz, ex, ey, ez, bx, by, bz);
    }

    /**
    * Computes the 'chi' parameter for electrons or positrons.
    *
    * @param[in] px,py,pz components of momentum (SI units)
    * @param[in] ex,ey,ez components of electric field (SI units)
    * @param[in] bx,by,bz components of magnetic field (SI units)
    * @return chi parameter
    */
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real chi_ele_pos(
        const amrex::ParticleReal px, const amrex::ParticleReal py,
        const amrex::ParticleReal pz, const amrex::ParticleReal ex,
        const amrex::ParticleReal ey, const amrex::ParticleReal ez,
        const amrex::ParticleReal bx, const amrex::ParticleReal by,
        const amrex::ParticleReal bz)
    {
            return detail::compute_chi_ele_pos<ablastr::qed::unit_system::SI>(
                px, py, pz, ex, ey, ez, bx, by, bz);
    }
}

#endif // ABLASTR_QED_CHI_FUNCTIONS

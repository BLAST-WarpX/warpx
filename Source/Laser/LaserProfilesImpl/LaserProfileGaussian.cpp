/* Copyright 2019 Axel Huebl, Luca Fedeli, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Laser/LaserProfiles.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace amrex;

void
WarpXLaserProfiles::GaussianLaserProfile::init (
    const amrex::ParmParse& ppl,
    CommonLaserParameters params)
{
    //Copy common params
    m_common_params = params;

    // Parse the properties of the Gaussian profile
    utils::parser::getWithParser(ppl, "profile_waist", m_params.waist);
    utils::parser::getWithParser(ppl, "profile_duration", m_params.duration);
    utils::parser::getWithParser(ppl, "profile_t_peak", m_params.t_peak);
    utils::parser::getWithParser(ppl, "profile_focal_distance", m_params.focal_distance);
    utils::parser::queryWithParser(ppl, "zeta", m_params.zeta);
    utils::parser::queryWithParser(ppl, "beta", m_params.beta);
    utils::parser::queryWithParser(ppl, "phi2", m_params.phi2);
    utils::parser::queryWithParser(ppl, "phi0", m_params.phi0);

    m_params.stc_direction = m_common_params.p_X;
    utils::parser::queryArrWithParser(ppl, "stc_direction", m_params.stc_direction);
    auto const s = 1.0_rt / std::sqrt(
        m_params.stc_direction[0]*m_params.stc_direction[0] +
        m_params.stc_direction[1]*m_params.stc_direction[1] +
        m_params.stc_direction[2]*m_params.stc_direction[2]);
    m_params.stc_direction = {
        m_params.stc_direction[0]*s,
        m_params.stc_direction[1]*s,
        m_params.stc_direction[2]*s };
    auto const dp2 =
        std::inner_product(
            m_common_params.nvec.begin(),
            m_common_params.nvec.end(),
            m_params.stc_direction.begin(), 0.0);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(dp2) < 1.0e-14,
        "stc_direction is not perpendicular to the laser plane vector");

    // Get angle between p_X and stc_direction
    // in 2d, stcs are in the simulation plane
#if defined(WARPX_DIM_3D)
    auto arg = m_params.stc_direction[0]*m_common_params.p_X[0] +
        m_params.stc_direction[1]*m_common_params.p_X[1] +
        m_params.stc_direction[2]*m_common_params.p_X[2];

    if (arg < -1.0_rt || arg > 1.0_rt) {
        m_params.theta_stc = 0._rt;
    } else {
        m_params.theta_stc = std::acos(arg);
    }
#else
    m_params.theta_stc = 0.;
#endif

}

/* \brief compute field amplitude for a Gaussian laser, at particles' position
 *
 * Both Xp and Yp are given in laser plane coordinate.
 * For each particle with position Xp and Yp, this routine computes the
 * amplitude of the laser electric field, stored in array amplitude.
 *
 * \param np: number of laser particles
 * \param Xp: pointer to first component of positions of laser particles
 * \param Yp: pointer to second component of positions of laser particles
 * \param t: Current physical time
 * \param amplitude: pointer to array of field amplitude.
 */
void
WarpXLaserProfiles::GaussianLaserProfile::fill_amplitude (
    const int np, Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    const Complex I(0,1);
    // Calculate a few factors which are independent of the macroparticle
    const Real k0 = 2._rt*MathConst::pi/m_common_params.wavelength;
    const Real inv_tau2 = 1._rt /(m_params.duration * m_params.duration);
    const Real oscillation_phase = k0 * PhysConst::c * ( t - m_params.t_peak ) + m_params.phi0;
    // The coefficients below contain info about Gouy phase,
    // laser diffraction, and phase front curvature
    const Complex diffract_factor =
        1._rt + I * m_params.focal_distance * 2._rt/
        ( k0 * m_params.waist * m_params.waist );
    const Complex inv_complex_waist_2 =
        1._rt /(m_params.waist*m_params.waist * diffract_factor );

    // Time stretching due to STCs and phi2 complex envelope
    // (1 if zeta=0, beta=0, phi2=0)
    const Complex stretch_factor = 1._rt + 4._rt *
        ((m_params.zeta+m_params.beta*m_params.focal_distance)*inv_tau2)
        * ((m_params.zeta+m_params.beta*m_params.focal_distance)*inv_complex_waist_2)
        + 2._rt*I*(m_params.phi2-m_params.beta*m_params.beta*k0*m_params.focal_distance)*inv_tau2;

    // Amplitude and monochromatic oscillations
    const Complex t_prefactor =
        m_common_params.e_max * amrex::exp( I * oscillation_phase );

    // Because diffract_factor is a complex, the code below takes into
    // account the impact of the dimensionality on both the Gouy phase
    // and the amplitude of the laser
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
    const Complex prefactor = t_prefactor / diffract_factor;
#elif defined(WARPX_DIM_XZ)
    const Complex prefactor = t_prefactor / amrex::sqrt(diffract_factor);
#else
    const Complex prefactor = t_prefactor;
#endif

    // Copy member variables to tmp copies for GPU runs.
    auto const tmp_profile_t_peak = m_params.t_peak;
    auto const tmp_beta = m_params.beta;
    auto const tmp_zeta = m_params.zeta;
    auto const tmp_theta_stc = m_params.theta_stc;
    auto const tmp_profile_focal_distance = m_params.focal_distance;
    // Loop through the macroparticle to calculate the proper amplitude
    amrex::ParallelFor(
        np,
        [=] AMREX_GPU_DEVICE (int i) {
            const Complex stc_exponent = 1._rt / stretch_factor * inv_tau2 *
                amrex::pow((t - tmp_profile_t_peak -
                    tmp_beta*k0*(Xp[i]*std::cos(tmp_theta_stc) + Yp[i]*std::sin(tmp_theta_stc)) -
                    2._rt *I*(Xp[i]*std::cos(tmp_theta_stc) + Yp[i]*std::sin(tmp_theta_stc))
                    *( tmp_zeta - tmp_beta*tmp_profile_focal_distance ) * inv_complex_waist_2),2);
            // stcfactor = everything but complex transverse envelope
            const Complex stcfactor = prefactor * amrex::exp( - stc_exponent );
            // Exp argument for transverse envelope
            const Complex exp_argument = - ( Xp[i]*Xp[i] + Yp[i]*Yp[i] ) * inv_complex_waist_2;
            // stcfactor + transverse envelope
            amplitude[i] = ( stcfactor * amrex::exp( exp_argument ) ).real();
        }
        );
}

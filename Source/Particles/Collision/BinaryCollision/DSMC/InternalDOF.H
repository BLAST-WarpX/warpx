/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_INTERNAL_DOF_H_
#define WARPX_PARTICLES_INTERNAL_DOF_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * \brief Helpers for molecular internal degrees of freedom (rotational / vibrational) and the
 * Borgnakke-Larsen (LB) translational<->internal energy exchange used by the DSMC collision kernel.
 *
 * All energies are in eV (matching the DSMC cross-section / collision-energy convention). Rotation is
 * treated classically (continuous, zeta_rot degrees of freedom, default 2 for a diatomic); vibration is
 * treated as a quantum harmonic oscillator (discrete levels E_vib = v * kB * theta_vib) so that it stays
 * frozen at low temperature (v=0) and activates correctly when the gas is hot.
 *
 * The translational LB exponent is a_tr = 3/2 - omega, where omega is the VHS viscosity-temperature
 * exponent (omega = 1/2 for hard spheres => a_tr = 1). This choice reproduces equipartition at
 * equilibrium: the mean internal energy E_int -> (zeta_int/2) kB T.  [Borgnakke & Larsen, JCP 18, 405 (1975); Bird 1994;
 * Boyd & Schwartzentruber 2017.]
 */
namespace InternalDOF {

    // Boltzmann constant in eV/K.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal kB_eV () {
        return static_cast<amrex::ParticleReal>(PhysConst::kb / PhysConst::q_e);
    }

    // Uniform random number in [0,1) as ParticleReal, clamped away from zero so that
    // log(U) and pow(U, .) stay finite.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal randomNonZero (amrex::RandomEngine const& engine) {
        const auto U = static_cast<amrex::ParticleReal>(amrex::Random(engine));
        return amrex::max(U, amrex::ParticleReal(1.e-30));
    }

    // Equilibrium single-molecule rotational energy [eV] for zeta_rot = 2 (continuous) at temperature
    // T [K]: the 2-DOF equilibrium distribution is exponential, E_rot = -kB T ln(U).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal sampleBoltzmannRotEnergy (amrex::ParticleReal T_K,
                                                  amrex::RandomEngine const& engine)
    {
        using namespace amrex::literals;
        if (T_K <= 0._prt) { return 0._prt; }
        const amrex::ParticleReal U = randomNonZero(engine);
        return -kB_eV() * T_K * std::log(U);
    }

    // Equilibrium single-molecule vibrational energy [eV] for a harmonic oscillator with characteristic
    // temperature theta_vib [K] at temperature T [K]. Level population is geometric:
    // P(v) = (1-x) x^v with x = exp(-theta_vib/T); the inverse-CDF gives v = floor( -(T/theta_vib) ln U ).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal sampleHarmonicVibEnergy (amrex::ParticleReal theta_vib_K, amrex::ParticleReal T_K,
                                                 amrex::RandomEngine const& engine)
    {
        using namespace amrex::literals;
        if (T_K <= 0._prt || theta_vib_K <= 0._prt) { return 0._prt; }
        const amrex::ParticleReal U = randomNonZero(engine);
        const amrex::ParticleReal v = amrex::Math::floor( -(T_K/theta_vib_K) * std::log(U) );
        return v * kB_eV() * theta_vib_K;
    }

    // Larsen-Borgnakke redistribution for a continuous internal mode with zeta_int DOF, given the total
    // collision energy E_c = E_trans + E_int [eV]. a_tr = 3/2 - omega is the translational exponent.
    // Returns the new internal energy [eV].
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal sampleLB_continuous (amrex::ParticleReal E_c, amrex::ParticleReal zeta_int,
                                             amrex::ParticleReal a_tr, amrex::RandomEngine const& engine)
    {
        using namespace amrex::literals;
        if (E_c <= 0._prt) { return 0._prt; }
        const amrex::ParticleReal a_int = zeta_int * 0.5_prt - 1.0_prt;
        // Closed-form inverse-CDF for zeta_int == 2 (a_int == 0): f(R) ∝ (1-R)^a_tr.
        if (a_int <= 0._prt) {
            const amrex::ParticleReal U = randomNonZero(engine);
            return E_c * (1.0_prt - std::pow(U, 1.0_prt/(a_tr + 1.0_prt)));
        }
        // General case: acceptance-rejection on f(R) = R^a_int (1-R)^a_tr, R = E_int/E_c.
        const amrex::ParticleReal Rmode = a_int / (a_int + a_tr);
        amrex::ParticleReal fmax = std::pow(Rmode, a_int) * std::pow(1.0_prt - Rmode, a_tr);
        fmax = amrex::max(fmax, amrex::ParticleReal(1.e-30));
        amrex::ParticleReal R = 0._prt;
        for (int guard = 0; guard < 64; ++guard) {
            R = amrex::Random(engine);
            const amrex::ParticleReal fr = std::pow(R, a_int) * std::pow(1.0_prt - R, a_tr);
            if (amrex::Random(engine) * fmax <= fr) { break; }
        }
        return R * E_c;
    }

    // Larsen-Borgnakke redistribution for a quantum harmonic vibrational mode, given the total collision
    // energy E_c = E_trans + E_vib [eV] and vibrational characteristic temperature theta_vib [K].
    // A trial level i in [0, i_max] is accepted with probability (1 - i*dev/E_c)^a_tr. Returns E_vib [eV].
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal sampleLB_harmonic (amrex::ParticleReal E_c, amrex::ParticleReal theta_vib_K,
                                           amrex::ParticleReal a_tr, amrex::RandomEngine const& engine)
    {
        using namespace amrex::literals;
        const amrex::ParticleReal dev = kB_eV() * theta_vib_K;   // vibrational quantum [eV]
        if (dev <= 0._prt || E_c <= 0._prt) { return 0._prt; }
        const int i_max = static_cast<int>(amrex::Math::floor(E_c / dev));
        if (i_max <= 0) { return 0._prt; }
        int i = 0;
        for (int guard = 0; guard < 128; ++guard) {
            i = static_cast<int>(amrex::Math::floor(amrex::Random(engine) * static_cast<amrex::ParticleReal>(i_max + 1)));
            i = amrex::min(i, i_max);
            const amrex::ParticleReal frac = 1.0_prt - static_cast<amrex::ParticleReal>(i) * dev / E_c;
            const amrex::ParticleReal accept = (a_tr > 0._prt) ? std::pow(amrex::max(frac,0._prt), a_tr) : 1.0_prt;
            if (amrex::Random(engine) <= accept) { break; }
        }
        return static_cast<amrex::ParticleReal>(i) * dev;
    }

} // namespace InternalDOF

#endif // WARPX_PARTICLES_INTERNAL_DOF_H_

/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_FRAMETRANSFORM_H_
#define WARPX_RADIATION_FRAMETRANSFORM_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace warpx::radiation
{
    using FourVector = amrex::GpuArray<amrex::Real, 4>;
    using StressEnergy = amrex::GpuArray<FourVector, 4>;

    /** Lorentz boost from a rest frame moving at c*beta into the lab. All four
     * components have the same units: use (energy,c*momentum) for a packet, or
     * (G0,Gx,Gy,Gz) for the four-force. A four-force need not be future timelike.
     * The caller must supply finite |beta|<1. Inverse transforms use -beta. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FourVector
    BoostFourVector (FourVector const& value, amrex::GpuArray<amrex::Real, 3> const& beta) noexcept
    {
        amrex::Real b2 = 0;
        amrex::Real dot = 0;
        for (int d = 0; d < 3; ++d) {
            b2 += beta[d] * beta[d];
            dot += beta[d] * value[d + 1];
        }
        amrex::Real const gamma = 1 / std::sqrt(1 - b2);
        // (gamma-1)/beta^2 = gamma^2/(gamma+1), including beta=0.
        amrex::Real const spatial = gamma * value[0] + gamma * gamma / (gamma + 1) * dot;
        FourVector result{};
        result[0] = gamma * (value[0] + dot);
        for (int d = 0; d < 3; ++d) {
            result[d + 1] = value[d + 1] + spatial * beta[d];
        }
        return result;
    }

    /** Transform radiation density moments, not an integrated packet vector.
     * T00=E, T0i=Ti0=Fi/c, Tij=Pij. Both indices must be transformed; boosting
     * only (E,F/c) gives the wrong radiation enthalpy and moving LTE state. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE StressEnergy
    BoostStressEnergy (StressEnergy const& value,
                       amrex::GpuArray<amrex::Real, 3> const& beta) noexcept
    {
        StressEnergy left{};
        StressEnergy result{};
        for (int column = 0; column < 4; ++column) {
            FourVector vector{};
            for (int row = 0; row < 4; ++row) {
                vector[row] = value[row][column];
            }
            auto const transformed = BoostFourVector(vector, beta);
            for (int row = 0; row < 4; ++row) {
                left[row][column] = transformed[row];
            }
        }
        for (int row = 0; row < 4; ++row) {
            result[row] = BoostFourVector(left[row], beta);
        }
        return result;
    }

    struct GreyFourForceResult {
        FourVector material_force{};
        bool valid = false;
    };

    /** Linear four-force map from comoving moments, also usable on signed
     * tangent tensors with equilibrium=0. The caller validates beta and rates. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FourVector
    GreyFourForceFromComoving (StressEnergy const& comoving,
                               amrex::GpuArray<amrex::Real, 3> const& beta, amrex::Real absorption,
                               amrex::Real scattering, amrex::Real equilibrium) noexcept
    {
        FourVector force{};
        force[0] = absorption * (comoving[0][0] - equilibrium);
        for (int d = 0; d < 3; ++d) {
            force[d + 1] = (absorption + scattering) * comoving[0][d + 1];
        }
        return BoostFourVector(force, beta);
    }

    // Evaluate alpha/gamma*(E_comoving-B) without first rounding a boosted
    // energy of order E and then subtracting the almost equal LTE bath.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    GreyEnergyMinusWork (StressEnergy const& laboratory,
                         amrex::GpuArray<amrex::Real, 3> const& beta, amrex::Real absorption,
                         amrex::Real equilibrium) noexcept
    {
        amrex::Real b2 = 0;
        amrex::Real correction = 0;
        for (int d = 0; d < 3; ++d) {
            b2 = std::fma(beta[d], beta[d], b2);
            correction = std::fma(-2 * beta[d], laboratory[0][d + 1], correction);
            for (int e = 0; e < 3; ++e) {
                correction = std::fma(beta[d] * beta[e], laboratory[d + 1][e + 1], correction);
            }
        }
        if (b2 > 1.e-4) {
            amrex::GpuArray<amrex::Real, 3> inverse{};
            for (int d = 0; d < 3; ++d) {
                inverse[d] = -beta[d];
            }
            auto const rest = BoostStressEnergy(laboratory, inverse);
            return absorption * std::sqrt(1 - b2) * (rest[0][0] - equilibrium);
        }
        correction = std::fma(b2, equilibrium, correction);
        return absorption / std::sqrt(1 - b2) * ((laboratory[0][0] - equilibrium) + correction);
    }

    /** Gray LTE absorption plus coherent isotropic comoving scattering.
     * alpha_absorption/scattering are inverse proper lengths, equilibrium is
     * a_R*T^4, moments are lab-frame energy densities. c*G0 is total material
     * heating+work per volume/time; Gi is force per volume. The equal opposite
     * radiation source must be used. This kernel supplies no angular closure,
     * transport integrator, caloric partition, or finite-frequency group terms.
     * The caller supplies realizable symmetric moments; valid only denotes a
     * finite frame/source evaluation. This is NOT permission to enable scalar-FLD
     * conversion/moving guards. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GreyFourForceResult
    EvaluateGreyFourForce (StressEnergy const& laboratory,
                           amrex::GpuArray<amrex::Real, 3> const& beta,
                           amrex::Real alpha_absorption, amrex::Real alpha_scattering,
                           amrex::Real equilibrium) noexcept
    {
        GreyFourForceResult result;
        amrex::Real b2 = 0;
        amrex::GpuArray<amrex::Real, 3> inverse{};
        for (int d = 0; d < 3; ++d) {
            b2 += beta[d] * beta[d];
            inverse[d] = -beta[d];
        }
        if (!(b2 < 1) || !amrex::Math::isfinite(b2) || !(alpha_absorption >= 0) ||
            !(alpha_scattering >= 0) || !(equilibrium >= 0) ||
            !amrex::Math::isfinite(alpha_absorption + alpha_scattering + equilibrium)) {
            return result;
        }
        for (auto const& row : laboratory) {
            for (auto value : row) {
                if (!amrex::Math::isfinite(value)) {
                    return result;
                }
            }
        }
        auto const comoving = BoostStressEnergy(laboratory, inverse);
        if (!(comoving[0][0] >= 0)) {
            return result;
        }
        auto const gamma = 1 / std::sqrt(1 - b2);
        amrex::Real beta_q = 0;
        amrex::Real beta_pressure_beta = 0;
        amrex::GpuArray<amrex::Real, 3> pressure_beta{};
        for (int d = 0; d < 3; ++d) {
            beta_q = std::fma(beta[d], laboratory[0][d + 1], beta_q);
            for (int e = 0; e < 3; ++e) {
                pressure_beta[d] = std::fma(laboratory[d + 1][e + 1], beta[e], pressure_beta[d]);
            }
            beta_pressure_beta = std::fma(beta[d], pressure_beta[d], beta_pressure_beta);
        }
        // E-E_comoving, retaining the small velocity-dependent difference.
        auto const energy_difference =
            gamma * gamma * (2 * beta_q - beta_pressure_beta - b2 * laboratory[0][0]);
        result.material_force[0] =
            gamma * (alpha_absorption * ((laboratory[0][0] - equilibrium) - beta_q) +
                     alpha_scattering * (energy_difference - beta_q));
        for (int d = 0; d < 3; ++d) {
            auto const projected_flux = laboratory[0][d + 1] - pressure_beta[d];
            result.material_force[d + 1] =
                gamma * (alpha_absorption * std::fma(-beta[d], equilibrium, projected_flux) +
                         alpha_scattering * std::fma(-beta[d], comoving[0][0], projected_flux));
        }
        if (b2 > 1.e-4) {
            result.material_force = GreyFourForceFromComoving(comoving, beta, alpha_absorption,
                                                              alpha_scattering, equilibrium);
        }
        for (auto value : result.material_force) {
            if (!amrex::Math::isfinite(value)) {
                result.material_force = {};
                return result;
            }
        }
        result.valid = true;
        return result;
    }
} // namespace warpx::radiation
#endif

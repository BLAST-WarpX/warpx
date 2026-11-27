/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COMOVINGDIPOLEPRESSURE_H_
#define WARPX_RADIATION_COMOVINGDIPOLEPRESSURE_H_

#include "ImplicitDipoleCollisionSource.H"

namespace warpx::radiation
{
struct ComovingDipolePressureResult
{
    StressEnergy tensor{};
    FourVector proper{};
    bool valid = false;
};

/** Frozen-beta linear pressure map for LAB (E,F/c), not an M1 closure.
 * Signed Krylov/tangent states may use physical=false; physical states require
 * the SAME nonnegative comoving dipole as the implicit collision source.
 * No transport, opacity, native material or runtime authority is supplied. */
AMREX_GPU_HOST_DEVICE inline ComovingDipolePressureResult
EvaluateComovingDipolePressure (
    FourVector const& lab, amrex::GpuArray<amrex::Real, 3> const& beta,
    bool physical = true)
{
    ComovingDipolePressureResult result;
    StressEnergy moment{}, source{};
    if (!BuildIsotropicDipoleFrameMatrices(beta, moment, source)) { return result; }
    amrex::Real scale = 0, beta2 = 0;
    for (auto value : beta) { beta2 += value * value; }
    if (beta2 > amrex::Real(1.e-4)) { return result; }
    for (auto value : lab)
    {
        if (!amrex::Math::isfinite(value)) { return result; }
        scale = amrex::max(scale, std::abs(value));
    }
    if (scale == 0) { result.valid = true; return result; }
    FourVector rhs{}, proper{};
    for (int d = 0; d < 4; ++d) { rhs[d] = lab[d] / scale; }
    if (!gray_absorption_detail::Solve(moment, rhs, proper) ||
        (physical && !gray_absorption_detail::PositiveDipole(proper)))
    {
        return result;
    }
    StressEnergy rest{};
    rest[0][0] = proper[0];
    for (int d = 0; d < 3; ++d)
    {
        rest[0][d + 1] = rest[d + 1][0] = proper[d + 1];
        rest[d + 1][d + 1] = proper[0] / 3;
    }
    auto tensor = BoostStressEnergy(rest, beta);
    for (int i = 0; i < 4; ++i)
    {
        result.proper[i] = scale * proper[i];
        for (int j = 0; j < 4; ++j)
        {
            result.tensor[i][j] = scale * tensor[i][j];
            if (!amrex::Math::isfinite(result.tensor[i][j])) { return result; }
        }
    }
    // These are supplied conserved inventories, not reconstructed pressure.
    // Keep their exact values when the inverse/forward boost differs by ULPs.
    for (int d = 0; d < 4; ++d)
    {
        result.tensor[0][d] = result.tensor[d][0] = lab[d];
    }
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
#endif

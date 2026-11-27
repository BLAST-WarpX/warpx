/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_ISOTROPICDIPOLEFRAMEMATRICES_H_
#define WARPX_RADIATION_ISOTROPICDIPOLEFRAMEMATRICES_H_

#include "FrameTransform.H"

namespace warpx::radiation
{
/** Exact gray LAB maps for a COMOVING isotropic-pressure angular dipole.
 * M maps rest (E,F/c) moments, using a two-index stress-energy boost with
 * P0=E0*I/3. N maps rest (q,c*f) interaction sources, using a four-vector
 * boost. These maps contain no opacity, timestep, volume, source solve or
 * angular closure inference. Tangent columns may be signed; physical states
 * are validated by the caller. Invalid beta preserves both outputs. */
AMREX_GPU_HOST_DEVICE inline bool
BuildIsotropicDipoleFrameMatrices (amrex::GpuArray<amrex::Real, 3> const &beta,
                                   StressEnergy &moment, StressEnergy &source)
{
    amrex::Real beta2 = 0;
    for (auto value : beta)
    {
        beta2 += value * value;
    }
    if (!(beta2 < 1) || !amrex::Math::isfinite(beta2))
    {
        return false;
    }
    StressEnergy m{}, s{};
    for (int column = 0; column < 4; ++column)
    {
        StressEnergy rest{};
        FourVector force{};
        force[column] = 1;
        if (column == 0)
        {
            rest[0][0] = 1;
            for (int d = 1; d < 4; ++d)
            {
                rest[d][d] = amrex::Real(1) / 3;
            }
        }
        else
        {
            rest[0][column] = 1;
            rest[column][0] = 1;
        }
        auto const tensor = BoostStressEnergy(rest, beta);
        auto const vector = BoostFourVector(force, beta);
        for (int row = 0; row < 4; ++row)
        {
            m[row][column] = tensor[0][row];
            s[row][column] = vector[row];
            if (!amrex::Math::isfinite(m[row][column]) || !amrex::Math::isfinite(s[row][column]))
            {
                return false;
            }
        }
    }
    moment = m;
    source = s;
    return true;
}
} // namespace warpx::radiation
#endif

/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SCALAR_PROBE_H_
#define WARPX_SCALAR_PROBE_H_

#include "ReducedDiags.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

/** Sample stored nodal phi and rho on the finest available Cartesian level.
 * Writes one table shared by all sample times and probe locations. */
class ScalarProbe : public ReducedDiags
{
public:
    explicit ScalarProbe (std::string const& rd_name);
    void InitData () override;
    void ComputeDiags (int step) override;
    void WriteToFile (int step) const override;

private:
    using Position = amrex::GpuArray<amrex::Real, 3>;

    std::vector<Position> m_positions;
    std::vector<int> m_levels;

    void CheckFields () const;
};
#endif

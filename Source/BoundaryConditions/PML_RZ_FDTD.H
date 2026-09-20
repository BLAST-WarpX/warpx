/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PML_RZ_FDTD_H_
#define WARPX_PML_RZ_FDTD_H_

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver_fwd.H"
#include "PML_RZ_FDTD_fwd.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <optional>
#include <string>

/** Exterior radial uniaxial PML for the cylindrical Yee solver.
 * The regular cylindrical curl is coupled to the constitutive tensor
 * diag(s_theta/s_r, s_r/s_theta, s_r*s_theta), where s_r=1+sigma/p
 * and s_theta=1+integral(sigma dr)/(r*p). No damping is applied along z.
 * Electric and magnetic auxiliary equations use trapezoidal time centering.
 */
class PML_RZ_FDTD
{
  public:
    PML_RZ_FDTD (amrex::BoxArray const& grids,
                 amrex::DistributionMapping const& dm,
                 amrex::Geometry const& geom, ablastr::fields::VectorField E,
                 ablastr::fields::VectorField B, int ncell, int delta);

    void EvolveE (FiniteDifferenceSolver& solver, amrex::Real dt);
    void EvolveB (FiniteDifferenceSolver& solver, amrex::Real dt);
    void FillBoundary (ablastr::fields::VectorField regular, bool electric,
                       bool single_precision, std::optional<bool> nodal_sync);
    void CheckPoint (std::string const& prefix) const;
    void Restart (std::string const& prefix);

  private:
    using Storage = std::array<std::unique_ptr<amrex::MultiFab>, 3>;
    static ablastr::fields::VectorField View (Storage const& fields);
    void UpdateConstitutive (Storage& field, Storage& auxiliary,
                             Storage const& increment, amrex::Real dt);

    amrex::Geometry const& m_geom;
    int m_delta;
    Storage m_E, m_B;
    // r/theta components store displacement; z stores the first filtered field.
    Storage m_aux_E, m_aux_B;
    Storage m_increment_E, m_increment_B, m_zero_current;
};
#endif

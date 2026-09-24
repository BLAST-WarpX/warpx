/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PROTOTYPE_RUNTIMEDIMS_SIMULATION_H_
#define WARPX_PROTOTYPE_RUNTIMEDIMS_SIMULATION_H_

#include "Particles/PIdx.H"
#include "Utils/WarpXDim.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <array>
#include <memory>
#include <string>
#include <vector>

/** Particle container with the unified (runtime-dimensionality) PIdx layout:
 *  positions are always the first three SoA real components (x, y, z). */
using UnifiedParticleContainer = amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>;
using UnifiedParIter = amrex::ParIterSoA<PIdx::nattribs, 0>;

/** One particle species read from the inputs (subset of WarpX's species support:
 *  constant density profile, NUniformPerCell injection, parsed momentum functions). */
struct Species
{
    std::string name;
    amrex::Real charge = 0.;            //!< charge [C]
    amrex::Real mass = 0.;              //!< mass [kg]
    amrex::Real density = 0.;           //!< number density [1/m^3]
    amrex::IntVect ppc{1, 1, 1};        //!< particles per cell along each logical axis
    amrex::RealBox inject_box;          //!< injection bounds in logical (x, y, z)
    std::unique_ptr<amrex::Parser> momentum_ux;
    std::unique_ptr<amrex::Parser> momentum_uy;
    std::unique_ptr<amrex::Parser> momentum_uz;
    std::unique_ptr<UnifiedParticleContainer> pc;
};

/** Minimal electromagnetic PIC simulation with runtime-selected dimensionality
 *
 * Reproduces the explicit Yee FDTD + Boris pusher + direct current deposition
 * PIC loop of WarpX (cf. WarpX::OneStep_nosub) on degenerate-3D AMReX data:
 * collapsed dimensions have extent 1, prob_lo/hi = [-0.5, 0.5) (cell size 1 m)
 * and zero guard cells.
 */
class Simulation
{
public:
    explicit Simulation (warpx::Dim dim);

    //! Set up geometry, fields and particles from the inputs file
    void InitData ();

    //! Run the PIC loop for max_step steps
    void Evolve ();

private:
    void MakeGeometry ();
    void ReadParameters ();
    void AllocateFields ();
    void InitParticles ();

    //! field gather + Boris momentum push + position push + current deposition
    void PushAndDeposit ();
    void EvolveB (amrex::Real a_dt);
    void EvolveE (amrex::Real a_dt);
    void SumBoundaryJ ();
    void FillBoundaryE ();
    void FillBoundaryB ();

    //! reduced field diagnostics (CSV) and plotfile output
    void RunDiagnostics (int step, amrex::Real time);

    //! nodal flags of a field, with collapsed dimensions forced to cell-centered
    [[nodiscard]] amrex::IntVect UnifiedStagger (amrex::IntVect logical_flags) const;

    warpx::Dim m_dim;

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_E;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_B;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_j;

    std::vector<Species> m_species;

    // stencil coefficients (1/dx, 1/dy, 1/dz), cf. FiniteDifferenceSolver
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_x;
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_y;
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;

    amrex::Real m_dt = 0.;
    amrex::Real m_cfl = 0.999;
    int m_max_step = 0;
    int m_nox = 1;                      //!< particle shape order
    bool m_galerkin_interpolation = true;
    int m_plot_int = -1;                //!< plotfile interval (diag1.intervals, simplified)
    std::string m_plotfile_prefix = "diags/plt";
};

#endif // WARPX_PROTOTYPE_RUNTIMEDIMS_SIMULATION_H_

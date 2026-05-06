/* Copyright 2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_RSPHERE)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/SphericalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/Parser/ParserUtils.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <memory>

/**
 * Compute the minimum of array x, where x has dimension AMREX_SPACEDIM
 */
AMREX_FORCE_INLINE amrex::Real
minDim (const amrex::Real* x)
{
    return std::min({AMREX_D_DECL(x[0], x[1], x[2])});
}

/**
 * Determine the timestep of the simulation. */
void
WarpX::ComputeDt ()
{
    // Handle cases where the timestep is not limited by the speed of light
    // and no constant timestep is provided
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_const_dt.has_value(), "warpx.const_dt must be specified with the hybrid-PIC solver.");
    } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_const_dt.has_value() || m_dt_update_interval.isActivated(),
            "warpx.const_dt must be specified with the electrostatic solver, or warpx.dt_update_interval must be > 0."
        );
    }

    // Determine the appropriate timestep as limited by the speed of light
    const amrex::Real* dx = geom[max_level].CellSize();
    amrex::Real deltat = 0.;

    if (m_const_dt.has_value()) {
        deltat = m_const_dt.value();
    } else if (electrostatic_solver_id  != ElectrostaticSolverAlgo::None) {
        // Set dt for electrostatic algorithm
        if (m_max_dt.has_value()) {
            deltat = m_max_dt.value();
        } else {
            deltat = cfl * minDim(dx) / PhysConst::c;
        }
    } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        // Computation of dt for spectral algorithm
        // (determined by the minimum cell size in all directions)
        deltat = cfl * minDim(dx) / PhysConst::c;
    } else {
        // Computation of dt for FDTD algorithm
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        // - In RZ geometry
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee) {
            deltat = cfl * CylindricalYeeAlgorithm::ComputeMaxDt(dx,  n_rz_azimuthal_modes);
#elif defined(WARPX_DIM_RSPHERE)
        // - In RZ geometry
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee) {
            deltat = cfl * SphericalYeeAlgorithm::ComputeMaxDt(dx);
#else
        // - In Cartesian geometry
        if (grid_type == GridType::Collocated) {
            deltat = cfl * CartesianNodalAlgorithm::ComputeMaxDt(dx);
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
                    || electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {
            deltat = cfl * CartesianYeeAlgorithm::ComputeMaxDt(dx);
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC) {
            deltat = cfl * CartesianCKCAlgorithm::ComputeMaxDt(dx);
#endif
        } else {
            WARPX_ABORT_WITH_MESSAGE("ComputeDt: Unknown algorithm");
        }
    }

    dt.resize(0);
    dt.resize(max_level+1,deltat);

    if (m_do_subcycling) {
        for (int lev = max_level-1; lev >= 0; --lev) {
            dt[lev] = dt[lev+1] * refRatio(lev)[0];
        }
    }
}

/**
 * Determine the simulation timestep from the maximum speed of all particles
 * Sets timestep so that a particle can only cross cfl*dx cells per timestep.
 */
std::optional<amrex::Real>
WarpX::DtLimitFromParticleSpeeds ()
{
    const amrex::Real* dx = geom[max_level].CellSize();
    const amrex::Real dx_min = minDim(dx);

    const amrex::ParticleReal max_v = mypc->maxParticleVelocity();

    if (max_v > 0.) {
        return cfl * dx_min / max_v;
    } else {
        return std::nullopt;
    }

}

std::optional<amrex::Real>
WarpX::DtLimitFromPlasmaFrequency ()
{
    if (!m_max_omegap_dt.has_value()) {
        return std::nullopt;
    }

    const std::unique_ptr<amrex::MultiFab> global_plasma_frequency = mypc->GetGlobalPlasmaFrequency(0);
    const amrex::Real global_plasma_frequency_max = global_plasma_frequency->max(0);

    if (global_plasma_frequency_max > 0.) {
        return m_max_omegap_dt.value()/global_plasma_frequency_max;
    } else {
        return std::nullopt;
    }
}

std::optional<amrex::Real>
WarpX::DtLimitFromCyclotronFrequency ()
{
    using ablastr::fields::Direction;

    if (!m_max_omegac_dt.has_value()) {
        return std::nullopt;
    }

    const amrex::ParmParse pp_particles("particles");
    amrex::Vector<amrex::Real> B_external_particle(3, 0.);
    utils::parser::queryArrWithParser(pp_particles, "B_external_particle", B_external_particle);

    amrex::MultiFab const & Bx = *m_fields.get("Bfield_fp", Direction{0}, 0);
    amrex::MultiFab const & By = *m_fields.get("Bfield_fp", Direction{1}, 0);
    amrex::MultiFab const & Bz = *m_fields.get("Bfield_fp", Direction{2}, 0);

    amrex::Real B_max = 0.;

    B_max = std::max(B_max, Bx.norm0(0));
    B_max = std::max(B_max, By.norm0(0));
    B_max = std::max(B_max, Bz.norm0(0));

    B_max = std::max(B_max, std::abs(B_external_particle[0]));
    B_max = std::max(B_max, std::abs(B_external_particle[1]));
    B_max = std::max(B_max, std::abs(B_external_particle[2]));

    amrex::Real omegac_max = 0.;

    const int n_containers = mypc->nContainers();
    for (int i = 0; i < n_containers; i++)
    {
        const WarpXParticleContainer& pc = mypc->GetParticleContainer(i);
        if (pc.getMass() > 0.) {
            const amrex::Real pc_omegac = pc.getCharge()*B_max/pc.getMass();
            omegac_max = std::max(omegac_max, pc_omegac);
        }
    }

    if (omegac_max > 0.) {
        return m_max_omegac_dt.value()/omegac_max;
    } else {
        return std::nullopt;
    }
}

void
WarpX::ApplyDtLimiters ()
{
    std::optional<amrex::Real> speed_limit = DtLimitFromParticleSpeeds();
    std::optional<amrex::Real> opmegap_limit = DtLimitFromPlasmaFrequency();
    std::optional<amrex::Real> opmegac_limit = DtLimitFromCyclotronFrequency();

    amrex::Real dt_new = std::numeric_limits<amrex::Real>::max();

    if (!speed_limit.has_value() &&
        !opmegap_limit.has_value() &&
        !opmegac_limit.has_value()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_dt.has_value(),
                                         "No valid time step size limit found, warpx.max_dt must be specified");
        dt_new = m_max_dt.value();
    }

    if (speed_limit.has_value()) { dt_new = std::min(dt_new, speed_limit.value()); }
    if (opmegap_limit.has_value()) { dt_new = std::min(dt_new, opmegap_limit.value()); }
    if (opmegac_limit.has_value()) { dt_new = std::min(dt_new, opmegac_limit.value()); }

    // Update dt
    dt[max_level] = dt_new;

    for (int lev = max_level-1; lev >= 0; --lev) {
        dt[lev] = dt[lev+1] * refRatio(lev)[0];
    }

}

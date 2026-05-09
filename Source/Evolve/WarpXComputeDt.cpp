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
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/Parser/ParserUtils.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <memory>
#include <filesystem>

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
 * Used to determine the simulation timestep from the maximum speed of all particles
 * Timestep will be set so that a particle can cross at most cfl*dx cells per timestep.
 */
amrex::Real
WarpX::ParticleGridSpeedMax ()
{
    const amrex::Real* dx = geom[max_level].CellSize();
    const amrex::Real dx_min = minDim(dx);

    const amrex::ParticleReal max_v = mypc->maxParticleVelocity();

    return max_v/dx_min;
}

amrex::Real
WarpX::GlobalPlasmaFrequencyMax ()
{
    const std::unique_ptr<amrex::MultiFab> global_plasma_frequency = mypc->GetGlobalPlasmaFrequency(0);
    const amrex::Real global_plasma_frequency_max = global_plasma_frequency->max(0);
    return global_plasma_frequency_max;
}

amrex::Real
WarpX::GlobalCyclotronFrequencyMax ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const amrex::ParmParse pp_particles("particles");
    amrex::Vector<amrex::Real> B_external_particle(3, 0.);
    utils::parser::queryArrWithParser(pp_particles, "B_external_particle", B_external_particle);

    const amrex::Real Bx_external = B_external_particle[0];
    const amrex::Real By_external = B_external_particle[1];
    const amrex::Real Bz_external = B_external_particle[2];

    amrex::Real B_max = 0.;

    // loop over refinement levels
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        // get MultiFab data at lev
        const amrex::MultiFab & Bx = *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
        const amrex::MultiFab & By = *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
        const amrex::MultiFab & Bz = *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

        // Prepare interpolation of field components to cell center
        // The arrays below store the index type (staggering) of each MultiFab, with the third
        // component set to zero in the two-dimensional case.
        auto Bxtype = amrex::GpuArray<int,3>{0, 0, 0};
        auto Bytype = amrex::GpuArray<int,3>{0, 0, 0};
        auto Bztype = amrex::GpuArray<int,3>{0, 0, 0};
        for (int i = 0; i < AMREX_SPACEDIM; ++i){
            Bxtype[i] = Bx.ixType()[i];
            Bytype[i] = By.ixType()[i];
            Bztype[i] = Bz.ixType()[i];
        }

        // General preparation of interpolation and reduction operations
        const amrex::GpuArray<int,3> cellCenteredtype{0,0,0};
        const amrex::GpuArray<int,3> reduction_coarsening_ratio{1,1,1};
        constexpr int reduction_comp = 0;

        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        // MFIter loop to interpolate fields to cell center and get maximum value
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(Bx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Make the box cell centered in preparation for the interpolation (and to avoid
            // including ghost cells in the calculation)
            const amrex::Box & box = enclosedCells(mfi.nodaltilebox());
            const auto& arrBx = Bx[mfi].array();
            const auto& arrBy = By[mfi].array();
            const auto& arrBz = Bz[mfi].array();

            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                const amrex::Real Bx_interp = ablastr::coarsen::sample::Interp(arrBx, Bxtype, cellCenteredtype,
                                                                               reduction_coarsening_ratio, i, j, k, reduction_comp);
                const amrex::Real By_interp = ablastr::coarsen::sample::Interp(arrBy, Bytype, cellCenteredtype,
                                                                               reduction_coarsening_ratio, i, j, k, reduction_comp);
                const amrex::Real Bz_interp = ablastr::coarsen::sample::Interp(arrBz, Bztype, cellCenteredtype,
                                                                               reduction_coarsening_ratio, i, j, k, reduction_comp);
                return {amrex::Math::powi<2>(Bx_interp + Bx_external) +
                        amrex::Math::powi<2>(By_interp + By_external) +
                        amrex::Math::powi<2>(Bz_interp + Bz_external)};
            });
        }

        const amrex::Real hv_Bsq = amrex::get<0>(reduce_data.value()); // highest value of |B|**2

        B_max = std::max(B_max, std::sqrt(hv_Bsq));

    }

    // MPI reduce
    amrex::ParallelDescriptor::ReduceRealMax({B_max});

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

    return omegac_max;
}

void
WarpX::ApplyDtLimiters ()
{
    using namespace amrex::literals;

    // Calculate limiting values from the simulation conditions
    const amrex::Real vmax_o_dx = ParticleGridSpeedMax();
    const amrex::Real omegap_max = m_max_omegap_dt.has_value() ? GlobalPlasmaFrequencyMax() : 0._rt;
    const amrex::Real omegac_max = m_max_omegac_dt.has_value() ? GlobalCyclotronFrequencyMax() : 0._rt;

    // Ensure that a valid time step value exists, either from the simulation conditions or from max_dt
    if (vmax_o_dx == 0._rt &&
        (!m_max_omegap_dt.has_value() || omegap_max == 0._rt) &&
        (!m_max_omegac_dt.has_value() || omegac_max == 0._rt)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_dt.has_value(),
                                         "No valid time step size limit found, warpx.max_dt must be specified");
    }

    amrex::Real dt_new = std::numeric_limits<amrex::Real>::max();

    if (vmax_o_dx > 0._rt) {
        dt_new = std::min(dt_new, cfl/vmax_o_dx);
    }
    if (m_max_omegap_dt.has_value() && omegap_max > 0._rt) {
        dt_new = std::min(dt_new, m_max_omegap_dt.value()/omegap_max);
    }
    if (m_max_omegac_dt.has_value() && omegac_max > 0._rt) {
        dt_new = std::min(dt_new, m_max_omegac_dt.value()/omegac_max);
    }

    if (m_max_dt.has_value()) {
        dt_new = std::min(dt_new, m_max_dt.value());
    }

    // Update dt
    dt[max_level] = dt_new;

    for (int lev = max_level-1; lev >= 0; --lev) {
        dt[lev] = dt[lev+1] * refRatio(lev)[0];
    }

    // Write diagnostics if requested
    if (amrex::ParallelDescriptor::IOProcessor()
        && !m_dt_update_diagnostic_file.empty()
        && !amrex::FileExists(m_dt_update_diagnostic_file)) {

        std::filesystem::path const diagnostic_path(m_dt_update_diagnostic_file);
        std::filesystem::path const diagnostic_dir = diagnostic_path.parent_path();
        if (!diagnostic_dir.empty()) {
            std::filesystem::create_directories(diagnostic_dir);
        }

        std::ofstream diagnostic_file{m_dt_update_diagnostic_file, std::ofstream::out | std::ofstream::trunc};
        if (!diagnostic_file.is_open()) {
            amrex::Abort("Failed to open file: " + m_dt_update_diagnostic_file);
        }

        int c = 0;
        diagnostic_file << "#";
        diagnostic_file << "[" << c++ << "]step()";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]time(s)";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]new_dt";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]vmax_dt";
        if (m_max_omegap_dt.has_value()) {
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]omegap_dt";
        }
        if (m_max_omegac_dt.has_value()) {
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]omegac_dt";
        }
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

    if (amrex::ParallelDescriptor::IOProcessor()
        && !m_dt_update_diagnostic_file.empty()) {

        std::ofstream diagnostic_file{m_dt_update_diagnostic_file, std::ofstream::out | std::ofstream::app};
        if (!diagnostic_file.is_open()) {
            amrex::Abort("Failed to open file: " + m_dt_update_diagnostic_file);
        }

        diagnostic_file << std::setprecision(14);
        diagnostic_file << istep[0] + 1;
        diagnostic_file << " ";
        diagnostic_file << t_new[0];
        diagnostic_file << " ";
        diagnostic_file << dt_new;
        diagnostic_file << " ";
        diagnostic_file << vmax_o_dx*dt_new;

        if (m_max_omegap_dt.has_value()) {
            diagnostic_file << " ";
            diagnostic_file << omegap_max*dt_new;
        }
        if (m_max_omegac_dt.has_value()) {
            diagnostic_file << " ";
            diagnostic_file << omegac_max*dt_new;
        }
        diagnostic_file << "\n";
        diagnostic_file.close();
    }
}

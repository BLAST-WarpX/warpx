/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * This file was copied and edited from PoissonSolver.H in the same directory.
 */
#ifndef ABLASTR_EFFECTIVE_POTENTIAL_POISSON_SOLVER_H
#define ABLASTR_EFFECTIVE_POTENTIAL_POISSON_SOLVER_H

#include <ablastr/constant.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/utils/TextMsg.H>
#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/fields/Interpolate.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include "PoissonSolver.H"

#if defined(WARPX_USE_FFT) && defined(WARPX_DIM_3D)
#include <ablastr/fields/IntegratedGreenFunctionSolver.H>
#endif

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFInterp_C.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <array>
#include <optional>


namespace ablastr::fields {

/** Compute the potential `phi` by solving the Poisson equation with a modifed dielectric function
 *
 * Uses `rho` as a source. This uses the AMReX solver.
 *
 * More specifically, this solves the equation
 * \f[
 *   \nabla \cdot \sigma \nabla \phi = - \rho/\epsilon_0
 * \f]
 *
 * \tparam T_PostPhiCalculationFunctor a calculation per level directly after phi was calculated
 * \tparam T_BoundaryHandler handler for boundary conditions, for example @see ElectrostaticSolver::PoissonBoundaryHandler
 * \tparam T_FArrayBoxFactory usually nothing or an amrex::EBFArrayBoxFactory (EB ONLY)
 * \param[in] rho The charge density a given species
 * \param[out] phi The potential to be computed by this function
 * \param[in] sigma The matrix representing the mass operator used to lower the local plasma frequency
 * \param[in] relative_tolerance The relative convergence threshold for the MLMG solver
 * \param[in] absolute_tolerance The absolute convergence threshold for the MLMG solver
 * \param[in] max_iters The maximum number of iterations allowed for the MLMG solver
 * \param[in] verbosity The verbosity setting for the MLMG solver
 * \param[in] geom the geometry per level (e.g., from AmrMesh)
 * \param[in] dmap the distribution mapping per level (e.g., from AmrMesh)
 * \param[in] grids the grids per level (e.g., from AmrMesh)
 * \param[in] is_solver_igf_on_lev0 boolean to select the Poisson solver: 1 for FFT on level 0 & Multigrid on other levels, 0 for Multigrid on all levels
 * \param[in] do_single_precision_comms perform communications in single precision
 * \param[in] rel_ref_ratio mesh refinement ratio between levels (default: 1)
 * \param[in] post_phi_calculation perform a calculation per level directly after phi was calculated; required for embedded boundaries (default: none)
 * \param[in] boundary_handler a handler for boundary conditions, for example @see ElectrostaticSolver::PoissonBoundaryHandler
 * \param[in] current_time the current time; required for embedded boundaries (default: none)
 * \param[in] eb_farray_box_factory a factory for field data, @see amrex::EBFArrayBoxFactory; required for embedded boundaries (default: none)
 */
template<
    typename T_PostPhiCalculationFunctor = std::nullopt_t,
    typename T_BoundaryHandler = std::nullopt_t,
    typename T_FArrayBoxFactory = void
>
void
computeEffectivePotentialPhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    amrex::MultiFab const & sigma,
    amrex::Real relative_tolerance,
    amrex::Real absolute_tolerance,
    int max_iters,
    int verbosity,
    amrex::Vector<amrex::Geometry> const& geom,
    amrex::Vector<amrex::DistributionMapping> const& dmap,
    amrex::Vector<amrex::BoxArray> const& grids,
    [[maybe_unused]] utils::enums::GridType grid_type,
    bool is_solver_igf_on_lev0,
    bool eb_enabled = false,
    bool do_single_precision_comms = false,
    std::optional<amrex::Vector<amrex::IntVect> > rel_ref_ratio = std::nullopt,
    [[maybe_unused]] T_PostPhiCalculationFunctor post_phi_calculation = std::nullopt,
    [[maybe_unused]] T_BoundaryHandler const boundary_handler = std::nullopt,
    [[maybe_unused]] std::optional<amrex::Real const> current_time = std::nullopt, // only used for EB
    [[maybe_unused]] std::optional<amrex::Vector<T_FArrayBoxFactory const *> > eb_farray_box_factory = std::nullopt // only used for EB
) {
    using namespace amrex::literals;

    ABLASTR_PROFILE("computeEffectivePotentialPhi");

    if (!rel_ref_ratio.has_value()) {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(rho.size() == 1u,
                                           "rel_ref_ratio must be set if mesh-refinement is used");
        rel_ref_ratio = amrex::Vector<amrex::IntVect>{{amrex::IntVect(AMREX_D_DECL(1, 1, 1))}};
    }

#if !defined(AMREX_USE_EB)
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!eb_enabled,
                                       "Embedded boundary solve requested but not compiled in");
#endif
    if (eb_enabled && std::is_same_v<void, T_FArrayBoxFactory>) {
        throw std::runtime_error("EB requested by eb_farray_box_factory not provided!");
    }

    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE( !is_solver_igf_on_lev0,
        "FFT solver cannot be used with effective potential Poisson solve");

#ifdef WARPX_DIM_RZ
        constexpr bool is_rz = true;
#else
        constexpr bool is_rz = false;
#endif

    auto const finest_level = static_cast<int>(rho.size() - 1);

    // determine if rho is zero everywhere
    const amrex::Real max_norm_b = getMaxNormRho(
        amrex::GetVecOfConstPtrs(rho), finest_level, absolute_tolerance);

    const amrex::LPInfo info;

    for (int lev=0; lev<=finest_level; lev++) {

        // Use the Multigrid (MLMG) solver but first scale rho appropriately
        using namespace ablastr::constant::SI;
        rho[lev]->mult(-1._rt/ep0);

        std::unique_ptr<amrex::MLNodeLinOp> linop;
        // In the presence of EB or RZ the EB enabled linear solver is used
        if (eb_enabled)
        {
#if defined(AMREX_USE_EB)
            auto linop_nodelap = std::make_unique<amrex::MLEBNodeFDLaplacian>();
            linop_nodelap->define(
                amrex::Vector<amrex::Geometry>{geom[lev]},
                amrex::Vector<amrex::BoxArray>{grids[lev]},
                amrex::Vector<amrex::DistributionMapping>{dmap[lev]},
                info,
                amrex::Vector<amrex::EBFArrayBoxFactory const*>{eb_farray_box_factory.value()[lev]}
            );
            if constexpr (!std::is_same_v<T_BoundaryHandler, std::nullopt_t>) {
                // if the EB potential only depends on time, the potential can be passed
                // as a float instead of a callable
                if (boundary_handler.phi_EB_only_t) {
                    linop_nodelap->setEBDirichlet(boundary_handler.potential_eb_t(current_time.value()));
                } else {
                    linop_nodelap->setEBDirichlet(boundary_handler.getPhiEB(current_time.value()));
                }
            }
            linop_nodelap->setSigma(lev, sigma);
            linop = std::move(linop_nodelap);
#endif
        }
        else if (is_rz)
        {
            auto linop_nodelap = std::make_unique<amrex::MLEBNodeFDLaplacian>();
            linop_nodelap->define(
                amrex::Vector<amrex::Geometry>{geom[lev]},
                amrex::Vector<amrex::BoxArray>{grids[lev]},
                amrex::Vector<amrex::DistributionMapping>{dmap[lev]},
                info
            );
            linop_nodelap->setRZ(true);
            linop_nodelap->setSigma(lev, sigma);
            linop = std::move(linop_nodelap);
        }
        else
        {
            auto linop_nodelap = std::make_unique<amrex::MLNodeLaplacian>();
            linop_nodelap->define(
                amrex::Vector<amrex::Geometry>{geom[lev]},
                amrex::Vector<amrex::BoxArray>{grids[lev]},
                amrex::Vector<amrex::DistributionMapping>{dmap[lev]},
                info
            );
            linop_nodelap->setSigma(lev, sigma);
            linop = std::move(linop_nodelap);
        }

        // Set domain boundary conditions
        if constexpr (std::is_same_v<T_BoundaryHandler, std::nullopt_t>) {
            amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const lobc = {AMREX_D_DECL(
                amrex::LinOpBCType::Dirichlet,
                amrex::LinOpBCType::Dirichlet,
                amrex::LinOpBCType::Dirichlet
            )};
            amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const hibc = lobc;
            linop->setDomainBC(lobc, hibc);
        } else {
            linop->setDomainBC(boundary_handler.lobc, boundary_handler.hibc);
        }

        // Solve the Poisson equation
        amrex::MLMG mlmg(*linop); // actual solver defined here
        mlmg.setVerbose(verbosity);
        mlmg.setMaxIter(max_iters);
        mlmg.setAlwaysUseBNorm((max_norm_b > 0));

        const int ng = int(grid_type == utils::enums::GridType::Collocated); // ghost cells
        if (ng) {
            // In this case, computeE needs to use ghost nodes data. So we
            // ask MLMG to fill BC for us after it solves the problem.
            mlmg.setFinalFillBC(true);
        }

        // Solve Poisson equation at lev
        mlmg.solve( {phi[lev]}, {rho[lev]},
                    relative_tolerance, absolute_tolerance );

        // needed for solving the levels by levels:
        // - coarser level is initial guess for finer level
        // - coarser level provides boundary values for finer level patch
        // Interpolation from phi[lev] to phi[lev+1]
        // (This provides both the boundary conditions and initial guess for phi[lev+1])
        if (lev < finest_level) {
            const amrex::IntVect& refratio = rel_ref_ratio.value()[lev];
            const int ncomp = linop->getNComp();
            interpolatePhiBetweenLevels(phi[lev],
                                        phi[lev+1],
                                        geom[lev],
                                        do_single_precision_comms,
                                        refratio,
                                        ncomp,
                                        ng);
        }

        // Run additional operations, such as calculation of the E field for embedded boundaries
        if constexpr (!std::is_same<T_PostPhiCalculationFunctor, std::nullopt_t>::value) {
            if (post_phi_calculation.has_value()) {
                post_phi_calculation.value()(mlmg, lev);
            }
        }
        rho[lev]->mult(-ep0);  // Multiply rho by epsilon again
    } // loop over lev(els)
}

} // namespace ablastr::fields

#endif // ABLASTR_EFFECTIVE_POTENTIAL_POISSON_SOLVER_H

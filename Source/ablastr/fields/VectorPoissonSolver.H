/* Copyright 2022-2024 S. Eric Clark (Helion Energy, formerly LLNL)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_VECTOR_POISSON_SOLVER_H
#define ABLASTR_VECTOR_POISSON_SOLVER_H

#include <ablastr/constant.H>
#include <ablastr/fields/Interpolate.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/utils/TextMsg.H>
#include <ablastr/warn_manager/WarnManager.H>

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
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <array>
#include <optional>

namespace ablastr::fields {

/** Compute the vector potential `A` by solving the Poisson equation
 *
 * Uses `J` as a source, assuming that the source moves at a
 * constant speed \f$\vec{\beta}\f$. This uses the AMReX solver.
 *
 * More specifically, this solves the equation
 * \f[
 *   \vec{\nabla}^2 r \vec{A} - (\vec{\beta}\cdot\vec{\nabla})^2 r \vec{A} = - r \mu_0 \vec{J}
 * \f]
 *
 * \tparam T_BoundaryHandler handler for boundary conditions, for example @see MagnetostaticSolver::MultiPoissonBoundaryHandler
 * \tparam T_PostACalculationFunctor a calculation per level directly after A was calculated
 * \tparam T_FArrayBoxFactory usually nothing or an amrex::EBFArrayBoxFactory (EB ONLY)
 * \param[in] curr The current density a given species
 * \param[out] A The vector potential to be computed by this function
 * \param[in] relative_tolerance The relative convergence threshold for the MLMG solver
 * \param[in] absolute_tolerance The absolute convergence threshold for the MLMG solver
 * \param[in] max_iters The maximum number of iterations allowed for the MLMG solver
 * \param[in] verbosity The verbosity setting for the MLMG solver
 * \param[in] geom the geometry per level (e.g., from AmrMesh)
 * \param[in] dmap the distribution mapping per level (e.g., from AmrMesh)
 * \param[in] grids the grids per level (e.g., from AmrMesh)
 * \param[in] boundary_handler a handler for boundary conditions, for example @see MagnetostaticSolver::VectorPoissonBoundaryHandler
 * \param[in] eb_enabled solve with embedded boundaries
 * \param[in] do_single_precision_comms perform communications in single precision
 * \param[in] rel_ref_ratio mesh refinement ratio between levels (default: 1)
 * \param[in] post_A_calculation perform a calculation per level directly after A was calculated; required for embedded boundaries (default: none)
 * \param[in] current_time the current time; required for embedded boundaries (default: none)
 * \param[in] eb_farray_box_factory a factory for field data, @see amrex::EBFArrayBoxFactory; required for embedded boundaries (default: none)
 */
template<
    typename T_BoundaryHandler,
    typename T_PostACalculationFunctor = std::nullopt_t,
    typename T_FArrayBoxFactory = void
>
void
computeVectorPotential   (  amrex::Vector<amrex::Array<amrex::MultiFab*, 3> > const & curr,
                            amrex::Vector<amrex::Array<amrex::MultiFab*, 3> > & A,
                            amrex::Real relative_tolerance,
                            amrex::Real absolute_tolerance,
                            int max_iters,
                            int verbosity,
                            amrex::Vector<amrex::Geometry> const& geom,
                            amrex::Vector<amrex::DistributionMapping> const& dmap,
                            amrex::Vector<amrex::BoxArray> const& grids,
                            T_BoundaryHandler const boundary_handler,
                            bool eb_enabled = false,
                            bool do_single_precision_comms = false,
                            std::optional<amrex::Vector<amrex::IntVect> > rel_ref_ratio = std::nullopt,
                            [[maybe_unused]] T_PostACalculationFunctor post_A_calculation = std::nullopt,
                            [[maybe_unused]] std::optional<amrex::Real const> current_time = std::nullopt, // only used for EB
                            [[maybe_unused]] std::optional<amrex::Vector<T_FArrayBoxFactory const *> > eb_farray_box_factory = std::nullopt // only used for EB
)
{
    using namespace amrex::literals;

    if (!rel_ref_ratio.has_value()) {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(curr.size() == 1u,
                                           "rel_ref_ratio must be set if mesh-refinement is used");
        rel_ref_ratio = amrex::Vector<amrex::IntVect>{{amrex::IntVect(AMREX_D_DECL(1, 1, 1))}};
    }

    auto const finest_level = static_cast<int>(curr.size()) - 1;

    // scale J appropriately; also determine if current is zero everywhere
    amrex::Real max_comp_J = 0.0;
    for (int lev=0; lev<=finest_level; lev++) {
        for (int adim=0; adim<3; adim++) {
            curr[lev][adim]->mult(-1._rt*ablastr::constant::SI::mu0); // Unscaled below
            max_comp_J = amrex::max(max_comp_J, curr[lev][adim]->norm0());
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(max_comp_J);

    const bool always_use_bnorm = (max_comp_J > 0);
    if (!always_use_bnorm) {
        if (absolute_tolerance == 0.0) { absolute_tolerance = amrex::Real(1e-6); }
        ablastr::warn_manager::WMRecordWarning(
                "MagnetostaticSolver",
                "Max norm of J is 0",
                ablastr::warn_manager::WarnPriority::low
        );
    }

    // Loop over dimensions of A to solve each component individually
    for (int lev=0; lev<=finest_level; lev++) {
        amrex::LPInfo info;

#ifdef WARPX_DIM_RZ
        constexpr bool is_rz = true;
#else
        constexpr bool is_rz = false;
#endif

        amrex::Array<amrex::Real,AMREX_SPACEDIM> const dx
            {AMREX_D_DECL(geom[lev].CellSize(0),
                          geom[lev].CellSize(1),
                          geom[lev].CellSize(2))};


        if (!eb_enabled && !is_rz) {
            // Determine whether to use semi-coarsening
            int max_semicoarsening_level = 0;
            int semicoarsening_direction = -1;
            const auto min_dir = static_cast<int>(std::distance(dx.begin(),
                                                                std::min_element(dx.begin(), dx.end())));
            const auto max_dir = static_cast<int>(std::distance(dx.begin(),
                                                                std::max_element(dx.begin(), dx.end())));
            if (dx[max_dir] > dx[min_dir]) {
                semicoarsening_direction = max_dir;
                max_semicoarsening_level = static_cast<int>(std::log2(dx[max_dir] / dx[min_dir]));
            }
            if (max_semicoarsening_level > 0) {
                info.setSemicoarsening(true);
                info.setMaxSemicoarseningLevel(max_semicoarsening_level);
                info.setSemicoarseningDirection(semicoarsening_direction);
            }
        }

        amrex::MLEBNodeFDLaplacian linopx, linopy, linopz;
        if (eb_enabled) {
#ifdef AMREX_USE_EB
            linopx.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info, {eb_farray_box_factory.value()[lev]});
            linopy.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info, {eb_farray_box_factory.value()[lev]});
            linopz.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info, {eb_farray_box_factory.value()[lev]});
#endif
        } else {
            linopx.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info);
            linopy.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info);
            linopz.define({geom[lev]}, {grids[lev]}, {dmap[lev]}, info);
        }

        amrex::Array<amrex::MLEBNodeFDLaplacian*,3> linop = {&linopx,&linopy,&linopz};
        amrex::Array<std::unique_ptr<amrex::MLMG>,3> mlmg;

        for (int adim=0; adim<3; adim++) {
            // Solve the Poisson equation
            // This is solving the self fields using the magnetostatic solver in the lab frame

            // Note: this assumes that beta is zero
            linop[adim]->setSigma({AMREX_D_DECL(1._rt, 1._rt, 1._rt)});

            // Set Homogeneous Dirichlet Boundary on EB
#if defined(AMREX_USE_EB)
            if (eb_enabled) { linop[adim]->setEBDirichlet(0_rt); }
#endif

#ifdef WARPX_DIM_RZ
            linop[adim]->setRZ(true);
#endif

            linop[adim]->setDomainBC( boundary_handler.lobc[adim], boundary_handler.hibc[adim] );

            mlmg[adim] = std::make_unique<amrex::MLMG>(*linop[adim]);

            mlmg[adim]->setVerbose(verbosity);
            mlmg[adim]->setMaxIter(max_iters);
            mlmg[adim]->setAlwaysUseBNorm(always_use_bnorm);

            // Solve Poisson equation at lev
            mlmg[adim]->solve( {A[lev][adim]}, {curr[lev][adim]},
                        relative_tolerance, absolute_tolerance );

            // Synchronize the ghost cells, do halo exchange
            ablastr::utils::communication::FillBoundary(
                *A[lev][adim], A[lev][adim]->nGrowVect(),
                do_single_precision_comms,
                geom[lev].periodicity(), false);

            // needed for solving the levels by levels:
            // - coarser level is initial guess for finer level
            // - coarser level provides boundary values for finer level patch
            // Interpolation from phi[lev] to phi[lev+1]
            // (This provides both the boundary conditions and initial guess for phi[lev+1])
            if (lev < finest_level) {

                // Allocate A_cp for lev+1
                amrex::BoxArray ba = A[lev+1][adim]->boxArray();
                const amrex::IntVect& refratio = rel_ref_ratio.value()[lev];
                ba.coarsen(refratio);
                const int ncomp = linop[adim]->getNComp();
                amrex::MultiFab A_cp(ba, A[lev+1][adim]->DistributionMap(), ncomp, 1);

                // Copy from A[lev] to A_cp (in parallel)
                const amrex::IntVect& ng = amrex::IntVect::TheUnitVector();
                const amrex::Periodicity& crse_period = geom[lev].periodicity();

                ablastr::utils::communication::ParallelCopy(
                    A_cp,
                    *A[lev][adim],
                    0,
                    0,
                    1,
                    ng,
                    ng,
                    do_single_precision_comms,
                    crse_period
                );

                // Local interpolation from A_cp to A[lev+1]
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*A[lev+1][adim],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Array4<amrex::Real> const& A_fp_arr = A[lev+1][adim]->array(mfi);
                    amrex::Array4<amrex::Real> const& A_cp_arr = A_cp.array(mfi);

                    details::PoissonInterpCPtoFP const interp(A_fp_arr, A_cp_arr, refratio);

                    amrex::Box const b = mfi.tilebox(A[lev + 1][adim]->ixType().toIntVect());
                    amrex::ParallelFor(b, interp);
                }

            }

            // Unscale current
            curr[lev][adim]->mult(-1._rt/ablastr::constant::SI::mu0);
        } // Loop over adim
        // Run additional operations, such as calculation of the B fields for embedded boundaries
        if constexpr (!std::is_same_v<T_PostACalculationFunctor, std::nullopt_t>) {
            if (post_A_calculation.has_value()) {
                    post_A_calculation.value()(mlmg, lev);
            }
        }
    } // loop over lev(els)
}
}  // namepace Magnetostatic
#endif

/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_FIELDS_MLMG_OPTIONS_H
#define ABLASTR_FIELDS_MLMG_OPTIONS_H

#include <ablastr/utils/TextMsg.H>

#include <AMReX_Config.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_REAL.H>

#include <string>


namespace ablastr::fields {

/** Optional settings for the AMReX MLMG solvers
 *
 * These control the bottom solve (the solve on the coarsest multigrid level)
 * and the depth of the multigrid hierarchy. Members set to a negative value
 * are ignored, i.e. the corresponding AMReX default is used.
 */
struct MLMGOptions
{
    /** Bottom solver used on the coarsest multigrid level. `Default` lets the
     *  linear operator pick its own default (usually BiCGStab).
     */
    amrex::BottomSolver bottom_solver = amrex::BottomSolver::Default;

    /** Verbosity of the bottom solver (0: silent) */
    int bottom_verbosity = 0;

    /** Maximum number of bottom solver iterations (AMReX default: 200) */
    int bottom_max_iters = -1;

    /** Relative tolerance of the bottom solve (AMReX default: 1e-4) */
    amrex::Real bottom_relative_tolerance = amrex::Real(-1.0);

    /** Absolute tolerance of the bottom solve (AMReX default: unused) */
    amrex::Real bottom_absolute_tolerance = amrex::Real(-1.0);

    /** Maximum multigrid coarsening level (AMReX default: 30). Lowering this
     *  leaves a larger problem to the bottom solver, which can pay off when a
     *  strong bottom solver such as HYPRE's BoomerAMG is used.
     */
    int max_coarsening_level = -1;

    /** Prefix of the input file namespace from which AMReX reads HYPRE options */
    std::string hypre_options_namespace = "hypre";
};

/** Translate the name of a bottom solver to the corresponding AMReX enum value
 *
 * This aborts with a descriptive message if the name is not known, or if the
 * requested solver was not compiled in.
 *
 * \param[in] name name of the bottom solver
 * \return the corresponding `amrex::BottomSolver` value
 */
inline amrex::BottomSolver
parseBottomSolver (std::string const & name)
{
    if (name == "default")  { return amrex::BottomSolver::Default; }
    if (name == "smoother") { return amrex::BottomSolver::smoother; }
    if (name == "bicgstab") { return amrex::BottomSolver::bicgstab; }
    if (name == "cg")       { return amrex::BottomSolver::cg; }
    if (name == "bicgcg")   { return amrex::BottomSolver::bicgcg; }
    if (name == "cgbicg")   { return amrex::BottomSolver::cgbicg; }
    if (name == "hypre") {
#if !defined(AMREX_USE_HYPRE)
        ABLASTR_ABORT_WITH_MESSAGE(
            "The 'hypre' bottom solver requires compiling with HYPRE support: "
            "configure with -DWarpX_HYPRE=ON");
#elif (AMREX_SPACEDIM == 1)
        ABLASTR_ABORT_WITH_MESSAGE(
            "The 'hypre' bottom solver is not available in 1D");
#else
        return amrex::BottomSolver::hypre;
#endif
    }
    if (name == "petsc") {
#if !defined(AMREX_USE_PETSC)
        ABLASTR_ABORT_WITH_MESSAGE(
            "The 'petsc' bottom solver requires compiling with PETSc support: "
            "configure with -DWarpX_PETSC=ON");
#elif (AMREX_SPACEDIM == 1)
        ABLASTR_ABORT_WITH_MESSAGE(
            "The 'petsc' bottom solver is not available in 1D");
#else
        return amrex::BottomSolver::petsc;
#endif
    }
    ABLASTR_ABORT_WITH_MESSAGE(
        "Unknown bottom solver '" + name + "'. Valid options are: default, "
        "smoother, bicgstab, cg, bicgcg, cgbicg, hypre, petsc");
    return amrex::BottomSolver::Default;
}

/** Apply the multigrid hierarchy related options to an `amrex::LPInfo` object
 *
 * \param[inout] info the LPInfo object used to define the linear operator
 * \param[in] options the MLMG options to apply
 */
inline void
applyMLMGOptions (amrex::LPInfo & info, MLMGOptions const & options)
{
    if (options.max_coarsening_level >= 0) {
        info.setMaxCoarseningLevel(options.max_coarsening_level);
    }
}

/** Apply the bottom solver related options to an `amrex::MLMG` object
 *
 * \param[inout] mlmg the MLMG solver object
 * \param[in] options the MLMG options to apply
 */
inline void
applyMLMGOptions (amrex::MLMG & mlmg, MLMGOptions const & options)
{
    mlmg.setBottomSolver(options.bottom_solver);
    mlmg.setBottomVerbose(options.bottom_verbosity);
    if (options.bottom_max_iters > 0) {
        mlmg.setBottomMaxIter(options.bottom_max_iters);
    }
    if (options.bottom_relative_tolerance >= amrex::Real(0.0)) {
        mlmg.setBottomTolerance(options.bottom_relative_tolerance);
    }
    if (options.bottom_absolute_tolerance >= amrex::Real(0.0)) {
        mlmg.setBottomToleranceAbs(options.bottom_absolute_tolerance);
    }
#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
    mlmg.setHypreOptionsNamespace(options.hypre_options_namespace);
#endif
}

} // namespace ablastr::fields

#endif // ABLASTR_FIELDS_MLMG_OPTIONS_H

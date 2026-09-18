#ifndef ABLASTR_MLMG_WRAPPER_H_
#define ABLASTR_MLMG_WRAPPER_H_

#include <ablastr/fields/MLMGTuningCache.H>
#include <ablastr/utils/TextMsg.H>

#include <AMReX.H>
#include <AMReX_Exception.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLNodeTensorLaplacian.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ablastr::fields {

template <typename T_BoundaryHandler>
class MLMGWrapper
{
public:

    MLMGWrapper (int lev, bool eb_enabled, bool is_rz,
                 [[maybe_unused]] bool is_solver_igf_on_lev0,
                 amrex::Array<amrex::Real,AMREX_SPACEDIM> const& dx_scaled,
                 amrex::Array<amrex::Real,AMREX_SPACEDIM> const& beta_solver,
                 amrex::Geometry const& geom, amrex::MultiFab const& rho,
                 [[maybe_unused]] T_BoundaryHandler const& boundary_handler,
                 [[maybe_unused]] amrex::Real current_time,
                 [[maybe_unused]] amrex::FabFactory<amrex::FArrayBox> const* factory,
                 int max_iters, int verbosity, std::optional<int> num_final_sweeps,
                 amrex::Real relative_tolerance, amrex::Real absolute_tolerance, int ng)
        : m_geom(geom), m_ng(ng)
    {
        using namespace amrex;
        using namespace amrex::literals;

        LPInfo info0;
        if (!eb_enabled && !is_rz) {
            // Determine whether to use semi-coarsening
            int max_semicoarsening_level = 0;
            int semicoarsening_direction = -1;
            const auto min_dir = static_cast<int>(std::distance(dx_scaled.begin(),
                                                                std::min_element(dx_scaled.begin(), dx_scaled.end())));
            const auto max_dir = static_cast<int>(std::distance(dx_scaled.begin(),
                                                                std::max_element(dx_scaled.begin(), dx_scaled.end())));
            if (dx_scaled[max_dir] > dx_scaled[min_dir]) {
                semicoarsening_direction = max_dir;
                max_semicoarsening_level = static_cast<int>(std::log2(dx_scaled[max_dir] / dx_scaled[min_dir]));
            }
            if (max_semicoarsening_level > 0) {
                info0.setSemicoarsening(true);
                info0.setMaxSemicoarseningLevel(max_semicoarsening_level);
                info0.setSemicoarseningDirection(semicoarsening_direction);
            }
        }

        auto make_mlmg_linop = [&] (LPInfo const& info, BoxArray const& ba,
                                    DistributionMapping const& dm,
                                    [[maybe_unused]] FabFactory<FArrayBox> const* fac)
        {
            std::unique_ptr<amrex::MLMG> l_mlmg;
            std::unique_ptr<amrex::MLNodeLinOp> l_linop;

            if (eb_enabled || is_rz) {
                // In the presence of EB or RZ: the solver assumes that the beam is
                // propagating along one of the axes of the grid, i.e. that only *one*
                // of the components of `beta` is non-negligible.
                auto linop_nodelap = std::make_unique<MLEBNodeFDLaplacian>();
                if (eb_enabled) {
#if defined(AMREX_USE_EB)
                    auto const* eb_factory = dynamic_cast<EBFArrayBoxFactory const*>(fac);
                    if (eb_factory) {
                        linop_nodelap->define(
                            Vector<Geometry>{geom},
                            Vector<BoxArray>{amrex::convert(ba,IntVect(0))},
                            Vector<DistributionMapping>{dm},
                            info, {eb_factory});
                    } else {
                        throw std::runtime_error("EB requested by eb_farray_box_factory not provided!");
                    }
#endif
                }
                else {
                    // TODO: rather use MLNodeTensorLaplacian (for RZ w/o EB) here? Semi-Coarsening would be nice here
                    linop_nodelap->define(
                        Vector<Geometry>{geom},
                        Vector<BoxArray>{amrex::convert(ba,IntVect(0))},
                        Vector<DistributionMapping>{dm},
                        info);
                }

                // Note: this assumes that the beam is propagating along
                // one of the axes of the grid, i.e. that only *one* of the
                // components of `beta` is non-negligible. // we use this
#if defined(WARPX_DIM_RZ)
                linop_nodelap->setRZ(true);
                linop_nodelap->setSigma({0._rt, 1._rt-beta_solver[1]*beta_solver[1]});
#else
                linop_nodelap->setSigma({AMREX_D_DECL(
                            1._rt-beta_solver[0]*beta_solver[0],
                            1._rt-beta_solver[1]*beta_solver[1],
                            1._rt-beta_solver[2]*beta_solver[2])});
#endif
#if defined(AMREX_USE_EB)
                if (eb_enabled) {
                    if constexpr (!std::is_same_v<T_BoundaryHandler, std::nullopt_t>) {
                        // if the EB potential only depends on time, the potential can be passed
                        // as a float instead of a callable
                        if (boundary_handler.phi_EB_only_t) {
                            linop_nodelap->setEBDirichlet(boundary_handler.potential_eb_t(current_time));
                        } else {
                            linop_nodelap->setEBDirichlet(boundary_handler.getPhiEB(current_time));
                        }
                    } else
                    {
                        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE( !is_solver_igf_on_lev0,
                                                            "EB Poisson solver enabled but no 'boundary_handler' passed!");
                    }
                }
#endif
                l_linop = std::move(linop_nodelap);
            } else {
                // In the absence of EB and RZ: use a more generic solver
                // that can handle beams propagating in any direction
                auto linop_tenslap = std::make_unique<MLNodeTensorLaplacian>(
                    Vector<Geometry>{geom},
                    Vector<BoxArray>{amrex::convert(ba,IntVect(0))},
                    Vector<DistributionMapping>{dm},
                    info);
                linop_tenslap->setBeta(beta_solver); // for the non-axis-aligned solver
                l_linop = std::move(linop_tenslap);
            }

            // Level 0 domain boundary
            if constexpr (std::is_same_v<T_BoundaryHandler, std::nullopt_t>) {
                Array<LinOpBCType, AMREX_SPACEDIM> const lobc = {AMREX_D_DECL(
                        LinOpBCType::Dirichlet,
                        LinOpBCType::Dirichlet,
                        LinOpBCType::Dirichlet
                        )};
                Array<LinOpBCType, AMREX_SPACEDIM> const hibc = lobc;
                l_linop->setDomainBC(lobc, hibc);
            } else {
                l_linop->setDomainBC(boundary_handler.lobc, boundary_handler.hibc);
            }

            l_mlmg = std::make_unique<MLMG>(*l_linop);
            l_mlmg->setVerbose(verbosity);
            l_mlmg->setMaxIter(max_iters);
            if (num_final_sweeps) {
                l_mlmg->setFinalSmooth(*num_final_sweeps);
            }
            l_mlmg->setConvergenceNormType(MLMGNormType::greater);
            l_mlmg->setNoGpuSync(Parms{}.no_gpu_sync); // baseline; tuned below

            if (ng) {
                // In this case, computeE needs to use ghost nodes data. So we
                // ask MLMG to fill BC for us after it solves the problem.
                l_mlmg->setFinalFillBC(true);
            }

            return std::make_pair(std::move(l_mlmg), std::move(l_linop));
        };

        // Tuning is expensive, because it performs a number of trial solves.
        // So it is opt-in. Without it, the default parameters are used.
        bool const tune_mlmg = MLMGTuningCache::tuning();
        auto const key = MLMGTuningCache::makeKey(lev, rho.boxArray(), rho.DistributionMap());
        auto const* cached = tune_mlmg ? MLMGTuningCache::find(key) : nullptr;
        if (cached)
        {
            m_parms = *cached;
        }
        else if (tune_mlmg)
        {
            MultiFab phi(rho.boxArray(), rho.DistributionMap(), 1, 1, MFInfo{}, rho.Factory());

            // We want to disable fpe trapping because some of the setups
            // tried below might blow up.
            FPExceptGuard const fpe_guard{};

            auto time_attempt = [&] (auto&& attempt)
            {
                auto t = t_failed;
                try {
                    phi.setVal(0);
                    Gpu::synchronize();
                    auto const t0 = amrex::second();
                    attempt();
                    Gpu::synchronize();
                    auto const t1 = amrex::second();
                    if (phi.contains_nan(0, 1, 0)) {
                        throw amrex::MLMG::error
                            ("ablastr::fields::MLMGWrapper: solution contains NaNs");
                    }
                    t = t1-t0;
                } catch (MLMG::error const&) {
                    t = t_failed; // the solve failed
                }
                amrex::ParallelAllReduce::Max(t, amrex::ParallelContext::CommunicatorSub());
                return t;
            };

            auto tsolve = t_failed;
            int mcl0, niter0;
            bool baseline_ok = false;
            { // base line
                auto tmp = make_mlmg_linop(info0, rho.boxArray(), rho.DistributionMap(), factory);
                tmp.first->setThrowException(true); // trial solve: failure is caught, not fatal
                mcl0 = tmp.second->NMGLevels(0) - 1;
                // The baseline solve itself might fail.
                niter0 = max_iters;
                tsolve = time_attempt([&] {
                    tmp.first->solve({&phi}, {&rho}, relative_tolerance,
                                     absolute_tolerance);
                });
                baseline_ok = (tsolve < t_failed);
                if (baseline_ok) {
                    niter0 = tmp.first->getNumIters();
                }
            }

            // If the baseline converged in zero iterations, there is nothing to tune.
            if (!baseline_ok || (niter0 > 0)) {
                // On GPUs, MLMG can run in a single-stream no-sync region, thus
                // avoiding stream synchronizations at the cost of losing the
                // concurrency of multiple streams. Which one is faster depends
                // on the problem, so we try both. On CPUs, this makes no
                // difference, and we therefore test one option only.
#ifdef AMREX_USE_GPU
                constexpr std::array<bool,2> no_gpu_sync_options{true, false};
#else
                constexpr std::array<bool,1> no_gpu_sync_options{true};
#endif

                // let's test single Box first
                if ((lev == 0) && (rho.size() > 1)
                    && (geom.Domain().numPts() <= (Long(256)*Long(256)*Long(256))))
                {
                    auto& sb = MLMGTuningCache::singleBox(lev, geom.Domain(), rho.ixType());
                    BoxArray const& ba = sb.ba;
                    DistributionMapping const& dm = sb.dm;

                    auto const* fac = factory;
#if defined(AMREX_USE_EB)
                    if (eb_enabled) {
                        if (!sb.eb_factory) {
                            sb.eb_factory = amrex::makeEBFabFactory
                                (static_cast<EBFArrayBoxFactory const*>(factory)->getEBIndexSpace(),
                                 geom, ba, dm, {1,1,1}, EBSupport::full);
                        }
                        fac = sb.eb_factory.get();
                    }
#endif

                    MultiFab phi_sb(ba, dm, 1, 1);
                    MultiFab rho_sb(ba, dm, 1, 0);

                    // For single Box case, agg_grid_size does not matter
                    for (int mcl = 0; mcl <= mcl0; ++mcl) {
                        auto info = info0;
                        info.setMaxCoarseningLevel(mcl);
                        auto tmp = make_mlmg_linop(info, ba, dm, fac);
                        tmp.first->setThrowException(true);
                        tmp.first->setMaxIter(std::min(max_iters, 2*niter0));
                        for (bool const nosync : no_gpu_sync_options) {
                            tmp.first->setNoGpuSync(nosync);
                            double tsolve_tmp = t_failed;
                            Parms parms_tmp{};
                            parms_tmp.single_box = true;
                            parms_tmp.mcl = mcl;
                            parms_tmp.no_gpu_sync = nosync;
                            for (int itry = 0; itry < 3; ++itry) {
                                auto const t_try = time_attempt([&] {
                                    solve_single_box(geom, *tmp.first, ng, phi, rho,
                                                     phi_sb, rho_sb, relative_tolerance,
                                                     absolute_tolerance);
                                });
                                if (t_try == t_failed) {
                                    tsolve_tmp = t_failed;
                                    break;
                                }
                                tsolve_tmp = std::min(tsolve_tmp, t_try);
                            }
                            if (tsolve_tmp < tsolve) {
                                tsolve = tsolve_tmp;
                                m_parms = parms_tmp;
                            }
                        }
                    }
                }

                // Now let's test the cases that we preserve the original grids.
                auto const& ba = rho.boxArray();
                auto const& dm = rho.DistributionMap();
                std::vector<int> agg_grid_size{-1};
                if (rho.size() > 1) {
                    agg_grid_size.push_back(LPInfo::getDefaultAgglomerationGridSize());
                    agg_grid_size.push_back(LPInfo::getDefaultAgglomerationGridSize() * 2);
                    agg_grid_size.push_back(LPInfo::getDefaultAgglomerationGridSize() * 4);
                }
                for (int mcl = 0; mcl <= mcl0; ++mcl) {
                    for (int const ags : agg_grid_size) {
                        // Without coarse levels there is no agglomeration.
                        if ((mcl == 0) && (ags != -1)) { continue; }
                        auto info = info0;
                        info.setMaxCoarseningLevel(mcl).setAgglomerationGridSize(ags);
                        auto tmp = make_mlmg_linop(info, ba, dm, factory);
                        tmp.first->setThrowException(true);
                        tmp.first->setMaxIter(std::min(max_iters, 2*niter0));
                        for (bool const nosync : no_gpu_sync_options) {
                            tmp.first->setNoGpuSync(nosync);
                            double tsolve_tmp = t_failed;
                            Parms parms_tmp{};
                            parms_tmp.single_box = false;
                            parms_tmp.mcl = mcl;
                            parms_tmp.ags = ags;
                            parms_tmp.no_gpu_sync = nosync;
                            for (int itry = 0; itry < 3; ++itry) {
                                auto const t_try = time_attempt([&] {
                                    tmp.first->solve({&phi}, {&rho}, relative_tolerance,
                                                     absolute_tolerance);
                                });
                                if (t_try == t_failed) {
                                    tsolve_tmp = t_failed;
                                    break;
                                }
                                tsolve_tmp = std::min(tsolve_tmp, t_try);
                            }
                            if (tsolve_tmp < tsolve) {
                                tsolve = tsolve_tmp;
                                m_parms = parms_tmp;
                            }
                        }
                    }
                }

                if (baseline_ok || (tsolve < t_failed))
                {
                    MLMGTuningCache::insert(key, m_parms);

                    if (verbosity > 0) {
                        std::string msg("MLMG parameters chosen:");
                        if (m_parms.single_box) { msg += " single_box=true,"; }
                        msg.append(" max_coarsening_level=").append(std::to_string(m_parms.mcl));
                        if (m_parms.ags != -1) {
                            msg.append(" agglomeration_grid_size=").append(std::to_string(m_parms.ags));
                        }
#ifdef AMREX_USE_GPU
                        msg.append(" no_gpu_sync=").append(m_parms.no_gpu_sync ? "true" : "false");
#endif
                        amrex::Print() << msg << '\n';
                    }
                }
                else
                {
                    // None of the setups worked for this rho. We fall back to the
                    // baseline setup (i.e., the default Parms) and hope that the
                    // solver will work later with the real inputs. If it does not,
                    // it will fail then anyway. The result is not cached, so that
                    // the tuning can be attempted again later.
                    m_parms = Parms{};
                    if (verbosity > 0) {
                        amrex::Print() << "MLMG tuning failed. Using the default parameters.\n";
                    }
                }

#if defined(AMREX_USE_EB)
                if (!m_parms.single_box) {
                    MLMGTuningCache::eraseSingleBox(lev);
                }
#endif
            } // end of if (!baseline_ok || (niter0 > 0))
        }

        auto const* fac = factory;
        if (m_parms.single_box) {
            auto& sb = MLMGTuningCache::singleBox(lev, geom.Domain(), rho.ixType());
            m_ba = sb.ba;
            m_dm = sb.dm;
            m_phi_tmp.define(m_ba, m_dm, 1, 1);
            m_phi_tmp.setVal(0); // ghost nodes at Dirichlet faces are never written by MLMG
            m_rho_tmp.define(m_ba, m_dm, 1, 0);
#if defined(AMREX_USE_EB)
            if (eb_enabled) {
                if (!sb.eb_factory) {
                    sb.eb_factory = amrex::makeEBFabFactory
                        (static_cast<EBFArrayBoxFactory const*>(factory)->getEBIndexSpace(),
                         m_geom, m_ba, m_dm, {1,1,1}, EBSupport::full);
                }
                fac = sb.eb_factory.get();
            }
#endif
        } else {
            m_ba = rho.boxArray();
            m_dm = rho.DistributionMap();
        }

        auto info = info0;
        info.setMaxCoarseningLevel(m_parms.mcl).setAgglomerationGridSize(m_parms.ags);

        auto tmp = make_mlmg_linop(info, m_ba, m_dm, fac);
        m_mlmg = std::move(tmp.first);
        m_linop = std::move(tmp.second);
        m_mlmg->setNoGpuSync(m_parms.no_gpu_sync);
    }

    amrex::MLMG& getMLMG () { return *m_mlmg; }

    amrex::MLNodeLinOp& getLinOp () { return *m_linop; }

    void solve (amrex::MultiFab& phi, amrex::MultiFab const& rho,
                amrex::Real relative_tolerance, amrex::Real absolute_tolerance)
    {
        if (m_parms.single_box) {
            solve_single_box(m_geom, *m_mlmg, m_ng, phi, rho, m_phi_tmp, m_rho_tmp,
                             relative_tolerance, absolute_tolerance);
        } else {
            m_mlmg->solve({&phi}, {&rho}, relative_tolerance, absolute_tolerance);
        }
    }

private:

    using Parms = MLMGTuningCache::Parms;

    Parms m_parms;
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    //! Temporaries of the single Box solve. They must be kept alive because
    //! MLMG aliases them and MLMG::getGradSolution in another part of WarpX
    //! will need them.
    amrex::MultiFab m_phi_tmp;
    amrex::MultiFab m_rho_tmp;
    std::unique_ptr<amrex::MLMG> m_mlmg;
    std::unique_ptr<amrex::MLNodeLinOp> m_linop;
    int m_ng;

    //! Elapsed time marking a setup that failed and must not be selected
    static constexpr double t_failed = std::numeric_limits<double>::max();

    //! RAII guard disabling floating point exception trapping.
    struct FPExceptGuard
    {
        FPExceptGuard ()
            : m_prev(amrex::OpenMP::get_max_threads(), amrex::FPExcept::none)
        {
            // Yes, each thread has its own settings.
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {
                m_prev[amrex::OpenMP::get_thread_num()]
                    = amrex::disableFPExcept(amrex::FPExcept::all);
            }
        }

        ~FPExceptGuard ()
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {
                std::feclearexcept(FE_ALL_EXCEPT);
                amrex::setFPExcept(m_prev[amrex::OpenMP::get_thread_num()]);
            }
        }

        FPExceptGuard (FPExceptGuard const&) = delete;
        FPExceptGuard (FPExceptGuard &&) = delete;
        FPExceptGuard& operator= (FPExceptGuard const&) = delete;
        FPExceptGuard& operator= (FPExceptGuard &&) = delete;
    private:
        amrex::Vector<amrex::FPExcept> m_prev;
    };

    //! Solve on a single Box covering the whole domain.
    static void solve_single_box (amrex::Geometry const& geom,
                                  amrex::MLMG& mlmg, int ng,
                                  amrex::MultiFab& phi, amrex::MultiFab const& rho,
                                  amrex::MultiFab& phi_tmp, amrex::MultiFab& rho_tmp,
                                  amrex::Real relative_tolerance,
                                  amrex::Real absolute_tolerance)
    {
        phi_tmp.ParallelCopy(phi, 0, 0, 1);
        rho_tmp.ParallelCopy(rho, 0, 0, 1);
        mlmg.solve({&phi_tmp}, {&rho_tmp}, relative_tolerance, absolute_tolerance);
        int const ngsrc = ng ? 1 : 0;
        int const ngdst = std::min(ngsrc, phi.nGrow());
        phi.ParallelCopy(phi_tmp, 0, 0, 1, ngsrc, ngdst, geom.periodicity());
    }
};

}

#endif

#ifndef ABLASTR_MLMG_WRAPPER_H_
#define ABLASTR_MLMG_WRAPPER_H_

#include <AMReX_Exception.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLNodeTensorLaplacian.H>
#include <AMReX_ParmParse.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <utility>

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

        ParmParse pp("warpx");
        pp.query("verbose", m_verbose);

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
                                    FabFactory<FArrayBox> const* fac)
        {
            std::unique_ptr<amrex::MLMG> l_mlmg;
            std::unique_ptr<amrex::MLNodeLinOp> l_linop;

            if (eb_enabled || is_rz) {
                // In the presence of EB or RZ: the solver assumes that the beam is
                // propagating along  one of the axes of the grid, i.e. that only *one*
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

            l_mlmg->setThrowException(true);

            return std::make_pair<std::unique_ptr<amrex::MLMG>,
                                  std::unique_ptr<amrex::MLNodeLinOp>>
                (std::move(l_mlmg), std::move(l_linop));
        };

        auto key = std::make_pair(lev, rho.getBDKey());
        if (auto it = m_parms_map.find(key); it != m_parms_map.end())
        {
            m_parms = it->second;
        }
        else
        {
            MultiFab phi(rho.boxArray(), rho.DistributionMap(), 1, 1, MFInfo{}, rho.Factory());

            auto tsolve = std::numeric_limits<double>::max();
            int mcl0, niter0;
            bool baseline_ok = false;
            { // base line
                auto tmp = make_mlmg_linop(info0, rho.boxArray(), rho.DistributionMap(), factory);
                mcl0 = tmp.second->NMGLevels(0) - 1;
                // The baseline solve itself might fail (e.g., it might not
                // converge for this rho). In that case, we keep tsolve at its
                // maximal value so that any setup that does work below will be
                // chosen, and we use max_iters for the other setups.
                niter0 = max_iters;
                try {
                    phi.setVal(0);
                    Gpu::synchronize();
                    auto t0 = amrex::second();
                    tmp.first->solve({&phi}, {&rho}, relative_tolerance, absolute_tolerance);
                    Gpu::synchronize();
                    auto t1 = amrex::second();
                    assert_no_nan(phi);
                    tsolve = t1-t0;
                    niter0 = tmp.first->getNumIters();
                    baseline_ok = true;
                } catch (...) {}
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
                    BoxArray ba(geom.Domain());
                    ba.convert(rho.ixType());
                    DistributionMapping dm(Vector<int>{0});

                    auto const* fac = factory;
#if defined(AMREX_USE_EB)
                    std::unique_ptr<EBFArrayBoxFactory> eb_fac;
                    if (eb_enabled) {
                        eb_fac = amrex::makeEBFabFactory
                            (static_cast<EBFArrayBoxFactory const*>(factory)->getEBIndexSpace(),
                             geom, ba, dm, {1,1,1}, EBSupport::full);
                        fac = static_cast<FabFactory<FArrayBox> const*>(eb_fac.get());
                    }
#endif

                    // For single Box case, agg_grid_size does not matter
                    for (int mcl = 0; mcl <= mcl0; ++mcl) {
                        auto info = info0;
                        info.setMaxCoarseningLevel(mcl);
                        auto tmp = make_mlmg_linop(info, ba, dm, fac);
                        tmp.first->setMaxIter(std::min(max_iters, 2*niter0));
                        for (bool nosync : no_gpu_sync_options) {
                            tmp.first->setNoGpuSync(nosync);
                            double tsolve_tmp = std::numeric_limits<double>::max();
                            Parms parms_tmp{};
                            parms_tmp.single_box = true;
                            parms_tmp.mcl = mcl;
                            parms_tmp.no_gpu_sync = nosync;
                            for (int itry = 0; itry < 3; ++itry) {
                                try {
                                    phi.setVal(0);
                                    Gpu::synchronize();
                                    auto t0 = amrex::second();
                                    solve_doit(parms_tmp, ba, dm, geom, *tmp.first, ng, phi, rho,
                                               relative_tolerance, absolute_tolerance);
                                    Gpu::synchronize();
                                    auto t1 = amrex::second();
                                    assert_no_nan(phi);
                                    tsolve_tmp = std::min(tsolve_tmp, t1-t0);
                                } catch (...) {
                                    tsolve_tmp = std::numeric_limits<double>::max();
                                    break;
                                }
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
                    for (int ags : agg_grid_size) {
                        auto info = info0;
                        info.setMaxCoarseningLevel(mcl).setAgglomerationGridSize(ags);
                        auto tmp = make_mlmg_linop(info, ba, dm, factory);
                        tmp.first->setMaxIter(std::min(max_iters, 2*niter0));
                        for (bool nosync : no_gpu_sync_options) {
                            tmp.first->setNoGpuSync(nosync);
                            double tsolve_tmp = std::numeric_limits<double>::max();
                            Parms parms_tmp{};
                            parms_tmp.single_box = false;
                            parms_tmp.mcl = mcl;
                            parms_tmp.ags = ags;
                            parms_tmp.no_gpu_sync = nosync;
                            for (int itry = 0; itry < 3; ++itry) {
                                try {
                                    phi.setVal(0);
                                    Gpu::synchronize();
                                    auto t0 = amrex::second();
                                    solve_doit(parms_tmp, ba, dm, geom, *tmp.first, ng, phi, rho,
                                               relative_tolerance, absolute_tolerance);
                                    Gpu::synchronize();
                                    auto t1 = amrex::second();
                                    assert_no_nan(phi);
                                    tsolve_tmp = std::min(tsolve_tmp, t1-t0);
                                } catch (...) {
                                    tsolve_tmp = std::numeric_limits<double>::max();
                                    break;
                                }
                            }
                            if (tsolve_tmp < tsolve) {
                                tsolve = tsolve_tmp;
                                m_parms = parms_tmp;
                            }
                        }
                    }
                }

                if (baseline_ok || (tsolve < std::numeric_limits<double>::max()))
                {
                    if (m_parms_map.size() > 1000) { m_parms_map.clear(); } // just in case

                    m_parms_map[key] = m_parms;

                    if (m_verbose) {
                        std::string msg("MLMG parameters chosen:");
                        if (m_parms.single_box) { msg += " original boxes agglomerated,"; }
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
                    if (m_verbose) {
                        amrex::Print() << "MLMG tuning failed. Using the default parameters.\n";
                    }
                }
            } // end of if (!baseline_ok || (niter0 > 0))
        }

        auto const* fac = factory;
#if defined(AMREX_USE_EB)
        std::unique_ptr<EBFArrayBoxFactory> eb_fac;
#endif
        if (m_parms.single_box) {
            m_ba = BoxArray(geom.Domain());
            m_ba.convert(rho.ixType());
            m_dm = DistributionMapping(Vector<int>{0});
#if defined(AMREX_USE_EB)
            if (eb_enabled) {
                eb_fac = amrex::makeEBFabFactory
                    (static_cast<EBFArrayBoxFactory const*>(factory)->getEBIndexSpace(),
                     m_geom, m_ba, m_dm, {1,1,1}, EBSupport::full);
                fac = static_cast<FabFactory<FArrayBox> const*>(eb_fac.get());
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
        solve_doit(m_parms, m_ba, m_dm, m_geom, *m_mlmg,  m_ng, phi, rho,
                   relative_tolerance, absolute_tolerance);
    }

private:

    struct Parms {
        bool single_box = false;
        int mcl = 30; // default in amrex
        int ags = -1; // default in amrex
        bool no_gpu_sync = true; // GPU only: single stream and no stream sync
    };

    inline static std::map<std::pair<int,amrex::FabArrayBase::BDKey>, Parms> m_parms_map;

    Parms m_parms;
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    std::unique_ptr<amrex::MLMG> m_mlmg;
    std::unique_ptr<amrex::MLNodeLinOp> m_linop;
    int m_ng;
    int m_verbose = 1;

    /** Throw an exception if the solution contains NaNs.
     *
     * MLMG's convergence test is based on the max norm, which ignores NaNs.
     * Therefore MLMG may report success even though the solution is polluted
     * by NaNs. So we must check the solution explicitly.
     */
    static void assert_no_nan (amrex::MultiFab const& phi)
    {
        if (phi.contains_nan(0, 1, 0)) {
            throw amrex::RuntimeError("ablastr::fields::MLMGWrapper: solution contains NaNs");
        }
    }

    static void solve_doit (Parms const& parms, amrex::BoxArray const& ba,
                            amrex::DistributionMapping const& dm,
                            amrex::Geometry const& geom,
                            amrex::MLMG& mlmg, int ng,
                            amrex::MultiFab& phi, amrex::MultiFab const& rho,
                            amrex::Real relative_tolerance, amrex::Real absolute_tolerance)
    {
        if (parms.single_box) {
            amrex::MultiFab phi_tmp(ba, dm, 1, 1);
            amrex::MultiFab rho_tmp(ba, dm, 1, 0);
            phi_tmp.ParallelCopy(phi, 0, 0, 1);
            rho_tmp.ParallelCopy(rho, 0, 0, 1);
            mlmg.solve({&phi_tmp}, {&rho_tmp}, relative_tolerance, absolute_tolerance);
            int ngsrc = ng ? 1 : 0;
            int ngdst = std::min(ngsrc, phi.nGrow());
            phi.ParallelCopy(phi_tmp, 0, 0, 1, ngsrc, ngdst, geom.periodicity());
        } else {
            mlmg.solve({&phi}, {&rho}, relative_tolerance, absolute_tolerance);
        }
    }
};

}

#endif

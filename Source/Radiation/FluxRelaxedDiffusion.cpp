/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FluxRelaxedDiffusion.H"

#include "Utils/WarpXConst.H"

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <memory>

namespace warpx::radiation
{
using namespace amrex::literals;
namespace
{
using Spacing = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;
using Faces = amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;
using DiagnosticSite = FluxRelaxedDiffusionAdmissibilitySite;

// The key selects one failed valid cell on one rank.  Every numerical field is
// carried with that same key, so a diagnostic reduction never mixes fields from
// different cells.  This structure is report-only and is not used by acceptance.
struct CellDiagnostic
{
    AMREX_GPU_HOST_DEVICE static constexpr CellDiagnostic max () noexcept
    {
        return CellDiagnostic{};
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
    operator< (CellDiagnostic const& other) const noexcept
    {
        return key < other.key;
    }

    amrex::Long key = std::numeric_limits<amrex::Long>::max();
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    amrex::Real input_energy = 0.0;
    amrex::Real input_q[3] = {};
    amrex::Real candidate_energy = 0.0;
    amrex::Real candidate_q[3] = {};
    amrex::Real q2 = 0.0;
    amrex::Real opacity = 0.0;
    amrex::Real beta[3] = {};
    amrex::Real requested_impulse[3] = {};
    amrex::Real requested_work = 0.0;
    amrex::Real rhs = 0.0;
    amrex::Real neighbor_energy_lo[3] = {};
    amrex::Real neighbor_energy_hi[3] = {};
    amrex::Real face_diffusion_lo[3] = {};
    amrex::Real face_diffusion_hi[3] = {};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
physicalDirection (int d) noexcept
{
#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(d);
    return 2;
#elif defined(WARPX_DIM_XZ)
    return d == 0 ? 0 : 2;
#else
    return d;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relativeError (amrex::Real error, amrex::Real scale) noexcept
{
    if (!amrex::Math::isfinite(error) || !amrex::Math::isfinite(scale))
    {
        return std::numeric_limits<amrex::Real>::infinity();
    }
    if (scale == 0)
    {
        return error == 0 ? 0 : std::numeric_limits<amrex::Real>::infinity();
    }
    return std::abs(error) / scale;
}

bool
collectiveValid (bool local)
{
    int invalid = local ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

constexpr char const*
diagnosticSiteName (DiagnosticSite site) noexcept
{
    switch (site)
    {
    case DiagnosticSite::NonfiniteRightHandSide:
        return "nonfinite_rhs";
    case DiagnosticSite::EvaluateState:
        return "evaluate_state";
    case DiagnosticSite::PhotonCone:
        return "photon_cone";
    case DiagnosticSite::RawGlobalLedger:
        return "raw_global_ledger";
    case DiagnosticSite::None:
        break;
    }
    return "none";
}

// Radiation and beta have different component layouts: radiation[0] is E.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
knownFaceFlux (amrex::Array4<amrex::Real const> const& old,
               amrex::Array4<amrex::Real const> const& energy,
               amrex::Array4<amrex::Real const> const& opacity,
               amrex::Array4<amrex::Real const> const& beta, int i, int j, int k, int d,
               amrex::Real dt) noexcept
{
    int left[3] = {i, j, k};
    --left[d];
    int const p = physicalDirection(d);
    auto const chi = 0.5_rt * opacity(left[0], left[1], left[2]) + 0.5_rt * opacity(i, j, k);
    auto const velocity = 0.5_rt * beta(left[0], left[1], left[2], p) + 0.5_rt * beta(i, j, k, p);
    auto const x = chi * PhysConst::c * dt;
    // Donor flux for the old directed component avoids sending a negative
    // upstream energy source out of a newly converted localized beam. This
    // term is first-order in space and vanishes in the stiff diffusion limit.
    auto const old_q = amrex::max(old(left[0], left[1], left[2], p + 1), 0._rt) +
                       amrex::min(old(i, j, k, p + 1), 0._rt);
    auto const upwind = velocity >= 0 ? energy(left[0], left[1], left[2]) : energy(i, j, k);
    return PhysConst::c * (old_q / (1 + x) + (x / (1 + x)) * (4._rt / 3._rt) * velocity * upwind);
}
} // namespace

FluxRelaxedDiffusionResult
TryAdvanceFluxRelaxedDiffusion (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                                amrex::MultiFab const& frozen_beta, amrex::Geometry const& geometry,
                                amrex::Real dt, FluxRelaxedDiffusionOptions const& options,
                                FluxRelaxedWorkCallback const& work_callback)
{
    FluxRelaxedDiffusionResult result;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    constexpr bool cartesian_build = false;
#else
    constexpr bool cartesian_build = true;
#endif
    auto const& boxes = radiation.boxArray();
    auto const& distribution = radiation.DistributionMap();
    result.requested_impulse.define(boxes, distribution, 3, 0);
    result.requested_work.define(boxes, distribution, 1, 0);
    result.requested_impulse.setVal(0);
    result.requested_work.setVal(0);
    auto fail = [&result] (FluxRelaxedDiffusionFailure reason)
    {
        result.failure = reason;
        result.valid = false;
        return std::move(result);
    };
    bool layout_valid =
        cartesian_build && radiation.nComp() == 4 && radiation.nGrowVect().min() >= 1 &&
        radiation.ixType().cellCentered() && opacity.nComp() == 1 && frozen_beta.nComp() == 3 &&
        opacity.boxArray() == boxes && frozen_beta.boxArray() == boxes &&
        opacity.DistributionMap() == distribution &&
        frozen_beta.DistributionMap() == distribution && geometry.Coord() == 0 &&
        geometry.isAllPeriodic() && std::isfinite(dt) && dt > 0 &&
        std::isfinite(PhysConst::c * dt) && static_cast<bool>(work_callback) &&
        options.max_iterations > 0 && options.max_linear_iterations > 0 &&
        std::isfinite(options.relaxation) && options.relaxation > 0 && options.relaxation <= 1 &&
        std::isfinite(options.tolerance) && options.tolerance > 0 &&
        options.tolerance <= 1.e-11_rt && std::isfinite(options.energy_tolerance) &&
        options.energy_tolerance > 0 && options.energy_tolerance <= 1.e-10_rt &&
        std::isfinite(options.linear_tolerance) && options.linear_tolerance > 0 &&
        options.linear_tolerance <= options.tolerance && std::isfinite(options.maximum_beta) &&
        options.maximum_beta > 0 && options.maximum_beta <= 1.e-3_rt &&
        std::isfinite(options.maximum_advective_cfl) && options.maximum_advective_cfl > 0 &&
        options.maximum_advective_cfl <= 0.5_rt;
    if (!collectiveValid(layout_valid))
    {
        return fail(FluxRelaxedDiffusionFailure::InvalidState);
    }
    Spacing const dx = geometry.CellSizeArray();
    amrex::ReduceOps<amrex::ReduceOpMax> input_ops;
    amrex::ReduceData<int> input_data(input_ops);
    auto const maximum_beta = options.maximum_beta;
    auto const maximum_cfl = options.maximum_advective_cfl;
    for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi)
    {
        auto const r = radiation.const_array(mfi);
        auto const chi = opacity.const_array(mfi);
        auto const beta = frozen_beta.const_array(mfi);
        input_ops.eval(mfi.validbox(), input_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                       {
                           bool bad = !amrex::Math::isfinite(r(i, j, k, 0)) || r(i, j, k, 0) < 0 ||
                                      !amrex::Math::isfinite(chi(i, j, k)) || chi(i, j, k) < 0 ||
                                      !amrex::Math::isfinite(chi(i, j, k) * PhysConst::c * dt);
                           amrex::Real q2 = 0, beta2 = 0, cfl = 0;
                           for (int p = 0; p < 3; ++p)
                           {
                               auto const q = r(i, j, k, p + 1);
                               bad = bad || !amrex::Math::isfinite(q) ||
                                     !amrex::Math::isfinite(beta(i, j, k, p));
                               beta2 += beta(i, j, k, p) * beta(i, j, k, p);
                               if (r(i, j, k, 0) > 0)
                               {
                                   q2 += (q / r(i, j, k, 0)) * (q / r(i, j, k, 0));
                               }
                               else
                               {
                                   bad = bad || q != 0;
                               }
                           }
                           for (int d = 0; d < AMREX_SPACEDIM; ++d)
                           {
                               cfl += (4._rt / 3._rt) *
                                      std::abs(beta(i, j, k, physicalDirection(d))) * PhysConst::c *
                                      dt / dx[d];
                           }
                           return {bad || !amrex::Math::isfinite(q2) ||
                                   q2 > 1 + 128 * std::numeric_limits<amrex::Real>::epsilon() ||
                                   beta2 > maximum_beta * maximum_beta || cfl > maximum_cfl};
                       });
    }
    if (!collectiveValid(amrex::get<0>(input_data.value()) == 0))
    {
        return fail(FluxRelaxedDiffusionFailure::InvalidState);
    }

    amrex::MultiFab old(boxes, distribution, 4, 1);
    amrex::MultiFab chi(boxes, distribution, 1, 1), beta(boxes, distribution, 3, 1);
    amrex::MultiFab energy(boxes, distribution, 1, 1), next(boxes, distribution, 1, 1);
    amrex::MultiFab rhs(boxes, distribution, 1, 0), a(boxes, distribution, 1, 0);
    amrex::MultiFab qnew(boxes, distribution, 3, 0);
    amrex::MultiFab impulse(boxes, distribution, 3, 0), work(boxes, distribution, 1, 0);
    amrex::MultiFab::Copy(old, radiation, 0, 0, 4, 0);
    amrex::MultiFab::Copy(energy, radiation, 0, 0, 1, 0);
    amrex::MultiFab::Copy(chi, opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(beta, frozen_beta, 0, 0, 3, 0);
    old.FillBoundary(geometry.periodicity());
    chi.FillBoundary(geometry.periodicity());
    beta.FillBoundary(geometry.periodicity());
    a.setVal(1);
    Faces diffusion, known_flux, pressure;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
        diffusion[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        known_flux[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        pressure[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        for (amrex::MFIter mfi(*diffusion[d]); mfi.isValid(); ++mfi)
        {
            auto const alpha = chi.const_array(mfi);
            auto const b = diffusion[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   int left[3] = {i, j, k};
                                   --left[d];
                                   auto const face_chi = 0.5_rt * alpha(i, j, k) +
                                                         0.5_rt * alpha(left[0], left[1], left[2]);
                                   b(i, j, k) =
                                       PhysConst::c *
                                       (PhysConst::c * dt / (1 + face_chi * PhysConst::c * dt)) / 3;
                               });
        }
    }
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> boundary;
    boundary.fill(amrex::LinOpBCType::Periodic);
    amrex::MLABecLaplacian op({geometry}, {boxes}, {distribution});
    op.setDomainBC(boundary, boundary);
    op.setLevelBC(0, nullptr);
    op.setScalars(1, dt);
    op.setACoeffs(0, a);
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(diffusion));
    amrex::MLMG solver(op);
    solver.setVerbose(options.verbose ? 1 : 0);
    solver.setMaxIter(options.max_linear_iterations);
    solver.setThrowException(true);

    auto fill_faces = [&] (amrex::MultiFab& e)
    {
        e.FillBoundary(geometry.periodicity());
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            for (amrex::MFIter mfi(*known_flux[d]); mfi.isValid(); ++mfi)
            {
                auto const en = e.const_array(mfi), original = old.const_array(mfi);
                auto const alpha = chi.const_array(mfi), velocity = beta.const_array(mfi);
                auto const flux = known_flux[d]->array(mfi), p = pressure[d]->array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        int left[3] = {i, j, k};
                        --left[d];
                        flux(i, j, k) =
                            knownFaceFlux(original, en, alpha, velocity, i, j, k, d, dt);
                        p(i, j, k) =
                            (0.5_rt * en(i, j, k) + 0.5_rt * en(left[0], left[1], left[2])) / 3;
                    });
            }
        }
    };
    auto stage_work = [&] (amrex::MultiFab const& e)
    {
        for (amrex::MFIter mfi(e); mfi.isValid(); ++mfi)
        {
            auto const en = e.const_array(mfi), original = old.const_array(mfi);
            auto const alpha = chi.const_array(mfi), velocity = beta.const_array(mfi);
            auto const q = qnew.array(mfi), request = impulse.array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> p;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                p[d] = pressure[d]->const_array(mfi);
            }
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::GpuArray<amrex::Real, 3> pressure_delta{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        int hi[3] = {i, j, k};
                        ++hi[d];
                        pressure_delta[physicalDirection(d)] =
                            PhysConst::c * dt / dx[d] * (p[d](hi[0], hi[1], hi[2]) - p[d](i, j, k));
                    }
                    auto const x = alpha(i, j, k) * PhysConst::c * dt;
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const equilibrium =
                            (4._rt / 3._rt) * en(i, j, k) * velocity(i, j, k, d);
                        auto const departure =
                            original(i, j, k, d + 1) - pressure_delta[d] - equilibrium;
                        q(i, j, k, d) = equilibrium + departure / (1 + x);
                        request(i, j, k, d) = (x / (1 + x)) / PhysConst::c * departure;
                    }
                });
        }
        work.setVal(0);
        return collectiveValid(work_callback(impulse, work));
    };

    amrex::Real absolute_equation_error = 0;
    auto const input_energy_norm = old.norm0(0, 0);
    auto report_cell_diagnostic = [&] (DiagnosticSite site)
    {
        amrex::ReduceOps<amrex::ReduceOpMin> ops;
        amrex::ReduceData<CellDiagnostic> data(ops);
        auto const rank = amrex::ParallelDescriptor::MyProc();
        auto const domain = geometry.Domain();
        amrex::GpuArray<amrex::Long, 3> domain_begin{};
        amrex::GpuArray<amrex::Long, 3> domain_shape{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            domain_begin[d] = domain.smallEnd(d);
            domain_shape[d] = domain.length(d);
        }
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
        {
            auto const candidate = energy.const_array(mfi), original = old.const_array(mfi);
            auto const q = qnew.const_array(mfi), request = impulse.const_array(mfi);
            auto const debit = work.const_array(mfi), alpha = chi.const_array(mfi);
            auto const velocity = beta.const_array(mfi), right = rhs.const_array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> b;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                b[d] = diffusion[d]->const_array(mfi);
            }
            ops.eval(
                mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<CellDiagnostic>
                {
                    bool bad = false;
                    if (site == DiagnosticSite::NonfiniteRightHandSide)
                    {
                        bad = !amrex::Math::isfinite(right(i, j, k));
                    }
                    else if (site == DiagnosticSite::EvaluateState)
                    {
                        bad = !amrex::Math::isfinite(candidate(i, j, k)) ||
                              candidate(i, j, k) < 0 ||
                              !amrex::Math::isfinite(debit(i, j, k));
                        for (int d = 0; d < 3; ++d)
                        {
                            bad = bad || !amrex::Math::isfinite(q(i, j, k, d)) ||
                                  !amrex::Math::isfinite(request(i, j, k, d));
                        }
                    }
                    else if (site == DiagnosticSite::PhotonCone)
                    {
                        amrex::Real photon_q2 = 0.0;
                        bool nonzero_vacuum_q = false;
                        for (int d = 0; d < 3; ++d)
                        {
                            if (candidate(i, j, k) > 0)
                            {
                                photon_q2 += (q(i, j, k, d) / candidate(i, j, k)) *
                                             (q(i, j, k, d) / candidate(i, j, k));
                            }
                            else
                            {
                                nonzero_vacuum_q = nonzero_vacuum_q || q(i, j, k, d) != 0;
                            }
                        }
                        bad = nonzero_vacuum_q || !amrex::Math::isfinite(photon_q2) ||
                              photon_q2 >
                                  1 + 128 * std::numeric_limits<amrex::Real>::epsilon();
                    }
                    if (!bad)
                    {
                        return {CellDiagnostic::max()};
                    }

                    CellDiagnostic diagnostic;
                    amrex::Long cell_index = static_cast<amrex::Long>(i) - domain_begin[0] +
                                             domain_shape[0] *
                                                 (static_cast<amrex::Long>(j) -
                                                  domain_begin[1] +
                                                  domain_shape[1] *
                                                      (static_cast<amrex::Long>(k) -
                                                       domain_begin[2]));
                    diagnostic.key = cell_index;
                    diagnostic.rank = rank;
                    diagnostic.i = i;
                    diagnostic.j = j;
                    diagnostic.k = k;
                    diagnostic.input_energy = original(i, j, k, 0);
                    diagnostic.candidate_energy = candidate(i, j, k);
                    diagnostic.opacity = alpha(i, j, k);
                    diagnostic.requested_work = debit(i, j, k);
                    diagnostic.rhs = site == DiagnosticSite::NonfiniteRightHandSide
                                         ? right(i, j, k)
                                         : 0.0;
                    diagnostic.q2 = 0.0;
                    for (int d = 0; d < 3; ++d)
                    {
                        diagnostic.input_q[d] = original(i, j, k, d + 1);
                        diagnostic.candidate_q[d] = q(i, j, k, d);
                        diagnostic.beta[d] = velocity(i, j, k, d);
                        diagnostic.requested_impulse[d] = request(i, j, k, d);
                        if (candidate(i, j, k) > 0)
                        {
                            diagnostic.q2 += (q(i, j, k, d) / candidate(i, j, k)) *
                                             (q(i, j, k, d) / candidate(i, j, k));
                        }
                    }
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        int lo[3] = {i, j, k}, hi[3] = {i, j, k};
                        --lo[d];
                        ++hi[d];
                        diagnostic.neighbor_energy_lo[d] = candidate(lo[0], lo[1], lo[2]);
                        diagnostic.neighbor_energy_hi[d] = candidate(hi[0], hi[1], hi[2]);
                        diagnostic.face_diffusion_lo[d] = b[d](i, j, k);
                        diagnostic.face_diffusion_hi[d] = b[d](hi[0], hi[1], hi[2]);
                    }
                    return {diagnostic};
                });
        }

        auto const local_tuple = data.value();
        auto const& local = amrex::get<0>(local_tuple);
        auto selected_key = local.key;
        amrex::ParallelDescriptor::ReduceLongMin(selected_key);
        if (selected_key == std::numeric_limits<amrex::Long>::max() ||
            local.key != selected_key)
        {
            return;
        }

        auto const norm_q = std::sqrt(local.q2);
        auto diagnostic_output = amrex::Print(local.rank);
        diagnostic_output.SetPrecision(17)
            << "Flux-relaxed admissibility site=" << diagnosticSiteName(site)
            << " predicate=";
        if (site == DiagnosticSite::NonfiniteRightHandSide)
        {
            diagnostic_output << "!isfinite(rhs)";
        }
        else if (site == DiagnosticSite::EvaluateState)
        {
            diagnostic_output
                << "!isfinite(E_new) || E_new<0 || !isfinite(work) || "
                   "any_d(!isfinite(q_new[d]) || !isfinite(impulse[d]))";
        }
        else
        {
            diagnostic_output
                << "(E_new<=0 && any q_new[d]!=0) || !isfinite(q2) || q2>"
                << 1 + 128 * std::numeric_limits<amrex::Real>::epsilon();
        }
        diagnostic_output
            << " rank=" << local.rank << " cell=(" << local.i << ',' << local.j << ','
            << local.k << ") dt_s=" << dt << " dx_m=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            diagnostic_output << (d == 0 ? "" : ",") << dx[d];
        }
        diagnostic_output
            << ") chi_1_m=" << local.opacity << " beta=(" << local.beta[0] << ','
            << local.beta[1] << ',' << local.beta[2] << ") input_E_J=" << local.input_energy
            << " input_q_J=(" << local.input_q[0] << ',' << local.input_q[1] << ','
            << local.input_q[2] << ") candidate_E_J=" << local.candidate_energy
            << " candidate_q_J=(" << local.candidate_q[0] << ',' << local.candidate_q[1]
            << ',' << local.candidate_q[2] << ") q2=" << local.q2
            << " norm_q_over_E=" << norm_q << " requested_impulse_kg_m_s=("
            << local.requested_impulse[0] << ',' << local.requested_impulse[1] << ','
            << local.requested_impulse[2] << ") requested_work_J=" << local.requested_work;
        if (site == DiagnosticSite::NonfiniteRightHandSide)
        {
            diagnostic_output << " rhs_J=" << local.rhs;
        }
        diagnostic_output << " neighbor_E_lo_J=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            diagnostic_output << (d == 0 ? "" : ",") << local.neighbor_energy_lo[d];
        }
        diagnostic_output << ") neighbor_E_hi_J=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            diagnostic_output << (d == 0 ? "" : ",") << local.neighbor_energy_hi[d];
        }
        diagnostic_output << ") face_diffusion_lo_m2_s=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            diagnostic_output << (d == 0 ? "" : ",") << local.face_diffusion_lo[d];
        }
        diagnostic_output << ") face_diffusion_hi_m2_s=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            diagnostic_output << (d == 0 ? "" : ",") << local.face_diffusion_hi[d];
        }
        diagnostic_output << ")\n";
    };
    auto evaluate = [&] (amrex::MultiFab const& e)
    {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpMax, amrex::ReduceOpMax>
            ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, int> data(ops);
        for (amrex::MFIter mfi(e); mfi.isValid(); ++mfi)
        {
            auto const en = e.const_array(mfi), original = old.const_array(mfi);
            auto const q = qnew.const_array(mfi), request = impulse.const_array(mfi);
            auto const debit = work.const_array(mfi), alpha = chi.const_array(mfi);
            auto const velocity = beta.const_array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> f, b, p;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                f[d] = known_flux[d]->const_array(mfi);
                b[d] = diffusion[d]->const_array(mfi);
                p[d] = pressure[d]->const_array(mfi);
            }
            ops.eval(
                mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, int>
                {
                    amrex::Real transport = 0, momentum_error = 0, collision_error = 0;
                    amrex::GpuArray<amrex::Real, 3> pressure_delta{};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        int hi[3] = {i, j, k}, lo[3] = {i, j, k};
                        ++hi[d];
                        --lo[d];
                        auto const flux_hi = f[d](hi[0], hi[1], hi[2]) -
                                             b[d](hi[0], hi[1], hi[2]) *
                                                 (en(hi[0], hi[1], hi[2]) - en(i, j, k)) / dx[d];
                        auto const flux_lo =
                            f[d](i, j, k) -
                            b[d](i, j, k) * (en(i, j, k) - en(lo[0], lo[1], lo[2])) / dx[d];
                        transport += dt * (flux_hi - flux_lo) / dx[d];
                        pressure_delta[physicalDirection(d)] =
                            PhysConst::c * dt / dx[d] * (p[d](hi[0], hi[1], hi[2]) - p[d](i, j, k));
                    }
                    auto const e_error =
                        en(i, j, k) - original(i, j, k, 0) + transport + debit(i, j, k);
                    auto const e_scale = std::abs(en(i, j, k)) + std::abs(original(i, j, k, 0)) +
                                         std::abs(transport) + std::abs(debit(i, j, k));
                    bool bad = !amrex::Math::isfinite(en(i, j, k)) || en(i, j, k) < 0 ||
                               !amrex::Math::isfinite(debit(i, j, k));
                    auto const x = alpha(i, j, k) * PhysConst::c * dt;
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const impulse_q = PhysConst::c * request(i, j, k, d);
                        auto const error = q(i, j, k, d) - original(i, j, k, d + 1) +
                                           pressure_delta[d] + impulse_q;
                        auto const scale = std::abs(q(i, j, k, d)) +
                                           std::abs(original(i, j, k, d + 1)) +
                                           std::abs(pressure_delta[d]) + std::abs(impulse_q);
                        momentum_error = amrex::max(momentum_error, relativeError(error, scale));
                        auto const equilibrium =
                            (4._rt / 3._rt) * en(i, j, k) * velocity(i, j, k, d);
                        auto const expected =
                            x * (original(i, j, k, d + 1) - pressure_delta[d] - equilibrium);
                        auto const represented = (1 + x) * impulse_q;
                        collision_error =
                            amrex::max(collision_error,
                                       relativeError(represented - expected,
                                                     std::abs(represented) + std::abs(expected)));
                        bad = bad || !amrex::Math::isfinite(q(i, j, k, d)) ||
                              !amrex::Math::isfinite(request(i, j, k, d));
                    }
                    return {relativeError(e_error, e_scale), momentum_error, collision_error,
                            std::abs(e_error), bad};
                });
        }
        auto const reduced = data.value();
        amrex::GpuArray<amrex::Real, 4> errors{amrex::get<0>(reduced), amrex::get<1>(reduced),
                                               amrex::get<2>(reduced), amrex::get<3>(reduced)};
        amrex::ParallelDescriptor::ReduceRealMax(errors.data(), 4);
        result.energy_equation_residual = errors[0];
        result.momentum_equation_residual = errors[1];
        result.collision_identity_residual = errors[2];
        absolute_equation_error = errors[3];
        result.energy_equation_absolute_residual = absolute_equation_error;
        return collectiveValid(amrex::get<4>(reduced) == 0);
    };

    bool converged = false;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        result.iterations = iteration + 1;
        fill_faces(energy);
        if (!stage_work(energy))
        {
            return fail(FluxRelaxedDiffusionFailure::InvalidCallback);
        }
        amrex::ReduceOps<amrex::ReduceOpMax> rhs_ops;
        amrex::ReduceData<int> rhs_data(rhs_ops);
        for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
        {
            auto const original = old.const_array(mfi), debit = work.const_array(mfi);
            auto const right = rhs.array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> f;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                f[d] = known_flux[d]->const_array(mfi);
            }
            rhs_ops.eval(mfi.validbox(), rhs_data,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                         {
                             amrex::Real divergence = 0;
                             for (int d = 0; d < AMREX_SPACEDIM; ++d)
                             {
                                 int hi[3] = {i, j, k};
                                 ++hi[d];
                                 divergence += (f[d](hi[0], hi[1], hi[2]) - f[d](i, j, k)) / dx[d];
                             }
                             right(i, j, k) =
                                 original(i, j, k, 0) - dt * divergence - debit(i, j, k);
                             return {!amrex::Math::isfinite(right(i, j, k))};
                         });
        }
        if (!collectiveValid(amrex::get<0>(rhs_data.value()) == 0))
        {
            result.admissibility_site =
                FluxRelaxedDiffusionAdmissibilitySite::NonfiniteRightHandSide;
            if (options.verbose)
            {
                report_cell_diagnostic(FluxRelaxedDiffusionAdmissibilitySite::
                                           NonfiniteRightHandSide);
            }
            return fail(FluxRelaxedDiffusionFailure::Admissibility);
        }
        amrex::MultiFab::Copy(next, energy, 0, 0, 1, 1);
        try
        {
            solver.solve({&next}, {&rhs}, options.linear_tolerance, 0);
        }
        catch (amrex::MLMG::error const&)
        {
            return fail(FluxRelaxedDiffusionFailure::LinearConvergence);
        }
        auto const relaxation = options.relaxation;
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
        {
            auto const previous = energy.array(mfi);
            auto const solved = next.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   previous(i, j, k) = (1 - relaxation) * previous(i, j, k) +
                                                       relaxation * solved(i, j, k);
                               });
        }
        fill_faces(energy);
        if (!stage_work(energy))
        {
            return fail(FluxRelaxedDiffusionFailure::InvalidCallback);
        }
        if (!evaluate(energy))
        {
            result.admissibility_site = FluxRelaxedDiffusionAdmissibilitySite::EvaluateState;
            if (options.verbose)
            {
                report_cell_diagnostic(FluxRelaxedDiffusionAdmissibilitySite::EvaluateState);
            }
            return fail(FluxRelaxedDiffusionFailure::Admissibility);
        }
        if (options.verbose)
        {
            amrex::Print() << "Flux-relaxed iteration " << result.iterations
                           << ": energy=" << result.energy_equation_residual
                           << " absolute_energy_J=" << absolute_equation_error
                           << " max_input_energy_J=" << input_energy_norm
                           << " momentum=" << result.momentum_equation_residual
                           << " collision=" << result.collision_identity_residual << '\n';
        }
        // A local relative residual is ill-conditioned in the near-zero
        // elliptic tail. Permit only a machine-roundoff absolute floor tied
        // to the physical input energy, not to c, a material energy, or 1 J.
        // The raw global ledger and the photon cone are still checked below.
        auto const energy_roundoff =
            32 * std::numeric_limits<amrex::Real>::epsilon() * input_energy_norm;
        result.energy_roundoff_bound = energy_roundoff;
        if ((result.energy_equation_residual <= options.tolerance ||
             absolute_equation_error <= energy_roundoff) &&
            result.momentum_equation_residual <= options.tolerance &&
            result.collision_identity_residual <= options.tolerance)
        {
            converged = true;
            break;
        }
    }
    if (!converged)
    {
        return fail(FluxRelaxedDiffusionFailure::Iteration);
    }

    amrex::ReduceOps<amrex::ReduceOpMax> cone_ops;
    amrex::ReduceData<int> cone_data(cone_ops);
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
    {
        auto const en = energy.const_array(mfi), q = qnew.const_array(mfi);
        cone_ops.eval(mfi.validbox(), cone_data,
                      [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                      {
                          amrex::Real q2 = 0;
                          bool bad = false;
                          for (int d = 0; d < 3; ++d)
                          {
                              if (en(i, j, k) > 0)
                              {
                                  q2 +=
                                      (q(i, j, k, d) / en(i, j, k)) * (q(i, j, k, d) / en(i, j, k));
                              }
                              else
                              {
                                  bad = bad || q(i, j, k, d) != 0;
                              }
                          }
                          return {bad || !amrex::Math::isfinite(q2) ||
                                  q2 > 1 + 128 * std::numeric_limits<amrex::Real>::epsilon()};
                      });
    }
    if (!collectiveValid(amrex::get<0>(cone_data.value()) == 0))
    {
        result.admissibility_site = FluxRelaxedDiffusionAdmissibilitySite::PhotonCone;
        if (options.verbose)
        {
            report_cell_diagnostic(FluxRelaxedDiffusionAdmissibilitySite::PhotonCone);
        }
        return fail(FluxRelaxedDiffusionFailure::Admissibility);
    }
    amrex::MultiFab delta(boxes, distribution, 1, 0);
    amrex::MultiFab::Copy(delta, energy, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(delta, old, 0, 0, 1, 0);
    auto const energy_error = delta.sum(0) + work.sum(0);
    auto const energy_scale = energy.norm1(0) + old.norm1(0) + work.norm1(0);
    result.raw_energy_residual = std::abs(energy_error);
    bool raw_valid = relativeError(energy_error, energy_scale) <= options.energy_tolerance;
    amrex::Real raw_momentum_errors[3] = {};
    amrex::Real raw_momentum_scales[3] = {};
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(delta, qnew, d, 0, 1, 0);
        amrex::MultiFab::Subtract(delta, old, d + 1, 0, 1, 0);
        auto const error = delta.sum(0) + PhysConst::c * impulse.sum(d);
        auto const scale = qnew.norm1(d) + old.norm1(d + 1) + PhysConst::c * impulse.norm1(d);
        raw_momentum_errors[d] = error;
        raw_momentum_scales[d] = scale;
        result.raw_momentum_residual = amrex::max(result.raw_momentum_residual, std::abs(error));
        raw_valid = raw_valid && relativeError(error, scale) <= options.energy_tolerance;
    }
    if (!raw_valid)
    {
        result.admissibility_site = FluxRelaxedDiffusionAdmissibilitySite::RawGlobalLedger;
        if (options.verbose)
        {
            amrex::Print()
                << "Flux-relaxed admissibility site=raw_global_ledger predicate="
                   "any_d(relative_error(raw_residual_d,raw_scale_d)>energy_tolerance)"
                << " energy_residual_J=" << energy_error << " energy_scale_J=" << energy_scale
                << " momentum_residuals_J=(" << raw_momentum_errors[0] << ','
                << raw_momentum_errors[1] << ',' << raw_momentum_errors[2]
                << ") momentum_scales_J=(" << raw_momentum_scales[0] << ','
                << raw_momentum_scales[1] << ',' << raw_momentum_scales[2]
                << ") energy_tolerance=" << options.energy_tolerance << '\n';
        }
        return fail(FluxRelaxedDiffusionFailure::Admissibility);
    }
    // The callback retained by the caller corresponds to this exact E/q state.
    amrex::MultiFab::Copy(radiation, energy, 0, 0, 1, 0);
    amrex::MultiFab::Copy(radiation, qnew, 0, 1, 3, 0);
    radiation.FillBoundary(geometry.periodicity());
    amrex::MultiFab::Copy(result.requested_impulse, impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(result.requested_work, work, 0, 0, 1, 0);
    result.valid = true;
    return result;
}
} // namespace warpx::radiation

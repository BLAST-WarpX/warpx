/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FixedShapeCurrent.H"

#include "WarpX.H"

#ifdef WARPX_DIM_3D
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <typeinfo>

using namespace amrex::literals;
using amrex::IntVect;
using amrex::Real;
using warpx::fields::FieldType;

namespace {
void
initialize_centers (WarpX& w, amrex::iMultiFab& centers) {
    auto const box = w.Geom(0).Domain();
    auto const filter =
        WarpX::use_filter ? WarpX::filter_npass_each_dir : IntVect(0);
    auto const ratio = w.refRatio(0);
    // Round the fine stencil radius up in each parent direction.
    auto const reach = amrex::coarsen(
        IntVect((WarpX::nox + 1) / 2 + 1) + filter + ratio - IntVect(1), ratio);
    if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor) {
        // Villasenor uses the same fixed shape in the parent's regular and
        // material cells. Its virtual current is restricted before the native
        // macroscopic and EB field update. WarpX scrapes particles at the EB
        // after that update. The refinement patch and its collar are checked
        // separately; only the physical box can truncate this source stencil.
        centers.setVal(0);
        auto const interior = amrex::grow(box, -reach);
        if (interior.ok()) { centers.setVal(1, interior, 0, 1); }
        return;
    }
    // The native Esirkepov operator changes shape near an embedded boundary.
    // Its virtual fixed-shape counterpart still requires a vacuum collar.
    auto const map = centers.DistributionMap();
    amrex::iMultiFab vacuum(amrex::BoxArray(box), map, 1, 0);
    auto const good = vacuum.array(0);
    auto* mat = w.get_pointer_MacroscopicProperties(0);
    bool const has_eb = EB::enabled(), has_mat = (mat != nullptr);
    auto const e = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    for (amrex::MFIter it(*e[0]); it.isValid(); ++it) {
        amrex::GpuArray<amrex::Array4<int const>, 3> ef{}, bf{};
        amrex::Array4<Real const> sigma, mu, eps;
        if (has_eb) {
            for (int d = 0; d < 3; d++) {
                ef[d] = w.GetEBUpdateEFlag()[0][d]->const_array(it);
                bf[d] = w.GetEBUpdateBFlag()[0][d]->const_array(it);
            }
        }
        if (has_mat) {
            sigma = mat->getsigma_mf().const_array(it);
            mu = mat->getmu_mf().const_array(it);
            eps = mat->getepsilon_mf().const_array(it);
        }
        amrex::ParallelFor(
            amrex::enclosedCells(it.validbox()),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                IntVect const p{i, j, k};
                bool ok =
                    !has_mat || (sigma(p) == 0._rt && mu(p) == PhysConst::mu0 &&
                                 eps(p) == PhysConst::epsilon_0);
                if (has_eb) {
                    for (int d = 0; d < 3; ++d) {
                        auto const u = IntVect::TheDimensionVector((d + 1) % 3);
                        auto const v = IntVect::TheDimensionVector((d + 2) % 3);
                        ok = ok && ef[d](p) && ef[d](p + u) && ef[d](p + v) &&
                             ef[d](p + u + v) && bf[d](p) &&
                             bf[d](p + IntVect::TheDimensionVector(d));
                    }
                }
                good(p) = int(ok);
            });
    }
    auto const valid = centers.array(0);
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        IntVect const p{i, j, k};
        bool ok = box.contains(p - reach) && box.contains(p + reach);
        for (int z = -reach[2]; ok && z <= reach[2]; ++z) {
            for (int y = -reach[1]; ok && y <= reach[1]; ++y) {
                for (int x = -reach[0]; ok && x <= reach[0]; ++x) {
                    ok = good(i + x, j + y, k + z) != 0;
                }
            }
        }
        valid(p) = int(ok);
    });
    amrex::Gpu::streamSynchronize();
}

template <typename F>
void
for_tiles (WarpX& w, F const& f) {
    auto& pcs = w.GetPartContainer();
    for (int s = 0; s < pcs.nSpecies(); ++s) {
        auto& pc = pcs.GetParticleContainer(s);
        if (!pc.isPushed() || pc.do_not_deposit || pc.getCharge() == 0._rt) {
            continue;
        }
        for (int lev = 0; lev <= 1; ++lev) {
            for (WarpXParIter it(pc, lev); it.isValid(); ++it) {
                f(it, pc.getCharge());
            }
        }
    }
}

template <int order>
void
deposit (WarpXParIter const& it, Real q, Real dt, amrex::Box const& box,
         amrex::XDim3 const& dinv, amrex::XDim3 const& lower,
         std::array<amrex::Array4<Real>, 3> const& j) {
    auto const pos = GetParticlePosition<PIdx>(it);
    auto const* weight = it.GetAttribs(PIdx::w).dataPtr();
    auto const* ux = it.GetAttribs(PIdx::ux).dataPtr();
    auto const* uy = it.GetAttribs(PIdx::uy).dataPtr();
    auto const* uz = it.GetAttribs(PIdx::uz).dataPtr();
    auto const n = it.numParticles();
    if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov) {
        doEsirkepovDepositionShapeN<order>(
            pos, weight, ux, uy, uz, nullptr, j[0], j[1], j[2], n, dt,
            -.5_rt * dt, dinv, lower, amrex::lbound(box), q, 1,
            amrex::Array4<int const>{}, false);
    } else {
        amrex::GpuArray<amrex::GpuArray<double, 2>, 3> bounds{};
        for (int d = 0; d < 3; ++d) {
            bounds[d][1] = box.length(d);
        }
        amrex::GpuArray<amrex::GpuArray<bool, 2>, 3> const crop{};
        doVillasenorDepositionShapeNExplicit<order>(
            pos, weight, ux, uy, uz, nullptr, j[0], j[1], j[2], n, dt,
            -.5_rt * dt, dinv, lower, bounds, crop, amrex::lbound(box), q, 1);
    }
}
} // namespace

FixedShapeCurrent::FixedShapeCurrent (WarpX& w) {
    auto& pcs = w.GetPartContainer();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        pcs.nContainers() == pcs.nSpecies() &&
            (WarpX::nox == 1 || WarpX::nox == 3) && WarpX::nox == WarpX::noy &&
            WarpX::nox == WarpX::noz &&
            (WarpX::current_deposition_algo ==
                 CurrentDepositionAlgo::Esirkepov ||
             WarpX::current_deposition_algo ==
                 CurrentDepositionAlgo::Villasenor),
        "The fixed-shape current control requires ordinary kinetic species and "
        "linear or cubic charge-conserving deposition.");
    for (int s = 0; s < pcs.nSpecies(); ++s) {
        auto& pc = pcs.GetParticleContainer(s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            typeid(pc) == typeid(PhysicalParticleContainer) &&
                !pc.doContinuousInjection() && !pc.DoFieldIonization() &&
                !pc.do_splitting,
            "The fixed-shape current control requires ordinary physical "
            "particles with fixed species charge.");
    }
    bool use_gather = false;
    amrex::ParmParse("warpx").query("composite_yee_fixed_shape_gather",
                                    use_gather);
    auto const domain = w.Geom(1).Domain();
    amrex::DistributionMapping const dm(amrex::Vector<int>{0});
    valid_centers.define(amrex::BoxArray(w.Geom(0).Domain()), dm, 1, 0);
    initialize_centers(w, valid_centers);
    auto const fine = w.m_fields.get_alldirs(FieldType::current_fp, 1);
    for (int d = 0; d < 3; ++d) {
        amrex::BoxArray const ba(amrex::convert(domain, fine[d]->ixType()));
        raw[d] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
        if (WarpX::use_filter) {
            filtered[d] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
        }
        fine_owns[d] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, 0);
        fine_owns[d]->setVal(0);
        if (use_gather) {
            for (int s = 0; s < 2; ++s) {
                auto const type =
                    s ? FieldType::Bfield_fp : FieldType::Efield_fp;
                auto const ix =
                    w.m_fields.get(type, ablastr::fields::Direction{d}, 0)
                        ->ixType();
                coarse_gather[s][d] = std::make_unique<amrex::MultiFab>(
                    amrex::BoxArray(amrex::convert(w.Geom(0).Domain(), ix)), dm,
                    1, 0);
                force[s][d] = w.m_fields.alloc_init(
                    s ? "composite_fixed_B" : "composite_fixed_E",
                    ablastr::fields::Direction{d}, 1,
                    amrex::BoxArray(amrex::convert(domain, ix)), dm, 1,
                    IntVect(WarpX::nox + 1), 0._rt, false, false, false);
            }
        }

        for (int k = 0; k < fine[d]->boxArray().size(); ++k) {
            fine_owns[d]->setVal(1, fine[d]->boxArray()[k], 0, 1);
        }
    }
}

void
FixedShapeCurrent::Assemble (WarpX& w, Real dt) {
    auto const box = w.Geom(1).Domain(), cbox = w.Geom(0).Domain();
    auto const dinv = WarpX::InvCellSize(1),
               lower = WarpX::LowerCorner(box, 1, .5_rt * dt);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        dt * PhysConst::c * amrex::max(dinv.x, amrex::max(dinv.y, dinv.z)) <=
            1._rt,
        "The virtual current stencil requires displacement within one fine "
        "cell per step.");
    auto const clo = w.Geom(0).ProbLoArray(),
               cdi = w.Geom(0).InvCellSizeArray();
    auto const safe = valid_centers.const_array(0);
    amrex::ReduceOps<amrex::ReduceOpLogicalOr> reduction;
    amrex::ReduceData<int> outside(reduction);
    using Result = decltype(outside)::Type;
    for_tiles(w, [&] (WarpXParIter const& it, Real) {
        auto const pos = GetParticlePosition<PIdx>(it);
        reduction.eval(
            it.numParticles(), outside, [=] AMREX_GPU_DEVICE(int n) noexcept -> Result {
                amrex::ParticleReal x, y, z;
                pos(n, x, y, z);
                IntVect const p{int(amrex::Math::floor((x - clo[0]) * cdi[0])),
                                int(amrex::Math::floor((y - clo[1]) * cdi[1])),
                                int(amrex::Math::floor((z - clo[2]) * cdi[2]))};
                return {static_cast<int>(!cbox.contains(p) || !safe(p))};
            });
    });
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::get<0>(outside.value()) == 0,
        WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor
            ? "A fixed-shape particle source stencil reached the outer physical boundary."
            : "A fixed-shape particle source stencil reached a material or physical boundary.");
    std::array<amrex::Array4<Real>, 3> j;
    for (int d = 0; d < 3; ++d) {
        raw[d]->setVal(0._rt);
        j[d] = raw[d]->array(0);
    }
    for_tiles(w, [&] (WarpXParIter const& it, Real q) {
        if (WarpX::nox == 3) {
            deposit<3>(it, q, dt, box, dinv, lower, j);
        } else {
            deposit<1>(it, q, dt, box, dinv, lower, j);
        }
    });
    auto const fine = w.m_fields.get_alldirs(FieldType::current_fp, 1);
    auto const coarse = w.m_fields.get_alldirs(FieldType::current_fp, 0);
    auto const cp = w.m_fields.get_alldirs(FieldType::current_cp, 1);
    auto const r = w.refRatio(0);
    Real const density_scale = 1._rt / (r[0] * r[1] * r[2]);
    for (int d = 0; d < 3; ++d) {
        if (filtered[d]) {
            w.bilinear_filter.ApplyStencil(*filtered[d], *raw[d], 1);
        }
        auto const& source = filtered[d] ? *filtered[d] : *raw[d];
        fine[d]->ParallelCopy(source, 0, 0, 1, 0, 0);
        auto const arr = source.const_array(0);
        auto const mask = fine_owns[d]->const_array(0);
        auto const dom = source.boxArray()[0];
        auto const u = (d + 1) % 3, v = (d + 2) % 3;
        for (amrex::MFIter it(*coarse[d]); it.isValid(); ++it) {
            auto const dest = coarse[d]->array(it);
            // Volume adjoint of exterior Whitney prolongation. Never deposit
            // the fine-owned trace again: its weak source enters the surface
            // solve.
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j_, int k) noexcept {
                    IntVect const base = IntVect(i, j_, k) * r;
                    Real sum = 0._rt;
                    for (int a = 0; a < r[d]; ++a) {
                        for (int b = 1 - r[u]; b < r[u]; ++b) {
                            for (int c = 1 - r[v]; c < r[v]; ++c) {
                                IntVect p = base;
                                p[d] += a;
                                p[u] += b;
                                p[v] += c;
                                if (dom.contains(p) && !mask(p)) {
                                    sum += (1._rt - Real(std::abs(b)) / r[u]) *
                                           (1._rt - Real(std::abs(c)) / r[v]) *
                                           arr(p);
                                }
                            }
                        }
                    }
                    dest(i, j_, k) = density_scale * sum;
                });
        }
        cp[d]->setVal(0._rt);
    }
}

void
FixedShapeCurrent::UpdateGather (WarpX& w) {
    if (!force[0][0]) {
        return;
    }
    auto const r = w.refRatio(0);
    for (int s = 0; s < 2; ++s) {
        auto const type = s ? FieldType::Bfield_fp : FieldType::Efield_fp;
        auto const coarse = w.m_fields.get_alldirs(type, 0),
                   fine = w.m_fields.get_alldirs(type, 1);
        for (int d = 0; d < 3; ++d) {
            auto& parent = *coarse_gather[s][d];
            auto& target = *force[s][d];
            parent.ParallelCopy(*coarse[d]);
            auto const src = parent.const_array(0);
            auto const dst = target.array(0);
            auto const nodal = target.ixType().toIntVect();
            // Whitney prolongation outside the refined patch. Fine-owned values
            // take precedence, including surface tangential E and normal B.
            amrex::ParallelFor(
                target.boxArray()[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    IntVect const q{i, j, k}, p = amrex::coarsen(q, r);
                    Real frac[3];
                    int hi[3];
                    for (int a = 0; a < 3; ++a) {
                        auto const rem = nodal[a] * (q[a] - p[a] * r[a]);
                        frac[a] = Real(rem) / r[a];
                        hi[a] = int(rem != 0);
                    }
                    Real sum = 0._rt;
                    for (int z = 0; z <= hi[2]; ++z) {
                        for (int y = 0; y <= hi[1]; ++y) {
                            for (int x = 0; x <= hi[0]; ++x) {
                                sum += (x ? frac[0] : 1._rt - frac[0]) *
                                       (y ? frac[1] : 1._rt - frac[1]) *
                                       (z ? frac[2] : 1._rt - frac[2]) *
                                       src(p + IntVect(x, y, z));
                            }
                        }
                    }
                    dst(q) = sum;
                });
            target.ParallelCopy(*fine[d]);
        }
    }
}

#endif

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
FixedShapeGather
get_fixed_shape_gather (
    WarpX& w, int lev, std::array<amrex::MultiFab const*, 6> const* requested) {
    FixedShapeGather out;
    if (!w.m_fields.has_vector("composite_fixed_E", 1)) {
        return out;
    }
    for (int s = 0; s < 2; ++s) {
        for (int d = 0; d < 3; ++d) {
            auto const dir = ablastr::fields::Direction{d};
            if (requested && (*requested)[3 * s + d] !=
                                 w.m_fields.get(s ? warpx::fields::FieldType::Bfield_aux
                                                  : warpx::fields::FieldType::Efield_aux,
                                                dir, lev)) {
                return {};
            }
            out.fields[3 * s + d] = &(*w.m_fields.get(
                s ? "composite_fixed_B" : "composite_fixed_E", dir, 1))[0];
        }
    }
    return out;
}
#else
FixedShapeGather
get_fixed_shape_gather (WarpX&, int,
                        std::array<amrex::MultiFab const*, 6> const*) {
    return {};
}
#endif

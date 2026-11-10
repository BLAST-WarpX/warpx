/* This file is part of WarpX. License: BSD-3-Clause-LBNL. */
#include "WarpX.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>

#include <algorithm>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
    enum class Trace { nodal, azimuthal_magnetic, axial_magnetic };

    // The regular interface uses one FAB per mesh. Its continuation through
    // the exterior axial PML uses the matching low-z and high-z strips.
    void couple_trace (amrex::MultiFab& fine, amrex::MultiFab& parent, int fine_node, Trace trace)
    {
        int const coarse_node = fine_node / 2;
        bool const nodal = trace == Trace::nodal;
        int const components = fine.nComp();
        amrex::Real const radius = static_cast<amrex::Real>(fine_node); // in units of dr_f
        amrex::Real const inner_radius = radius - 0.5_rt;
        amrex::Real const outer_radius = radius + 1._rt;
        amrex::Real const dual_area =
            0.5_rt * (outer_radius * outer_radius - inner_radius * inner_radius);

        for (int fi : fine.IndexArray()) {
            auto const& fine_box = fine.boxArray()[fi];
            if (fine_box.bigEnd(0) != fine_node - (nodal ? 0 : 1)) { continue; }
            for (int ci : parent.IndexArray()) {
                auto const& parent_box = parent.boxArray()[ci];
                if (parent_box.smallEnd(0) > coarse_node || parent_box.bigEnd(0) < coarse_node ||
                    parent_box.bigEnd(1) < fine_box.smallEnd(1) ||
                    parent_box.smallEnd(1) > fine_box.bigEnd(1)) {
                    continue;
                }
                auto& fine_fab = fine[fi];
                auto& coarse_fab = parent[ci];
                int const lo = std::max(fine_fab.box().smallEnd(1), coarse_fab.box().smallEnd(1));
                int const hi = std::min(fine_fab.box().bigEnd(1), coarse_fab.box().bigEnd(1));
                auto const f = fine_fab.array();
                auto const c = coarse_fab.array();
                amrex::ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE (int n) noexcept {
                    int const j = lo + n;
                    if (nodal) {
                        // The tangential E or normal B trace owns its axial PML memories too.
                        for (int m = 0; m < components; ++m) {
                            c(coarse_node, j, 0, m) = f(fine_node, j, 0, m);
                        }
                    } else {
                        amrex::Real const inner = f(fine_node - 1, j, 0);
                        amrex::Real const outer = c(coarse_node, j, 0);
                        // Give the existing fine-grid Ampere stencil its nonuniform
                        // dual-cell curl using the physical exterior magnetic sample.
                        if (trace == Trace::azimuthal_magnetic) {
                            amrex::Real const curl =
                                (outer_radius * outer - inner_radius * inner) / dual_area;
                            f(fine_node, j, 0) = (radius * curl + inner_radius * inner) / (radius + 0.5_rt);
                        } else {
                            f(fine_node, j, 0) = inner + (outer - inner) / (outer_radius - inner_radius);
                        }
                    }
                });
            }
        }
    }
}

void
WarpX::ShareRadialInterface (bool electric)
{
    int const rf = boxArray(1)[0].bigEnd(0) + 1;
    for (auto type : {electric ? FieldType::Efield_fp : FieldType::Bfield_fp,
                      electric ? FieldType::pml_E_fp : FieldType::pml_B_fp}) {
        if (!m_fields.has_vector(type, 1)) { continue; }
        auto fine = m_fields.get_alldirs(type, 1);
        auto parent = m_fields.get_alldirs(type, 0);
        if (electric) {
            couple_trace(*fine[1], *parent[1], rf, Trace::nodal);
            couple_trace(*fine[2], *parent[2], rf, Trace::nodal);
        } else {
            couple_trace(*fine[0], *parent[0], rf, Trace::nodal);
            couple_trace(*fine[1], *parent[1], rf, Trace::azimuthal_magnetic);
            couple_trace(*fine[2], *parent[2], rf, Trace::axial_magnetic);
        }
    }
}

void
WarpX::CheckRadialInterfaceCollar ()
{
    // Gather and deposition still use a uniform-grid shape within each mesh.
    // A source-free collar keeps every particle stencil on one side of the join.
    int const fine_node = boxArray(1)[0].bigEnd(0) + 1;
    amrex::Real const radius = fine_node * Geom(1).CellSize(0);
    amrex::ReduceOps<amrex::ReduceOpMax> reduction;
    amrex::ReduceData<int> result(reduction);
    using Result = decltype(result)::Type;
    for (int lev = 0; lev <= 1; ++lev) {
        amrex::Real const guard =
            0.5_rt * (nox + 2) * Geom(lev).CellSize(0) + PhysConst::c * dt[0];
        for (int species = 0; species < mypc->nSpecies(); ++species) {
            auto& pc = mypc->GetParticleContainer(species);
            for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
                auto const* radial = pti.GetStructOfArrays().GetRealData(PIdx::r).data();
                reduction.eval(pti.numParticles(), result,
                    [=] AMREX_GPU_DEVICE (int i) noexcept -> Result {
                        return {static_cast<int>(amrex::Math::abs(radial[i] - radius) <= guard)};
                    });
            }
        }
        int const node = lev == 0 ? fine_node / 2 : fine_node;
        for (auto* current : m_fields.get_alldirs(FieldType::current_fp, lev)) {
            amrex::Box collar = current->boxArray()[0];
            collar.setSmall(0, std::max(collar.smallEnd(0), node - 2));
            collar.setBig(0, std::min(collar.bigEnd(0), node + 1));
            auto const jfield = (*current)[0].const_array();
            reduction.eval(collar, result,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Result {
                    return {static_cast<int>(jfield(i, j, k,0) != 0._rt)};
                });
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::get<0>(result.value()) == 0,
        "A particle stencil or a deposited current reached the vacuum collar "
        "of the radial interface.");
}

void
WarpX::FillRadialInterfaceGatherGuards ()
{
    int const rf = boxArray(1)[0].bigEnd(0) + 1;
    int const rc = rf / 2;
    for (auto type : {FieldType::Efield_aux, FieldType::Bfield_aux}) {
        auto const parent = m_fields.get_alldirs(type, 0);
        auto const fine = m_fields.get_alldirs(type, 1);
        for (int d = 0; d < 3; ++d) {
            bool const nodal = fine[d]->ixType().nodeCentered(0);
            auto& ff = (*fine[d])[0];
            auto const& cf = (*parent[d])[0];
            amrex::Box band = ff.box();
            band.setSmall(0, rf + (nodal ? 1 : 0));
            band.setSmall(1, std::max(band.smallEnd(1), cf.box().smallEnd(1)));
            band.setBig(1, std::min(band.bigEnd(1), cf.box().bigEnd(1)));
            if (!band.ok()) { continue; }
            auto f = ff.array();
            auto c = cf.const_array();
            amrex::ParallelFor(band, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (!nodal && i == rf) {
                    // This guard lies between the real fine and parent cell centers.
                    f(i, j, k) = (f(rf - 1, j, k) + 2._rt * c(rc, j, k)) / 3._rt;
                } else {
                    amrex::Real const x = nodal ? 0.5_rt * i : 0.5_rt * i - 0.25_rt;
                    int const first = static_cast<int>(x);
                    amrex::Real const frac = x - first;
                    f(i, j, k) = (1._rt - frac) * c(first, j, k) + frac * c(first + 1, j, k);
                }
            });
        }
    }
}

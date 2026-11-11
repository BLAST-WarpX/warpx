/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#ifdef WARPX_DIM_XZ
#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include <AMReX_MultiFab.H>

#include <cmath>
#include <typeinfo>

using namespace amrex::literals;
using ablastr::fields::Direction;
using warpx::fields::FieldType;

namespace {
    constexpr char reference_current[] = "xz_fixed_raw_current";
    constexpr char filtered_current[] = "xz_fixed_filtered_current";

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int periodic_index (int i, int n)
    {
        return i < 0 ? i + n : (i >= n ? i - n : i);
    }

    template <int order>
    void deposit_tile (WarpXParIter const& it, amrex::Real q, amrex::Real dt,
                       amrex::Box const& box, amrex::XDim3 dinv, amrex::XDim3 lower,
                       amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& j)
    {
        auto const pos = GetParticlePosition<PIdx>(it);
        auto const* w = it.GetAttribs(PIdx::w).dataPtr();
        auto const* ux = it.GetAttribs(PIdx::ux).dataPtr();
        auto const* uy = it.GetAttribs(PIdx::uy).dataPtr();
        auto const* uz = it.GetAttribs(PIdx::uz).dataPtr();
        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov) {
            doEsirkepovDepositionShapeN<order>(
                pos, w, ux, uy, uz, nullptr, j[0], j[1], j[2], it.numParticles(),
                dt, -0.5_rt*dt, dinv, lower, amrex::lbound(box), q, 1,
                amrex::Array4<int const>{}, false);
        } else {
            amrex::GpuArray<amrex::GpuArray<double, 2>, AMREX_SPACEDIM> const bounds{};
            amrex::GpuArray<amrex::GpuArray<bool, 2>, AMREX_SPACEDIM> const crop{};
            doVillasenorDepositionShapeNExplicit<order>(
                pos, w, ux, uy, uz, nullptr, j[0], j[1], j[2], it.numParticles(),
                dt, -0.5_rt*dt, dinv, lower, bounds, crop, amrex::lbound(box), q, 1);
        }
    }
}

void WarpX::InitCompositeXZParticleCoupling ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mypc->nSpecies() == mypc->nContainers() && nox >= 1 && nox <= 3 &&
        (current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
         current_deposition_algo == CurrentDepositionAlgo::Villasenor),
        "Fixed-shape XZ coupling requires ordinary kinetic species and charge-conserving deposition.");
    for (int s = 0; s < mypc->nSpecies(); ++s) {
        auto const& pc = mypc->GetParticleContainer(s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            typeid(pc) == typeid(PhysicalParticleContainer),
            "Fixed-shape XZ coupling requires ordinary particles with fixed species charge.");
    }
    auto const domain = Geom(1).Domain();
    auto const pass = use_filter ? filter_npass_each_dir : amrex::IntVect(0);
    auto const ng = amrex::IntVect(nox + 1) + pass;
    for (int d = 0; d < 3; ++d) {
        auto const dir = Direction{d};
        auto const* j = m_fields.get(FieldType::current_fp, dir, 1);
        auto const current_boxes = amrex::BoxArray(amrex::convert(domain, j->ixType()));
        m_fields.alloc_init(reference_current, dir, 1, current_boxes, j->DistributionMap(),
                            1, ng, 0._rt, false, false, false);
        if (use_filter) {
            m_fields.alloc_init(filtered_current, dir, 1, current_boxes, j->DistributionMap(),
                                1, ng, 0._rt, false, false, false);
        }
        for (int family = 0; family < 2; ++family) {
            auto const type = family ? FieldType::Bfield_fp : FieldType::Efield_fp;
            auto const* f = m_fields.get(type, dir, 1);
            m_fields.alloc_init(family ? "composite_fixed_B" : "composite_fixed_E", dir,
                1, amrex::BoxArray(amrex::convert(domain, f->ixType())), f->DistributionMap(),
                1, amrex::IntVect(nox + 1), 0._rt, false, false, false);
        }
    }
}

void WarpX::AssembleCompositeXZCurrent (amrex::Real a_dt)
{
    auto const raw = m_fields.get_alldirs(reference_current, 1);
    auto const box = Geom(1).Domain();
    auto const dinv = InvCellSize(1);
    // The shape kernels use local cell coordinates. Include the guard cells so
    // a step through a low periodic face still has nonnegative local indices.
    auto const deposit_box = amrex::grow(box, raw[0]->nGrowVect());
    auto const lower = LowerCorner(deposit_box, 1, 0._rt);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(PhysConst::c*a_dt <= amrex::min(Geom(1).CellSize(0), Geom(1).CellSize(1)),
        "A fixed-shape XZ trajectory must move at most one finest cell per step.");
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> current;
    for (int d = 0; d < 3; ++d) {
        raw[d]->setVal(0._rt);
        current[d] = raw[d]->array(0);
    }
    for (int s = 0; s < mypc->nSpecies(); ++s) {
        auto& pc = mypc->GetParticleContainer(s);
        if (!pc.isPushed() || pc.do_not_deposit || pc.getCharge() == 0._rt) { continue; }
        for (int lev = 0; lev <= 1; ++lev) {
            for (WarpXParIter it(pc, lev); it.isValid(); ++it) {
                switch (nox) {
                case 1: deposit_tile<1>(it, pc.getCharge(), a_dt, deposit_box, dinv, lower, current); break;
                case 2: deposit_tile<2>(it, pc.getCharge(), a_dt, deposit_box, dinv, lower, current); break;
                case 3: deposit_tile<3>(it, pc.getCharge(), a_dt, deposit_box, dinv, lower, current); break;
                default: WARPX_ABORT_WITH_MESSAGE("Unknown fixed-shape XZ particle order.");
                }
            }
        }
    }
    auto const ratio = refRatio(0);
    int const rx = ratio[0], rz = ratio[1];
    int const nx = box.length(0), nz = box.length(1);
    bool const periodic_x = Geom(1).isPeriodic(0);
    for (int d = 0; d < 3; ++d) {
        auto const dir = Direction{d};
        auto* source = raw[d];
        source->SumBoundary(Geom(1).periodicity());
        source->FillBoundary(Geom(1).periodicity());
        if (use_filter) {
            auto* output = m_fields.get(filtered_current, dir, 1);
            bilinear_filter.ApplyStencil(*output, *source, 1);
            source = output;
            source->FillBoundary(Geom(1).periodicity());
        }
        auto* fine = m_fields.get(FieldType::current_fp, dir, 1);
        auto* coarse = m_fields.get(FieldType::current_fp, dir, 0);
        // Native sources (including afterdeposition callbacks) have already
        // been synchronized. Add only the kinetic contribution to each leaf.
        fine->ParallelAdd(*source);
        auto const jf = source->const_array(0);
        auto const jc = coarse->array(0);
        auto const ix = coarse->ixType().toIntVect();
        int const fl = fine->boxArray()[0].smallEnd(0);
        int const fr = fine->boxArray()[0].bigEnd(0);
        // Volume adjoint of the edge interpolation. The fine trace carries
        // its own source in the joined Ampere equation.
        amrex::ParallelFor(coarse->boxArray()[0],
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
                amrex::Real sum = 0._rt;
                for (int a = ix[0] ? 1-rx : 0; a < rx; ++a) {
                    int x = rx*i + a;
                    if (periodic_x) { x = periodic_index(x, nx); }
                    if (x < 0 || x >= nx+ix[0] || (x >= fl && x <= fr)) { continue; }
                    auto const wx = ix[0] ? 1._rt-amrex::Real(std::abs(a))/rx : 1._rt;
                    for (int b = ix[1] ? 1-rz : 0; b < rz; ++b) {
                        int const z = periodic_index(rz*j + b, nz);
                        auto const wz = ix[1] ? 1._rt-amrex::Real(std::abs(b))/rz : 1._rt;
                        sum += wx*wz*jf(x,z,0);
                    }
                }
                jc(i,j,0) += sum / amrex::Real(rx*rz);
            });
        coarse->FillBoundary(Geom(0).periodicity());
        fine->FillBoundary(Geom(1).periodicity());
    }
}

void WarpX::UpdateCompositeXZGather ()
{
    auto const ratio = refRatio(0);
    for (int family = 0; family < 2; ++family) {
        auto const type = family ? FieldType::Bfield_fp : FieldType::Efield_fp;
        for (int d = 0; d < 3; ++d) {
            auto const dir = Direction{d};
            auto* target = m_fields.get(family ? "composite_fixed_B" : "composite_fixed_E", dir, 1);
            auto const* fine = m_fields.get(type, dir, 1);
            auto const pc = m_fields.get(type, dir, 0)->const_array(0);
            auto const dst = target->array(0);
            auto const ix = target->ixType().toIntVect();
            amrex::ParallelFor(target->boxArray()[0],
                [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
                    int const x = i/ratio[0], z = j/ratio[1];
                    auto const fx = amrex::Real(ix[0]*(i%ratio[0]))/ratio[0];
                    auto const fz = amrex::Real(ix[1]*(j%ratio[1]))/ratio[1];
                    amrex::Real sum = 0._rt;
                    for (int a = 0; a <= int(fx != 0._rt); ++a) {
                        for (int b = 0; b <= int(fz != 0._rt); ++b) {
                            sum += (a ? fx : 1._rt-fx)*(b ? fz : 1._rt-fz)*pc(x+a,z+b,0);
                        }
                    }
                    dst(i,j,0) = sum;
                });
            target->ParallelCopy(*fine);
            target->FillBoundary(Geom(1).periodicity());
        }
    }
}
#endif

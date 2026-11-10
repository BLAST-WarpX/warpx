/* This file is part of WarpX. License: BSD-3-Clause-LBNL. */
#include "WarpX.H"

#ifdef WARPX_DIM_RZ
#include "Fields.H"
#include "Utils/TextMsg.H"

#include <AMReX_MultiFab.H>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    // Exactly one regular FAB per level is required. An exterior axial PML
    // additionally has one or two matching strips at the radius of the join.
    template <typename F>
    void visit_interface (MultiFab& fine, MultiFab& parent, int fine_node, F const& fn)
    {
        int const coarse_node = fine_node / 2;
        bool const nodal = fine.ixType().nodeCentered(0);
        for (int fi : fine.IndexArray()) {
            Box const& valid = fine.boxArray()[fi];
            if (valid.bigEnd(0) != fine_node - (nodal ? 0 : 1)) { continue; }
            for (int ci : parent.IndexArray()) {
                Box const& pv = parent.boxArray()[ci];
                if (pv.smallEnd(0) > coarse_node || pv.bigEnd(0) < coarse_node ||
                    pv.bigEnd(1) < valid.smallEnd(1) || pv.smallEnd(1) > valid.bigEnd(1)) {
                    continue;
                }
                auto& ff = fine[fi];
                auto& cf = parent[ci];
                int const low = std::max(ff.box().smallEnd(1), cf.box().smallEnd(1));
                int const high = std::min(ff.box().bigEnd(1), cf.box().bigEnd(1));
                fn(ff.array(), cf.array(), low, high, fine_node, coarse_node, fine.nComp());
            }
        }
    }

    void share_node (MultiFab& fine, MultiFab& parent, int fine_node)
    {
        visit_interface(fine, parent, fine_node,
            [] (auto f, auto c, int lo, int hi, int rf, int rc, int ncomp) {
                amrex::ParallelFor(hi-lo+1,
                    [=] AMREX_GPU_DEVICE (int n) noexcept {
                        int const j = lo + n;
                        for (int m = 0; m < ncomp; ++m) { c(rc,j,0,m) = f(rf,j,0,m); }
                    });
            });
    }

    void give_fine_outer_magnetic_sample (MultiFab& fine, MultiFab& parent,
                                           int fine_node, bool azimuthal)
    {
        visit_interface(fine, parent, fine_node,
            [azimuthal] (auto f, auto c, int lo, int hi, int rf, int rc, int) {
                // Dimensionless radii in units of dr_f. The dual cell faces
                // pass through the actual fine and parent magnetic samples.
                Real const R = static_cast<Real>(rf);
                Real const rm = R - 0.5_rt, rp = R + 1._rt;
                amrex::ParallelFor(hi-lo+1,
                    [=] AMREX_GPU_DEVICE (int n) noexcept {
                        int const j = lo + n;
                        Real const inner = f(rf-1,j,0,0);
                        Real const outer = c(rc,j,0,0);
                        if (azimuthal) {
                            Real const curl = (rp*outer - rm*inner) / (0.5_rt*(rp*rp-rm*rm));
                            f(rf,j,0,0) = (R*curl + rm*inner) / (R+0.5_rt);
                        } else {
                            f(rf,j,0,0) = inner + (outer-inner)/(rp-rm);
                        }
                    });
            });
    }
}

void WarpX::ShareRadialInterface (bool electric)
{
    int const rf = boxArray(1)[0].bigEnd(0) + 1;
    for (auto type : {electric ? FieldType::Efield_fp : FieldType::Bfield_fp,
                      electric ? FieldType::pml_E_fp : FieldType::pml_B_fp}) {
        if (!m_fields.has_vector(type, 1)) { continue; }
        auto fine = m_fields.get_alldirs(type, 1);
        auto parent = m_fields.get_alldirs(type, 0);
        if (electric) {
            share_node(*fine[1], *parent[1], rf);
            share_node(*fine[2], *parent[2], rf);
        } else {
            share_node(*fine[0], *parent[0], rf);
            give_fine_outer_magnetic_sample(*fine[1], *parent[1], rf, true);
            give_fine_outer_magnetic_sample(*fine[2], *parent[2], rf, false);
        }
    }
}

void WarpX::CheckRadialInterfaceCurrent ()
{
    // Level-local particle assignment has no cross-mesh charge-conserving
    // shape at this join. Reject a deposited source in its vacuum collar.
    int const rf = boxArray(1)[0].bigEnd(0) + 1;
    for (int lev = 0; lev <= 1; ++lev) {
        int const node = lev == 0 ? rf / 2 : rf;
        for (auto* current : m_fields.get_alldirs(FieldType::current_fp, lev)) {
            Box collar = current->boxArray()[0];
            collar.setSmall(0, std::max(collar.smallEnd(0),node-2));
            collar.setBig(0, std::min(collar.bigEnd(0),node+1));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (*current)[0].norm<RunOn::Device>(collar, 0, 0, 1) == 0._rt,
                "A source reached the vacuum collar of the conforming radial interface.");
        }
    }
}
#endif

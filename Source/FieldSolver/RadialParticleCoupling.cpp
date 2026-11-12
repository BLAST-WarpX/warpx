/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadialParticleCoupling.H"

#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <typeinfo>

using namespace amrex::literals;
using ablastr::fields::Direction;
using amrex::Real;
using warpx::fields::FieldType;

namespace warpx::radial_particles {
namespace {
constexpr char const* raw_name = "radial_fixed_J";
constexpr char const* identity_file = "RadialFixedShape";

std::string
identity () {
    std::ostringstream record;
    record << "Axisymmetric fixed-shape Esirkepov 1 " << WarpX::nox << ' '
           << WarpX::use_filter << ' ' << WarpX::filter_npass_each_dir << '\n';
    return record.str();
}

// A guarded checkpoint with kinetic particles confined to the fine mesh
// already uses the same physical particle shapes, current and force there.
// Keep this compatibility path one-way and check the loaded particles.
void
check_guarded_restart (WarpX& w) {
    Real const dr = w.Geom(1).CellSize(0);
    Real const edge = (w.boxArray(1)[0].bigEnd(0) + 1) * dr;
    Real const limit =
        edge - 0.5_rt * (WarpX::nox + 3) * dr - PhysConst::c * w.getdt(1);
    amrex::ReduceOps<amrex::ReduceOpMax> op;
    amrex::ReduceData<int> data(op);
    using Result = decltype(data)::Type;
    op.eval(1, data,
            [] AMREX_GPU_DEVICE(int) noexcept -> Result { return {0}; });
    auto& pcs = w.GetPartContainer();
    for (int s = 0; s < pcs.nContainers(); ++s) {
        auto& pc = pcs.GetParticleContainer(s);
        for (int lev = 0; lev <= 1; ++lev) {
            for (WarpXParIter it(pc, lev); it.isValid(); ++it) {
                auto const* r = it.GetAttribs(PIdx::r).dataPtr();
                op.eval(it.numParticles(), data,
                        [=] AMREX_GPU_DEVICE(int i) noexcept -> Result {
                            return {static_cast<int>(
                                lev == 0 || !std::isfinite(r[i]) ||
                                r[i] < 0._rt || r[i] >= limit)};
                        });
            }
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::get<0>(data.value()) == 0,
        "A guarded radial checkpoint can enable fixed-shape coupling only with "
        "every trajectory stencil still inside the fine bore.");
}

template <int order>
void
deposit_tile (WarpXParIter const& it, Real charge, Real dt,
              amrex::Box const& box, amrex::XDim3 const& dinv,
              amrex::XDim3 const& lower,
              ablastr::fields::VectorField const& current) {
    doEsirkepovDepositionShapeN<order>(
        GetParticlePosition<PIdx>(it), it.GetAttribs(PIdx::w).dataPtr(),
        it.GetAttribs(PIdx::ux).dataPtr(), it.GetAttribs(PIdx::uy).dataPtr(),
        it.GetAttribs(PIdx::uz).dataPtr(), nullptr, current[0]->array(0),
        current[1]->array(0), current[2]->array(0), it.numParticles(), dt,
        -0.5_rt * dt, dinv, lower, amrex::lbound(box), charge, 1,
        amrex::Array4<int const>{}, false);
}
} // namespace

bool
enabled (ablastr::fields::MultiFabRegister const& fields) {
    return fields.has_vector(raw_name, 1);
}

void
initialize (WarpX& w, std::string const& restart) {
    bool requested = false;
    amrex::ParmParse("warpx").query("conforming_radial_fixed_shape", requested);
    bool from_guarded = false;
    if (!restart.empty()) {
        auto const path = std::filesystem::path(restart) / identity_file;
        bool const stored = std::filesystem::exists(path);
        from_guarded = requested && !stored;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !stored || requested,
            "The checkpoint and requested radial particle coupling differ.");
        if (stored) {
            std::ifstream const file(path);
            std::ostringstream bytes;
            bytes << file.rdbuf();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                bytes.str() == identity(), "The checkpoint uses a different "
                                           "radial particle shape or filter.");
        }
    }
    if (!requested) {
        return;
    }
    auto& populations = w.GetPartContainer();
    bool const kinetic = populations.nSpecies() != 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        w.usesConformingRadialInterface() &&
            w.field_gathering_algo == GatheringAlgo::EnergyConserving &&
            (!kinetic || (WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov &&
                          WarpX::nox >= 1 && WarpX::nox <= 3)) && !w.do_current_centering &&
            (!WarpX::use_filter || WarpX::filter_npass_each_dir[0] == 0) &&
            !w.DoFluidSpecies() && w.GetPartContainer().nLasers() == 0,
        "Radial fixed-shape coupling requires a joined (2,1) Yee bore, "
        "energy-conserving "
        "gather, linear through cubic Esirkepov deposition and at most axial "
        "filtering.");
    bool const antenna_trajectories = populations.nContainers() > populations.nSpecies() &&
        WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov &&
        WarpX::nox >= 1 && WarpX::nox <= 3;
    if (!kinetic && !antenna_trajectories) { return; }
    for (int s = populations.nSpecies(); s < populations.nContainers(); ++s) {
        auto& pc = populations.GetParticleContainer(s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc.DeferTrajectoryCurrent(),
            "Radial fixed-shape coupling requires an explicit fixed-charge trajectory for this source.");
    }
    if (from_guarded) {
        check_guarded_restart(w);
    }
    for (int s = 0; s < populations.nSpecies(); ++s) {
        auto& pc = populations.GetParticleContainer(s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            typeid(pc) == typeid(PhysicalParticleContainer) &&
                !pc.DoFieldIonization() && !pc.doContinuousInjection() &&
                !pc.do_splitting,
            "Radial fixed-shape current requires fixed-charge kinetic "
            "species.");
    }
    auto const box = w.Geom(1).Domain();
    auto const fine = w.m_fields.get_alldirs(FieldType::current_fp, 1);
    for (int d = 0; d < 3; ++d) {
        auto const dir = Direction{d};
        w.m_fields.alloc_init(
            raw_name, dir, 1,
            amrex::BoxArray(amrex::convert(box, fine[d]->ixType())),
            fine[d]->DistributionMap(), 1, fine[d]->nGrowVect(), 0._rt, false,
            false, false);
        for (auto kind : {FieldType::Efield_aux, FieldType::Bfield_aux}) {
            auto const* f = w.m_fields.get(kind, dir, 1);
            auto const* c = w.m_fields.get(kind, dir, 0);
            auto const ng = amrex::min(f->nGrowVect(), c->nGrowVect());
            w.m_fields.alloc_init(
                kind == FieldType::Efield_aux ? "composite_fixed_E"
                                              : "composite_fixed_B",
                dir, 1, amrex::BoxArray(amrex::convert(box, f->ixType())),
                f->DistributionMap(), 1, ng, 0._rt, false, false, false);
        }
    }
}

void
write_identity (WarpX& w, std::string const& directory) {
    bool requested = false;
    amrex::ParmParse("warpx").query("conforming_radial_fixed_shape", requested);
    if (!requested || !w.usesConformingRadialInterface()) { return; }
    std::ofstream file(std::filesystem::path(directory) / identity_file);
    file << identity();
    file.close();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        file.good(), "Could not save the radial particle coupling.");
}

void
update_gather (WarpX& w) {
    if (!enabled(w.m_fields)) {
        return;
    }
    for (auto kind : {FieldType::Efield_aux, FieldType::Bfield_aux}) {
        auto const parent = w.m_fields.get_alldirs(kind, 0);
        auto const fine = w.m_fields.get_alldirs(kind, 1);
        auto const target = w.m_fields.get_alldirs(kind == FieldType::Efield_aux
                                                       ? "composite_fixed_E"
                                                       : "composite_fixed_B",
                                                   1);
        for (int d = 0; d < 3; ++d) {
            bool const nodal = fine[d]->ixType().nodeCentered(0);
            int const last = fine[d]->boxArray()[0].bigEnd(0);
            auto const f = fine[d]->const_array(0);
            auto const c = parent[d]->const_array(0);
            auto& fab = (*target[d])[0];
            auto const g = fab.array();
            // Inside the fine leaf, retain the native axis and axial guards. In
            // the exterior radial edges carry a constant field, nodes
            // interpolate linearly.
            amrex::ParallelFor(fab.box(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int) noexcept {
                if (i <= last) {
                    g(i, j, 0) = f(i, j, 0);
                } else {
                    int const p = i / 2;
                    g(i, j, 0) = nodal && i % 2
                                     ? 0.5_rt * (c(p, j, 0) + c(p + 1, j, 0))
                                     : c(p, j, 0);
                }
            });
        }
    }
}

ablastr::fields::MultiLevelVectorField
deposit (WarpX& w, Real dt) {
    auto const dinv = WarpX::InvCellSize(1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        dt * PhysConst::c * amrex::max(dinv.x, dinv.z) <= 1._rt,
        "The radial fixed-shape trajectory must move at most one fine cell per "
        "step.");
    ablastr::fields::MultiLevelVectorField levels(2);
    auto& current = levels[1];
    current = w.m_fields.get_alldirs(raw_name, 1);
    // Keep the cylindrical axis at r = 0 and include the axial guards so
    // deposition through the low axial boundary uses positive local indices.
    auto box = w.Geom(1).Domain();
    box.grow(1, current[0]->nGrowVect()[1]);
    auto const lower = WarpX::LowerCorner(box, 1, 0._rt);
    for (auto* field : current) {
        field->setVal(0._rt);
    }
    auto& pcs = w.GetPartContainer();
    for (int s = 0; s < pcs.nContainers(); ++s) {
        auto& pc = pcs.GetParticleContainer(s);
        if (!pc.isPushed() || pc.do_not_deposit || pc.getCharge() == 0._rt) {
            continue;
        }
        for (int lev = 0; lev <= 1; ++lev) {
            for (WarpXParIter it(pc, lev); it.isValid(); ++it) {
                switch (WarpX::nox) {
                case 1:
                    deposit_tile<1>(it, pc.getCharge(), dt, box, dinv, lower,
                                    current);
                    break;
                case 2:
                    deposit_tile<2>(it, pc.getCharge(), dt, box, dinv, lower,
                                    current);
                    break;
                case 3:
                    deposit_tile<3>(it, pc.getCharge(), dt, box, dinv, lower,
                                    current);
                    break;
                default:
                    WARPX_ABORT_WITH_MESSAGE(
                        "Unknown RZ fixed-shape deposition order.");
                }
            }
        }
    }
    return levels;
}

void
add_current (WarpX& w, ablastr::fields::VectorField const& current) {
    int const seam = w.boxArray(1)[0].bigEnd(0) + 1;
    int const outer = w.Geom(1).Domain().bigEnd(0) + 1;
    auto const fine = w.m_fields.get_alldirs(FieldType::current_fp, 1);
    auto const parent = w.m_fields.get_alldirs(FieldType::current_fp, 0);
    for (int d = 0; d < 3; ++d) {
        auto const raw = current[d]->const_array(0);
        bool const radial = d == 0;
        // Use radii in units of dr_f. Ez lives on an annular axial face,
        // Etheta on the ring times its radial dual width. A nodal virtual
        // source immediately outside the join loads each neighbor by half.
        Real const R = seam;
        Real const mass = d == 2 ? 0.5_rt * ((R + 1._rt) * (R + 1._rt) -
                                             (R - .5_rt) * (R - .5_rt))
                                 : 1.5_rt * R;
        auto const f = fine[d]->array(0);
        amrex::ParallelFor(fine[d]->boxArray()[0],
                           [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                               if (i == seam && !radial) {
                                   f(i, j, 0) += (R * raw(i, j, 0) +
                                                  0.5_rt * (R + 1._rt) *
                                                      raw(i + 1, j, 0)) /
                                                 mass;
                               } else {
                                   f(i, j, 0) += raw(i, j, 0);
                               }
                           });
        auto const c = parent[d]->array(0);
        amrex::ParallelFor(
            parent[d]->boxArray()[0],
            [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                int const a = 2 * i;
                if (a < seam) {
                    return;
                }
                if (radial) {
                    // The coarse radial flux is the mean of its two fine-face
                    // fluxes.
                    c(i, j, 0) += ((a + 0.5_rt) * raw(a, j, 0) +
                                   (a + 1.5_rt) * raw(a + 1, j, 0)) /
                                  (2._rt * (a + 1._rt));
                } else if (a == seam) {
                    c(i, j, 0) += (R * raw(a, j, 0) +
                                   0.5_rt * (R + 1._rt) * raw(a + 1, j, 0)) /
                                  mass;
                } else {
                    Real const right =
                        a < outer ? 0.5_rt * (a + 1._rt) * raw(a + 1, j, 0)
                                  : 0._rt;
                    c(i, j, 0) += (0.5_rt * (a - 1._rt) * raw(a - 1, j, 0) +
                                   a * raw(a, j, 0) + right) /
                                  (2._rt * a);
                }
            });
    }
}
} // namespace warpx::radial_particles

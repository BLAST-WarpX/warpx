/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/PacketDiffusionInterfacePhotons.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_BoxList.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    void Require (bool valid, std::string const& message)
    {
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid) {
            throw std::runtime_error(message);
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        // Reverse spatial box order so the owner of the periodic face need
        // not be its lower-index representation. Rank2 is empty in the3-rank run.
        amrex::Box const domain(amrex::IntVect(8), amrex::IntVect(15));
        amrex::BoxList boxes_list;
        boxes_list.push_back(amrex::Box(amrex::IntVect(12), amrex::IntVect(15)));
        boxes_list.push_back(amrex::Box(amrex::IntVect(8), amrex::IntVect(11)));
        amrex::BoxArray const boxes(boxes_list);
        amrex::Vector<int> ranks{0, amrex::ParallelDescriptor::NProcs() > 1 ? 1 : 0};
        amrex::DistributionMapping const distribution(ranks);
        amrex::RealBox const bounds({0.0}, {1.0});
        int const periodic[] = {1};
        amrex::Geometry const geometry(domain, &bounds, 0, periodic);
        amrex::iMultiFab mask(boxes, distribution, 1, 1);
        for (amrex::MFIter iterator(mask); iterator.isValid(); ++iterator) {
            auto const values = mask.array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                values(i, j, k) = i < 12 ? 1 : 0;
            });
        }
        mask.FillBoundary(geometry.periodicity());

        auto const face_boxes = amrex::convert(boxes, amrex::IntVect::TheUnitVector());
        amrex::MultiFab leakage(face_boxes, distribution, 1, 0);
        using warpx::radiation::BuildPacketDiffusionLeakageEmissionPlan;
        using warpx::radiation::detail::EmissionPlan;
        for (int const selected_face : {8, 12}) {
            double const expected_energy = selected_face == 8 ? 2.0 : 5.0;
            double const normal = selected_face == 8 ? -1.0 : 1.0;
            leakage.setVal(0.0);
            for (amrex::MFIter iterator(leakage); iterator.isValid(); ++iterator) {
                auto const values = leakage.array(iterator);
                amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE
                    (int i, int j, int k) {
                        bool const is_face = i == selected_face ||
                                             (selected_face == 8 && i == 16);
                        values(i, j, k) = is_face ? expected_energy : 0.0;
                    });
            }
            leakage.OverrideSync(geometry.periodicity());
            leakage.FillBoundary(geometry.periodicity());
            EmissionPlan plan;
            Require(BuildPacketDiffusionLeakageEmissionPlan(
                        leakage, mask, geometry, 1.e-15, plan),
                    "native leakage rejected a valid periodic owned face");
            std::array<amrex::Real, 6> moments{};
            bool valid_positions = true;
            long ray_count = static_cast<long>(plan.weight.size());
            for (amrex::Long ray = 0; ray < plan.weight.size(); ++ray) {
                long double const ux = plan.ux[ray];
                long double const uy = plan.uy[ray];
                long double const uz = plan.uz[ray];
                long double const magnitude = std::sqrt(ux * ux + uy * uy + uz * uz);
                long double const energy = plan.weight[ray] * PhysConst::m_e *
                                           PhysConst::c * magnitude;
                long double const mu = normal * uz / magnitude;
                moments[0] += static_cast<amrex::Real>(energy);
                moments[1] += static_cast<amrex::Real>(energy * ux / magnitude);
                moments[2] += static_cast<amrex::Real>(energy * uy / magnitude);
                moments[3] += static_cast<amrex::Real>(energy * uz / magnitude);
                moments[4] += static_cast<amrex::Real>(energy * mu);
                moments[5] += static_cast<amrex::Real>(energy * mu * mu);
                double const z = plan.z[ray];
                int const cell = 8 + static_cast<int>(std::floor(z * 8.0));
                valid_positions = valid_positions && std::isfinite(z) && z >= 0.0 &&
                                  z < 1.0 && cell >= 12 && cell <= 15 &&
                                  energy > 0.0 && mu > 0.0 && mu < 1.0;
            }
            Require(valid_positions, "leakage birth is outside the kinetic side");
            amrex::ParallelDescriptor::ReduceLongSum(ray_count);
            amrex::ParallelDescriptor::ReduceRealSum(moments.data(), 6);
            Require(ray_count == 8, "periodic/shared face lost or duplicated leakage rays");
            std::array<double, 6> const expected{
                expected_energy, 0.0, 0.0, normal * expected_energy * 2.0 / 3.0,
                expected_energy * 2.0 / 3.0, expected_energy / 2.0};
            for (int component = 0; component < 6; ++component) {
                Require(std::isfinite(moments[component]) &&
                            std::abs(moments[component] - expected[component]) <=
                                2.e-13 * expected_energy,
                        "stored leakage rays violate energy or angular moments");
            }
            for (int component = 0; component < 4; ++component) {
                Require(std::abs(plan.inventory[component] - moments[component]) <=
                            2.e-13 * expected_energy,
                        "plan inventory differs from stored leakage four-vectors");
            }
        }
        leakage.setVal(0.0);
        EmissionPlan empty;
        Require(BuildPacketDiffusionLeakageEmissionPlan(leakage, mask, geometry, 1.e-15, empty),
                "zero leakage rejected");
        Require(empty.weight.empty(), "zero leakage created rays");
        amrex::Print() << "Native leakage faces: nonzero indices, reversed ownership, "
                          "periodic placement and stored angular moments pass.\n";
    } catch (std::exception const& error) {
        amrex::Print() << error.what() << '\n';
        status = 1;
    }
    amrex::Finalize();
    return status;
}

/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/PacketDiffusionInterfacePhotons.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_BoxList.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace
{
void Require (bool valid, char const* message)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { throw std::runtime_error(message); }
}
int Physical (int d)
{
    return AMREX_SPACEDIM == 1 ? 2 : (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
}
void CheckRedistributedBirths (
    warpx::radiation::detail::EmissionPlan const& plan, amrex::Geometry const& geometry,
    amrex::BoxArray const& boxes, amrex::DistributionMapping const& distribution,
    int axis, int expected_cell, long expected_count)
{
    // Use WarpX's native pure-SoA particle base and the actual CPU/GPU
    // Redistribute path. Checking just the emission plan misses CUDA's
    // unconditional enforcement of the roundoff-safe periodic interval.
    WarpXParticleContainer::Base staged(geometry,distribution,boxes);
    staged.SetArena(amrex::The_Arena());
    if (!plan.weight.empty()) {
        int grid = -1;
        for (int g = 0; g < boxes.size(); ++g) {
            if (distribution[g] == amrex::ParallelDescriptor::MyProc()) { grid = g; break; }
        }
        AMREX_ALWAYS_ASSERT(grid >= 0);
        auto& tile = staged.DefineAndReturnParticleTile(0,grid,0);
        tile.resize(plan.weight.size());
        auto& soa = tile.GetStructOfArrays();
        auto copy = [&](int component, auto const& values) {
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,values.begin(),values.end(),
                            soa.GetRealData(component).begin());
        };
#if defined(WARPX_DIM_3D)
        copy(PIdx::x,plan.x); copy(PIdx::y,plan.y);
#elif defined(WARPX_DIM_XZ)
        copy(PIdx::x,plan.x);
#endif
        copy(PIdx::z,plan.z); copy(PIdx::w,plan.weight);
        copy(PIdx::ux,plan.ux); copy(PIdx::uy,plan.uy); copy(PIdx::uz,plan.uz);
        amrex::Vector<std::uint64_t> identities(plan.weight.size());
        for (std::size_t p = 0; p < identities.size(); ++p) {
            identities[p] = amrex::SetParticleIDandCPU(static_cast<amrex::Long>(p)+1,
                                                       amrex::ParallelDescriptor::MyProc());
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,identities.begin(),identities.end(),
                        soa.GetIdCPUData().begin());
    }
    staged.Redistribute();
    bool valid = true;
    long count = 0;
    for (auto const& [key,tile] : staged.GetParticles(0)) {
        amrex::ignore_unused(key);
        auto const& positions = tile.GetStructOfArrays().GetRealData(axis);
        amrex::Vector<amrex::ParticleReal> host(positions.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,positions.begin(),positions.end(),host.begin());
        count += static_cast<long>(host.size());
        for (auto coordinate : host) {
            int const cell = geometry.Domain().smallEnd(axis)+static_cast<int>(std::floor(
                (coordinate-geometry.ProbLo(axis))*geometry.InvCellSize(axis)));
            valid = valid && cell == expected_cell;
        }
    }
    Require(valid,"native redistribution moved a Cartesian leakage birth across its K-side face");
    amrex::ParallelDescriptor::ReduceLongSum(count);
    Require(count == expected_count,"native redistribution lost Cartesian leakage births");
}
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        using warpx::radiation::BuildPacketDiffusionLeakageEmissionPlan;
        using warpx::radiation::detail::EmissionPlan;
        amrex::Box const domain(amrex::IntVect(8), amrex::IntVect(15));
        amrex::RealBox const bounds({AMREX_D_DECL(0.,0.,0.)},
                                    {AMREX_D_DECL(1.,1.,1.)});
        int const periodic[] = {AMREX_D_DECL(1,1,1)};
        amrex::Geometry const geometry(domain, &bounds, 0, periodic);
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
            auto lower = domain;
            auto upper = domain;
            lower.setBig(axis, 11);
            upper.setSmall(axis, 12);
            amrex::BoxList list;
            list.push_back(upper);
            list.push_back(lower);
            amrex::BoxArray const boxes(list);
            amrex::Vector<int> ranks{0, amrex::ParallelDescriptor::NProcs() > 1 ? 1 : 0};
            amrex::DistributionMapping const distribution(ranks);
            amrex::iMultiFab mask(boxes, distribution, 1, 1);
            amrex::MultiFab velocity(boxes, distribution, 3, 0);
            for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
                auto const m = mask.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    int const iv[] = {i,j,k};
                    m(i,j,k) = iv[axis] < 12 ? 1 : 0;
                });
            }
            mask.FillBoundary(geometry.periodicity());
            auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(axis));
            amrex::MultiFab leakage(faces, distribution, 1, 0);
            amrex::MultiFab inventory(faces, distribution, 4, 0);
            int const normal = Physical(axis);
            int const tangent = (normal+1)%3;
            for (int selected : {8,12}) {
                int const sign = selected == 8 ? -1 : 1;
                for (int moving : {0,1}) {
                    for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
                        auto const v = velocity.array(mfi);
                        auto const m = mask.const_array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            for (int n = 0; n < 3; ++n) {
                                v(i,j,k,n) = moving && n == tangent ?
                                    (m(i,j,k) == 1 ? 1.e-4 : -2.e-4)*PhysConst::c : 0;
                            }
                        });
                    }
                    for (amrex::MFIter mfi(leakage); mfi.isValid(); ++mfi) {
                        auto const f = leakage.array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            int const iv[] = {i,j,k};
                            double energy = 1;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                if (d != axis) { energy += .125*(iv[d]-8); }
                            }
                            f(i,j,k) = iv[axis] == selected ||
                                (selected == 8 && iv[axis] == 16) ? energy : 0;
                        });
                    }
                    leakage.OverrideSync(geometry.periodicity());
                    EmissionPlan plan;
                    Require(BuildPacketDiffusionLeakageEmissionPlan(leakage, mask, geometry,
                        1.e-15, plan, &inventory, &velocity, axis), "Cartesian face rejected");
                    std::array<amrex::Real, 4> measured{};
                    bool valid = true;
                    for (amrex::Long ray = 0; ray < plan.weight.size(); ++ray) {
                        std::array<double,3> const x{plan.x[ray],plan.y[ray],plan.z[ray]};
                        std::array<double,3> const u{plan.ux[ray],plan.uy[ray],plan.uz[ray]};
                        auto const magnitude = std::hypot(u[0],u[1],u[2]);
                        auto const energy = plan.weight[ray]*PhysConst::m_e*PhysConst::c*magnitude;
                        measured[0] += energy;
                        for (int n = 0; n < 3; ++n) { measured[n+1] += energy*u[n]/magnitude; }
                        valid = valid && sign*u[normal] > 0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            auto const coordinate = x[Physical(d)];
                            int const cell = 8+static_cast<int>(std::floor(8*coordinate));
                            valid = valid && coordinate >= 0 && coordinate < 1;
                            if (d == axis) { valid = valid && cell >= 12 && cell <= 15; }
                            else { valid = valid && coordinate == (cell-8+.5)/8; }
                        }
                    }
                    Require(valid, "birth coordinates or outward ray direction incorrect");
                    amrex::ParallelDescriptor::ReduceRealSum(measured.data(), 4);
                    long count = static_cast<long>(plan.weight.size());
                    amrex::ParallelDescriptor::ReduceLongSum(count);
                    long expected_faces = 1;
                    for (int d = 1; d < AMREX_SPACEDIM; ++d) { expected_faces *= 8; }
                    Require(count == 8*expected_faces, "duplicate or lost shared/periodic face");
                    CheckRedistributedBirths(plan,geometry,boxes,distribution,axis,
                                            selected == 8 ? 15 : 12,8*expected_faces);
                    double const expected_energy = expected_faces*(1+.4375*(AMREX_SPACEDIM-1));
                    Require(std::abs(measured[0]-expected_energy) < 2.e-13*expected_energy,
                            "transverse face identity lost or energy duplicated");
                    for (int n = 0; n < 4; ++n) {
                        Require(std::abs(plan.inventory[n]-measured[n]) < 2.e-13*expected_energy,
                                "stored native photon inventory mismatch");
                    }
                    Require(moving ? measured[tangent+1] > 0 :
                                     std::abs(measured[tangent+1]) < 2.e-13*expected_energy,
                            "tangential velocity uses wrong side or orientation");
                    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
                    amrex::MultiFab host(faces, distribution, 4, 0, host_info);
                    amrex::dtoh_memcpy(host, inventory);
                    amrex::Gpu::streamSynchronize();
                    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
                        auto const data = host.const_array(mfi);
                        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                            auto const& iv = it();
                            double expected = 1;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                if (d != axis) { expected += .125*(iv[d]-8); }
                            }
                            if (iv[axis] != selected && !(selected == 8 && iv[axis] == 16)) {
                                expected = 0;
                            }
                            valid = valid && std::abs(data(iv,0)-expected) < 2.e-13*(1+expected);
                        }
                    }
                    Require(valid, "native ledger lost a face or shared/periodic alias");
                }
            }
        }
        amrex::Print() << "Cartesian native leakage faces passed.\n";
    } catch (std::exception const& error) {
        amrex::Print() << error.what() << '\n';
        status = 1;
    }
    amrex::Finalize();
    return status;
}

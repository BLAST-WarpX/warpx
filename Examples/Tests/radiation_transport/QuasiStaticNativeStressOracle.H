/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QUASI_STATIC_NATIVE_STRESS_ORACLE_H_
#define WARPX_QUASI_STATIC_NATIVE_STRESS_ORACLE_H_

#include "Radiation/PacketDiffusionNativeInterval.H"
#include "Radiation/QuasiStaticInterfaceTraction.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
// Caller supplies an accepted quasi-static interval; K-cell collisions are
// outside this oracle. Shared nodal face copies are read per owned D cell.
inline void CheckQuasiStaticNativeStress (PacketDiffusionNativeIntervalResult const& result,
    amrex::iMultiFab const& mask, amrex::Geometry const& geometry, amrex::Real dt,
    amrex::MultiFab const* old_conserved_q = nullptr,
    amrex::MultiFab const* new_conserved_q = nullptr)
{
    AMREX_ALWAYS_ASSERT(result.valid && AMREX_SPACEDIM == 1);
    AMREX_ALWAYS_ASSERT((old_conserved_q == nullptr) == (new_conserved_q == nullptr));
    AMREX_ALWAYS_ASSERT(mask.nGrowVect().allGE(amrex::IntVect(1)));
    auto const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    auto copy = [&] (amrex::MultiFab const& source) {
        amrex::MultiFab host(source.boxArray(), source.DistributionMap(), source.nComp(),
                             source.nGrowVect(), host_info);
        amrex::dtoh_memcpy(host, source);
        return host;
    };
    auto crossing = copy(result.interface_crossing_inventory);
    auto leakage = copy(result.interface_leakage_inventory);
    auto pressure = copy(*result.field.face_pressure[0]);
    auto actual = copy(result.actual_material_impulse_field);
    auto carry = copy(result.material_momentum_carry_change_field);
    amrex::MultiFab old_q, new_q;
    if (old_conserved_q) {
        old_q = copy(*old_conserved_q);
        new_q = copy(*new_conserved_q);
    }
    amrex::iMultiFab regions(mask.boxArray(), mask.DistributionMap(), 1,
                             mask.nGrowVect(), host_info);
    amrex::dtoh_memcpy(regions, mask);
    amrex::Gpu::streamSynchronize();
    double const c = PhysConst::c;
    double volume = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { volume *= geometry.CellSize(d); }
    double const area = volume/geometry.CellSize(0);
    for (amrex::MFIter mfi(regions); mfi.isValid(); ++mfi) {
        auto const region = regions.const_array(mfi);
        auto const crossed = crossing.const_array(mfi), leaked = leakage.const_array(mfi);
        auto const stress = pressure.const_array(mfi);
        auto const impulse = actual.const_array(mfi), remainder = carry.const_array(mfi);
        auto const q_before = old_conserved_q ? old_q.const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        auto const q_after = new_conserved_q ? new_q.const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        for (int i = mfi.validbox().smallEnd(0); i <= mfi.validbox().bigEnd(0); ++i) {
            if (region(i,0,0) != 1) { continue; }
            std::vector<QuasiStaticInterfaceFace> faces(2);
            std::array<double, 3> scale{};
            for (int n = 0; n < 2; ++n) {
                int const side = 2*n-1, face = i+n;
                auto& value = faces[n];
                value.face_id = static_cast<std::uint64_t>(n);
                value.side = side;
                value.face_area = area;
                if (region(i+side,0,0) == 0) {
                    value.kind = QuasiStaticInterfaceFaceKind::RayBoundary;
                    for (int k = 0; k < 4; ++k) {
                        value.ray_incoming[k] = crossed(face,0,0,k);
                        value.ray_outgoing[k] = crossed(face,0,0,k+4) + leaked(face,0,0,k);
                        if (k > 0) { scale[k-1] += (std::abs(crossed(face,0,0,k)) +
                            std::abs(crossed(face,0,0,k+4)) + std::abs(leaked(face,0,0,k)))/c; }
                    }
                } else {
                    value.kind = QuasiStaticInterfaceFaceKind::IsotropicInterior;
                    value.interior_pressure = stress(face,0,0);
                    scale[2] += dt*area*std::abs(value.interior_pressure);
                }
            }
            auto const reference = ComputeQuasiStaticInterfaceTraction(faces, dt, c);
            AMREX_ALWAYS_ASSERT(reference.valid);
            for (int d = 0; d < 3; ++d) {
                double const dq = old_conserved_q
                    ? (q_after(i,0,0,d)-q_before(i,0,0,d))/c : 0;
                if (old_conserved_q) {
                    scale[d] += (std::abs(q_after(i,0,0,d))+
                                 std::abs(q_before(i,0,0,d)))/c;
                }
                double const observed = impulse(i,0,0,d) + remainder(i,0,0,d);
                double const roundoff = 64*std::numeric_limits<double>::epsilon() *
                    (scale[d] + std::abs(impulse(i,0,0,d)) + std::abs(remainder(i,0,0,d)));
                AMREX_ALWAYS_ASSERT(std::isfinite(observed) && std::isfinite(scale[d]) &&
                                    std::isfinite(roundoff));
                AMREX_ALWAYS_ASSERT(std::abs(observed-(reference.material_impulse[d]-dq)) <=
                                    std::max(1.e-11*scale[d], roundoff));
            }
        }
    }
}
}
#endif

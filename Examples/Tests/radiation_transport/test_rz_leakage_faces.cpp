/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/PacketDiffusionInterfacePhotons.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Radiation/RZMomentGeometry.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
void Require (bool value, char const* message)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    if (!value) { throw std::runtime_error(message); }
}
template<typename Data>
auto Host (Data const& values)
{
    std::vector<typename Data::value_type> result(values.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.end(), result.begin());
    return result;
}
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    int status = 0;
    try {
        using namespace warpx::radiation;
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& photons = sim.GetPartContainer().GetParticleContainerFromName("photons");
        auto const& geometry = sim.Geom(0);
        auto const& boxes = photons.ParticleBoxArray(0);
        auto const& distribution = photons.ParticleDistributionMap(0);
        RZMomentMetric const metric(geometry);
        amrex::iMultiFab mask(boxes,distribution,1,1);
        amrex::MultiFab velocity(boxes,distribution,3,0);
        for (int axis : {0,1}) {
            int const split = geometry.Domain().smallEnd(axis)+geometry.Domain().length(axis)/2;
            auto const face_boxes = amrex::convert(boxes,amrex::IntVect::TheDimensionVector(axis));
            amrex::MultiFab energy(face_boxes,distribution,1,0);
            amrex::MultiFab ledger(face_boxes,distribution,5,0);
            for (int sign : {-1,1}) {
                for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
                    auto const m = mask.array(mfi);
                    auto const v = velocity.array(mfi);
                    amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        int const index = axis == 0 ? i : j;
                        bool const diffusion = sign > 0 ? index < split : index >= split;
                        m(i,j,k) = diffusion ? 1 : 0;
                        v(i,j,k,0) = 1.e-4*PhysConst::c;
                        v(i,j,k,1) = (diffusion ? 1.5e-4 : -1.5e-4)*PhysConst::c;
                        v(i,j,k,2) = -2.e-4*PhysConst::c;
                    });
                }
                mask.FillBoundary(geometry.periodicity());
                for (int event = 0; event < (axis == 1 ? 2 : 1); ++event) {
                    int const selected = event == 0 ? split : geometry.Domain().smallEnd(axis);
                    int const periodic_alias = geometry.Domain().bigEnd(axis)+1;
                    int const outward = event == 0 ? sign : -sign;
                    int const target_cell = event == 0 ? split-(outward < 0 ? 1 : 0) :
                        (outward > 0 ? geometry.Domain().smallEnd(axis) : geometry.Domain().bigEnd(axis));
                    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
                        auto const e = energy.array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            int const index = axis == 0 ? i : j;
                            e(i,j,k) = index == selected || (event == 1 && index == periodic_alias) ?
                                1+.125*(axis == 0 ? j : i) : 0;
                        });
                    }
                    energy.OverrideSync(geometry.periodicity());
                    detail::EmissionPlan plan;
                    Require(BuildRZPacketDiffusionLeakageEmissionPlan(energy,mask,geometry,
                        1.e-15,plan,&ledger,&velocity,axis), "valid RZ leakage rejected");
                    PhotonPacketTransaction trial(photons);
                    Require(detail::StageEmissionPlan(trial,plan), "native RZ leakage birth staging failed");
                    std::array<amrex::Real,5> actual{};
                    long count = 0;
                    bool valid = true;
                    for (auto const& [key,tile] : trial.Particles().GetParticles(0)) {
                        amrex::ignore_unused(key);
                        auto const& soa = tile.GetStructOfArrays();
                        auto const marks = Host(soa.GetIntData(tile.NumIntComps()-1));
                        auto const r = Host(soa.GetRealData(PIdx::r));
                        auto const theta = Host(soa.GetRealData(PIdx::theta));
                        auto const z = Host(soa.GetRealData(PIdx::z));
                        auto const w = Host(soa.GetRealData(PIdx::w));
                        auto const ux = Host(soa.GetRealData(PIdx::ux));
                        auto const uy = Host(soa.GetRealData(PIdx::uy));
                        auto const uz = Host(soa.GetRealData(PIdx::uz));
                        for (std::size_t p = 0; p < marks.size(); ++p) {
                            if (!marks[p]) { continue; }
                            bool const previous_valid = valid;
                            ++count;
                            auto const factor = static_cast<long double>(w[p])*PhysConst::m_e*PhysConst::c;
                            actual[0] += factor*std::hypot(ux[p],uy[p],uz[p]);
                            actual[1] += factor*ux[p]; actual[2] += factor*uy[p];
                            actual[3] += factor*uz[p]; actual[4] += factor*r[p]*uy[p];
                            int const i = geometry.Domain().smallEnd(0)+static_cast<int>(std::floor(
                                (r[p]-geometry.ProbLo(0))*geometry.InvCellSize(0)));
                            int const j = geometry.Domain().smallEnd(1)+static_cast<int>(std::floor(
                                (z[p]-geometry.ProbLo(1))*geometry.InvCellSize(1)));
                            int const normal_cell = axis == 0 ? i : j;
                            valid = valid && theta[p] == 0 && normal_cell == target_cell;
                            valid = valid && outward*(axis == 0 ? ux[p] : uz[p]) > 0;
                        if (axis == 1) {
                            long double const lower = geometry.ProbLo(0)+
                                static_cast<long double>(i-geometry.Domain().smallEnd(0))*geometry.CellSize(0);
                            long double const upper = lower+geometry.CellSize(0);
                            auto const requested = static_cast<amrex::ParticleReal>(
                                std::sqrt((lower*lower+upper*upper)/2));
                            // Production and this independent long-double
                            // integral can round the radius differently.
                            valid = valid && std::abs(r[p]-std::sqrt(requested*requested)) <=
                                8*std::numeric_limits<amrex::ParticleReal>::epsilon()*requested;
                            }
                            if (previous_valid && !valid) {
                                std::cerr << std::setprecision(17) << "RZ birth mismatch axis=" << axis
                                    << " sign=" << sign << " event=" << event << " r=" << r[p]
                                    << " z=" << z[p] << " theta=" << theta[p] << " i=" << i
                                    << " j=" << j << " target=" << target_cell << " ux=" << ux[p]
                                    << " uz=" << uz[p] << '\n';
                            }
                        }
                    }
                    Require(valid,"stored RZ birth has wrong kinetic cell, radius, theta or direction");
                    amrex::ParallelDescriptor::ReduceRealSum(actual.data(),5);
                    amrex::ParallelDescriptor::ReduceLongSum(count);
                    Require(count == 8L*geometry.Domain().length(1-axis),"RZ face duplication or loss");
                    auto const owner = ledger.OwnerMask(geometry.periodicity());
                    auto const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
                    amrex::MultiFab host_ledger(face_boxes,distribution,5,0,host_info);
                    amrex::iMultiFab host_owner(face_boxes,distribution,1,0,host_info);
                    amrex::dtoh_memcpy(host_ledger,ledger);
                    amrex::dtoh_memcpy(host_owner,*owner);
                    amrex::Gpu::streamSynchronize();
                    std::array<amrex::Real,5> reported{};
                    for (amrex::MFIter mfi(host_ledger); mfi.isValid(); ++mfi) {
                        auto const data = host_ledger.const_array(mfi);
                        auto const own = host_owner.const_array(mfi);
                        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                            auto const& iv = it();
                            if (!own(iv)) { continue; }
                            for (int n = 0; n < 5; ++n) { reported[n] += data(iv,n); }
                            double const expected = iv[axis] == selected ||
                                (event == 1 && iv[axis] == periodic_alias) ? 1+.125*iv[1-axis] : 0;
                        valid = valid && std::abs(data(iv,0)-expected) <= 2.e-13*(1+expected);
                        if (axis == 1 && expected > 0) {
                            long double const lower = geometry.ProbLo(0)+
                                static_cast<long double>(iv[0]-geometry.Domain().smallEnd(0))*
                                geometry.CellSize(0);
                            long double const upper = lower+geometry.CellSize(0);
                            long double const mean_radius = (2.L/3)*
                                (lower*lower+lower*upper+upper*upper)/(lower+upper);
                            long double const mean_squared_radius = (lower*lower+upper*upper)/2;
                            // Independent continuum angular flux under rigid
                            // rotation: cLz/E = beta_theta_conjugate*<r^2>/rbar.
                            // The normalized cosine law's quadratic tangential
                            // terms cancel; third derivative <=2200 gives the
                            // bound below for |beta|<3e-4. This distinguishes
                            // RMS from mean-radius births even in the axis cell.
                            long double const expected_angular = expected*1.5e-4L*
                                mean_squared_radius/mean_radius;
                            long double const bound = expected*upper*(2200.L/6)*
                                3.e-4L*3.e-4L*3.e-4L+2.e-13L*expected;
                            valid = valid && std::abs(data(iv,4)-expected_angular) <= bound;
                        }
                        }
                    }
                    Require(valid,"RZ ledger applies an extra annular energy factor");
                    amrex::ParallelDescriptor::ReduceRealSum(reported.data(),5);
                    for (int n = 0; n < 5; ++n) {
                        Require(std::abs(reported[n]-actual[n]) <= 2.e-13*actual[0],
                                "RZ ledger differs from actual native E/cp/cLz");
                    }
                    Require(actual[2] > 0 && actual[4] > 0,"swirl uses wrong material side or frame");
                    double swirl_beta = 1.5e-4;
                    if (axis == 0) {
                        int const emitting_cell = selected-(outward > 0 ? 1 : 0);
                        swirl_beta *= metric.Radius(selected)/metric.Angular(emitting_cell).mean_radius;
                    } else {
                        long double weighted = 0, total = 0;
                        for (int i = geometry.Domain().smallEnd(0); i <= geometry.Domain().bigEnd(0); ++i) {
                            long double const lower = geometry.ProbLo(0)+
                                static_cast<long double>(i-geometry.Domain().smallEnd(0))*geometry.CellSize(0);
                            long double const upper = lower+geometry.CellSize(0);
                            long double const mean = (2.L/3)*
                                (lower*lower+lower*upper+upper*upper)/(lower+upper);
                            long double const rms = std::sqrt((lower*lower+upper*upper)/2);
                            weighted += (1+.125L*i)*rms/mean;
                            total += 1+.125L*i;
                        }
                        swirl_beta *= static_cast<double>(weighted/total);
                    }
                    // Independent continuum first derivatives of normalized
                    // D^4 cosine intensity: tangent=beta_t, normal=2s/3+2beta_n/9.
                    // The normalized moment Hessian is bounded by 106 over
                    // |beta|<=.001; 53*|beta|^2 is below 6e-6 in these cases.
                    Require(std::abs(actual[2]/actual[0]-swirl_beta) < 6.e-6,
                            "RZ swirl was not evaluated at birth radius from conjugate velocity");
                    auto const normal_beta = axis == 0 ? 1.e-4 : -2.e-4;
                    auto const normal_moment = actual[axis == 0 ? 1 : 3]/actual[0];
                    Require(std::abs(normal_moment-outward*2./3-2.*normal_beta/9) < 6.e-6,
                            "RZ normal moving-intensity response is incorrect");
                    Require(trial.ValidateLive(),"RZ leakage changed live photons");
                }
            }
            if (axis == 0) {
                int const boundary = geometry.Domain().smallEnd(0);
                energy.setVal(0);
                for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
                    auto const e = energy.array(mfi);
                    amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        if (i == boundary) { e(i,j,k) = 1; }
                    });
                }
                detail::EmissionPlan rejected;
                Require(!BuildRZPacketDiffusionLeakageEmissionPlan(energy,mask,geometry,
                    1.e-15,rejected,&ledger,&velocity,axis), "radial reflecting face leaked");
                Require(rejected.weight.empty() && ledger.norm0() == 0,
                        "rejected radial boundary leakage retained a plan or ledger");
            }
        }
        amrex::Print() << "Native annular RZ leakage E/cp/cLz faces passed.\n";
    } catch (std::exception const& error) {
        amrex::Print() << error.what() << '\n';
        status = 1;
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
    return status;
}

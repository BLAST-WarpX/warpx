/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/FluxRelaxedPackets.H"
#include "Radiation/PacketDiffusionInterface.H"
#include "Radiation/PacketDiffusionInterfacePhotons.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iomanip>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
using EmissionPlan = warpx::radiation::detail::EmissionPlan;
using ParticleType = WarpXParticleContainer::ParticleType;

constexpr double kOpacity = 25.0;
constexpr amrex::Real kDt = 1.0e-10_rt;
constexpr double kLeakagePhotonEnergy = 1.0e-13;

void
ReduceSum (double* values, int count)
{
    amrex::ParallelDescriptor::ReduceRealSum(values, static_cast<int>(count));
}

void
Require (bool condition, char const* message)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(condition, message);
}

void
InitializeMaterials (amrex::iMultiFab& mask, amrex::MultiFab& opacity,
                     amrex::MultiFab& faces, amrex::Geometry const& geometry,
                     std::vector<int> const& d_cells)
{
    mask.setVal(0);
    opacity.setVal(0.0);
    faces.setVal(0.0);
    for (int const cell : d_cells)
    {
        amrex::Box const region{amrex::IntVect(cell), amrex::IntVect(cell)};
        mask.setVal(1, region, 0);
        opacity.setVal(static_cast<amrex::Real>(kOpacity), region, 0);
    }
    mask.FillBoundary(geometry.periodicity());
    opacity.FillBoundary(geometry.periodicity());
    faces.FillBoundary(geometry.periodicity());
}

std::map<int, double>
OwnerFaceSums (amrex::MultiFab const& faces, amrex::Geometry const& geometry, int component = 0)
{
    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    amrex::MultiFab host_faces(faces.boxArray(), faces.DistributionMap(), faces.nComp(), faces.nGrowVect(),
                               host_info);
    amrex::dtoh_memcpy(host_faces, faces);
    amrex::Gpu::streamSynchronize();
    auto const owner = host_faces.OwnerMask(geometry.periodicity());
    amrex::iMultiFab host_owner(owner->boxArray(), owner->DistributionMap(), 1, 0, host_info);
    amrex::dtoh_memcpy(host_owner, *owner);
    amrex::Gpu::streamSynchronize();
    std::map<int, double> sums;
    int const domain_lo = geometry.Domain().smallEnd(0);
    int const ncells = geometry.Domain().length(0);
    for (amrex::MFIter iterator(host_faces); iterator.isValid(); ++iterator)
    {
        auto const face_array = host_faces.const_array(iterator);
        auto const owner_array = host_owner.const_array(iterator);
        for (int face = iterator.validbox().smallEnd(0);
             face <= iterator.validbox().bigEnd(0); ++face)
        {
            if (owner_array(face, 0, 0, 0) == 0)
            {
                continue;
            }
            int const canonical =
                domain_lo + (((face - domain_lo) % ncells + ncells) % ncells);
            sums[canonical] += static_cast<double>(face_array(face, 0, 0, component));
        }
    }
    for (int face = domain_lo; face < domain_lo + ncells; ++face)
    {
        double value = sums[face];
        amrex::ParallelDescriptor::ReduceRealSum(value);
        sums[face] = value;
    }
    return sums;
}

struct AngularMoments
{
    double energy = 0.0;
    double cpx = 0.0;
    double cpy = 0.0;
    double cpz = 0.0;
    double mu_z_squared = 0.0;
    long count = 0;
};

AngularMoments
LocalAngularMoments (EmissionPlan const& plan)
{
    AngularMoments result;
    result.count = static_cast<long>(plan.weight.size());
    long const ray_count = static_cast<long>(plan.weight.size());
    for (long i = 0; i < ray_count; ++i)
    {
        long double const speed = std::sqrt(
            static_cast<long double>(plan.ux[i]) * plan.ux[i] +
            static_cast<long double>(plan.uy[i]) * plan.uy[i] +
            static_cast<long double>(plan.uz[i]) * plan.uz[i]);
        long double const energy =
            static_cast<long double>(plan.weight[i]) * PhysConst::m_e * PhysConst::c * speed;
        long double const ux = plan.ux[i] / speed;
        long double const uy = plan.uy[i] / speed;
        long double const uz = plan.uz[i] / speed;
        result.energy += static_cast<double>(energy);
        result.cpx += static_cast<double>(energy * ux);
        result.cpy += static_cast<double>(energy * uy);
        result.cpz += static_cast<double>(energy * uz);
        result.mu_z_squared += static_cast<double>(energy * uz * uz);
    }
    return result;
}

AngularMoments
GlobalAngularMoments (EmissionPlan const& plan)
{
    AngularMoments result = LocalAngularMoments(plan);
    ReduceSum(&result.energy, 4);
    amrex::ParallelDescriptor::ReduceLongSum(result.count);
    amrex::ParallelDescriptor::ReduceRealSum(result.mu_z_squared);
    return result;
}

struct NeumaierSum
{
    long double sum = 0.0L;
    long double correction = 0.0L;

    void
    add (long double value)
    {
        long double const next = sum + value;
        if (std::abs(sum) >= std::abs(value))
        {
            correction += (sum - next) + value;
        }
        else
        {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    [[nodiscard]] long double
    value () const
    {
        return sum + correction;
    }
};

struct InventoryOracle
{
    std::array<NeumaierSum, 4> components{};
    std::array<NeumaierSum, 4> absolute_components{};
    long count = 0;

    void
    addOperands (long double energy, long double cpx, long double cpy, long double cpz)
    {
        std::array<long double, 4> const values{energy, cpx, cpy, cpz};
        for (int component = 0; component < 4; ++component)
        {
            components[component].add(values[component]);
            absolute_components[component].add(std::abs(values[component]));
        }
        ++count;
    }
};

bool
AddPlanInventory (EmissionPlan const& plan, InventoryOracle& inventory)
{
    bool const same_length = plan.x.size() == plan.y.size() &&
                             plan.x.size() == plan.z.size() &&
                             plan.x.size() == plan.ux.size() &&
                             plan.x.size() == plan.uy.size() &&
                             plan.x.size() == plan.uz.size() &&
                             plan.x.size() == plan.weight.size();
    if (!same_length)
    {
        return false;
    }
    long const ray_count = static_cast<long>(plan.weight.size());
    for (long i = 0; i < ray_count; ++i)
    {
        long double const ux = plan.ux[i];
        long double const uy = plan.uy[i];
        long double const uz = plan.uz[i];
        long double const speed_squared = ux * ux + uy * uy + uz * uz;
        if (!(plan.weight[i] > 0.0_prt) || !std::isfinite(plan.weight[i]) ||
            !std::isfinite(speed_squared) || !(speed_squared > 0.0L))
        {
            return false;
        }
        long double const speed = std::sqrt(speed_squared);
        long double const energy = static_cast<long double>(plan.weight[i]) *
                                   static_cast<long double>(PhysConst::m_e) *
                                   static_cast<long double>(PhysConst::c) * speed;
        if (!std::isfinite(energy) || !(energy > 0.0L))
        {
            return false;
        }
        inventory.addOperands(energy, energy * ux / speed, energy * uy / speed,
                              energy * uz / speed);
    }
    return true;
}

void
ReduceStableLongDouble (long double& value)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(value);
#endif
}

InventoryOracle
GlobalInventory (InventoryOracle local, bool& valid)
{
    for (int component = 0; component < 4; ++component)
    {
        long double value = local.components[component].value();
        long double absolute = local.absolute_components[component].value();
        ReduceStableLongDouble(value);
        ReduceStableLongDouble(absolute);
        local.components[component] = NeumaierSum{value, 0.0L};
        local.absolute_components[component] = NeumaierSum{absolute, 0.0L};
    }
    amrex::ParallelDescriptor::ReduceLongSum(local.count);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return {};
    }
    return local;
}

InventoryOracle
IndependentPhotonInventory (WarpXParticleContainer::Base& photons, bool& valid)
{
    InventoryOracle local;
    bool local_valid = true;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        long const tile_count = tile.numParticles();
        if (tile_count == 0)
        {
            continue;
        }
        auto const& soa = tile.GetStructOfArrays();
        amrex::Vector<amrex::ParticleReal> weight(tile_count);
        amrex::Vector<amrex::ParticleReal> ux(tile_count);
        amrex::Vector<amrex::ParticleReal> uy(tile_count);
        amrex::Vector<amrex::ParticleReal> uz(tile_count);
        amrex::Vector<std::uint64_t> idcpu(tile_count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::w).begin(),
                         soa.GetRealData(PIdx::w).begin() + tile_count, weight.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::ux).begin(),
                         soa.GetRealData(PIdx::ux).begin() + tile_count, ux.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::uy).begin(),
                         soa.GetRealData(PIdx::uy).begin() + tile_count, uy.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::uz).begin(),
                         soa.GetRealData(PIdx::uz).begin() + tile_count, uz.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                         soa.GetIdCPUData().begin() + tile_count, idcpu.begin());
        amrex::Gpu::streamSynchronize();
        for (long i = 0; i < tile_count; ++i)
        {
            if (!amrex::ParticleIDWrapper{idcpu[i]}.is_valid() || !(weight[i] > 0.0_prt))
            {
                continue;
            }
            long double const local_ux = ux[i];
            long double const local_uy = uy[i];
            long double const local_uz = uz[i];
            long double const speed_squared = local_ux * local_ux + local_uy * local_uy +
                                              local_uz * local_uz;
            if (!std::isfinite(weight[i]) || !std::isfinite(speed_squared) ||
                !(speed_squared > 0.0L))
            {
                local_valid = false;
                break;
            }
            long double const speed = std::sqrt(speed_squared);
            long double const energy = static_cast<long double>(weight[i]) *
                                       static_cast<long double>(PhysConst::m_e) *
                                       static_cast<long double>(PhysConst::c) * speed;
            if (!std::isfinite(energy) || !(energy > 0.0L))
            {
                local_valid = false;
                break;
            }
            local.addOperands(energy, energy * local_ux / speed,
                              energy * local_uy / speed, energy * local_uz / speed);
        }
        if (!local_valid)
        {
            break;
        }
        amrex::ignore_unused(key);
    }
    valid = local_valid;
    return GlobalInventory(std::move(local), local_valid);
}

InventoryOracle
IndependentPlanInventory (EmissionPlan const& plan, bool& valid)
{
    InventoryOracle local;
    valid = AddPlanInventory(plan, local);
    return GlobalInventory(std::move(local), valid);
}

InventoryOracle
AddInventories (InventoryOracle const& left, InventoryOracle const& right)
{
    InventoryOracle result;
    for (int component = 0; component < 4; ++component)
    {
        result.components[component].add(left.components[component].value());
        result.components[component].add(right.components[component].value());
        result.absolute_components[component].add(
            left.absolute_components[component].value());
        result.absolute_components[component].add(
            right.absolute_components[component].value());
    }
    result.count = left.count + right.count;
    return result;
}

[[nodiscard]] long double
InventoryRoundoffBound (InventoryOracle const& left, InventoryOracle const& right,
                        int component, bool production_precision = false)
{
    long const term_count = std::max(left.count, right.count);
    int const rank_count = amrex::ParallelDescriptor::NProcs();
    long double const uncancelled = std::max(
        left.absolute_components[component].value(),
        right.absolute_components[component].value());
    long double const unit_roundoff = production_precision
        ? static_cast<long double>(std::numeric_limits<amrex::Real>::epsilon()) / 2.0L
        : std::numeric_limits<long double>::epsilon() / 2.0L;
    // Two separately evaluated inventories: per-ray arithmetic, local sums,
    // and rank reduction. gamma_n includes higher-order rounding terms.
    long double const operation_count =
        16.0L + 2.0L * static_cast<long double>(term_count + rank_count + 1);
    long double const accumulated_roundoff = operation_count * unit_roundoff;
    Require(accumulated_roundoff < 1.0L, "Inventory error envelope is undefined.");
    return accumulated_roundoff / (1.0L - accumulated_roundoff) * uncancelled;
}

[[nodiscard]] bool
InventoryWithinRoundoff (InventoryOracle const& actual, InventoryOracle const& expected,
                         int component)
{
    long double const difference = actual.components[component].value() -
                                   expected.components[component].value();
    return std::abs(difference) <= InventoryRoundoffBound(actual, expected, component);
}

void
RequireIndependentInventory (char const* label, InventoryOracle const& actual,
                             InventoryOracle const& expected)
{
    Require(actual.count == expected.count, "Independent native inventory term count differs.");
    for (int component = 0; component < 4; ++component)
    {
        long double const difference = actual.components[component].value() -
                                       expected.components[component].value();
        long double const bound = InventoryRoundoffBound(actual, expected, component);
        amrex::Print() << std::scientific << std::setprecision(17)
                       << "native independent inventory label=" << label
                       << " component=" << component
                       << " difference=" << static_cast<double>(difference)
                       << " bound=" << static_cast<double>(bound)
                       << " uncancelled_operands="
                       << static_cast<double>(expected.absolute_components[component].value())
                       << " terms=" << expected.count
                       << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
        Require(InventoryWithinRoundoff(actual, expected, component),
                "Independent compensated native inventory conservation failed.");
    }
}

void
RequireProductionInventory (char const* label, amrex::GpuArray<amrex::Real, 4> const& production,
                            InventoryOracle const& oracle)
{
    InventoryOracle production_oracle;
    production_oracle.count = oracle.count;
    for (int component = 0; component < 4; ++component)
    {
        production_oracle.components[component].add(production[component]);
        production_oracle.absolute_components[component].add(std::abs(production[component]));
    }
    for (int component = 0; component < 4; ++component)
    {
        long double const difference = production_oracle.components[component].value() -
                                       oracle.components[component].value();
        long double const bound =
            InventoryRoundoffBound(production_oracle, oracle, component, true);
        amrex::Print() << std::scientific << std::setprecision(17)
                       << "native production inventory label=" << label
                       << " component=" << component
                       << " production=" << production[component]
                       << " oracle=" << static_cast<double>(oracle.components[component].value())
                       << " difference=" << static_cast<double>(difference)
                       << " bound=" << static_cast<double>(bound)
                       << " uncancelled_operands="
                       << static_cast<double>(oracle.absolute_components[component].value())
                       << " terms=" << oracle.count
                       << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
        Require(std::abs(difference) <= bound,
                "Production native inventory exceeds its explicit roundoff bound.");
    }
}

void
TestInventoryCounterexample ()
{
    InventoryOracle expected;
    expected.addOperands(1.0L, 1.0L, 0.0L, 0.0L);
    expected.addOperands(1.0L, -1.0L, 0.0L, 0.0L);
    expected.addOperands(1.0L, 1.0e-12L, 0.0L, 0.0L);

    InventoryOracle perturbed;
    perturbed.addOperands(1.0L, 1.0L, 0.0L, 0.0L);
    perturbed.addOperands(1.0L, -1.0L, 0.0L, 0.0L);
    perturbed.addOperands(1.0L, 0.0L, 0.0L, 0.0L);

    bool const caught = !InventoryWithinRoundoff(perturbed, expected, 1);
    long double const difference = perturbed.components[1].value() -
                                   expected.components[1].value();
    long double const bound = InventoryRoundoffBound(perturbed, expected, 1);
    amrex::Print() << std::scientific << std::setprecision(17)
                   << "native inventory counterexample difference="
                   << static_cast<double>(difference) << " bound="
                   << static_cast<double>(bound)
                   << " ranks=" << amrex::ParallelDescriptor::NProcs()
                   << " caught=" << caught << '\n';
    Require(caught, "Compensated inventory gate missed a cancelled genuine loss.");
}

void
CheckQuadrature (EmissionPlan const& plan, double expected_energy,
                 double expected_mean_mu = 2.0 / 3.0)
{
    AngularMoments const moments = GlobalAngularMoments(plan);
    Require(moments.count % 8 == 0, "Leakage quadrature count is not a multiple of eight.");
    double const scale = std::abs(expected_energy) + std::numeric_limits<double>::min();
    Require(std::abs(moments.energy - expected_energy) <= 1.e-11 * scale,
            "Leakage quadrature energy is incorrect.");
    Require(std::abs(moments.cpx) <= 1.e-11 * scale &&
                std::abs(moments.cpy) <= 1.e-11 * scale,
            "Leakage quadrature transverse mean is not zero.");
    Require(std::abs(moments.cpz / moments.energy - expected_mean_mu) <= 1.e-11,
            "Leakage quadrature mean mu is not 2/3.");
    Require(std::abs(moments.mu_z_squared / moments.energy - 0.5) <= 1.e-11,
            "Leakage quadrature second angular moment is not 1/2.");
}

EmissionPlan
InitialNativePlan ()
{
    EmissionPlan plan;
    if (amrex::ParallelDescriptor::MyProc() != 0)
    {
        return plan;
    }

    std::array<amrex::ParticleReal, 6> const weight = {
        3.25_prt, 0.125_prt, 7.0_prt, 2.0_prt, 5.0_prt, 0.5_prt};
    std::array<amrex::ParticleReal, 6> const ux = {
        300.0_prt, -300.0_prt, 0.0_prt, 0.99498743710662_prt, 0.0_prt, 600.0_prt};
    std::array<amrex::ParticleReal, 6> const uy = {
        400.0_prt, -400.0_prt, 0.0_prt, 0.0_prt, 0.0_prt, 800.0_prt};
    std::array<amrex::ParticleReal, 6> const uz = {
        500.0_prt, -500.0_prt, 900.0_prt, 0.1_prt, 1000.0_prt, 0.0_prt};
    long double const first_mu = 500.0L / std::sqrt(500000.0L);
    amrex::Real const first_z = static_cast<amrex::Real>(
        0.125L - static_cast<long double>(PhysConst::c) * kDt * first_mu);
    std::array<amrex::ParticleReal, 6> const z = {
        first_z, 0.39_prt, 0.5625_prt, 0.105_prt,
        static_cast<amrex::ParticleReal>(0.75 - PhysConst::c * kDt), 0.6875_prt};

    for (int i = 0; i < 6; ++i)
    {
        plan.x.push_back(0.0_prt);
        plan.y.push_back(0.0_prt);
        plan.z.push_back(z[i]);
        plan.ux.push_back(ux[i]);
        plan.uy.push_back(uy[i]);
        plan.uz.push_back(uz[i]);
        plan.weight.push_back(weight[i]);
    }
    return plan;
}

std::array<long double, 2>
ExpectedInterfaceSplit (int packet)
{
    std::array<long double, 2> momentum{};
    std::array<long double, 3> u{};
    long double weight = 0.0L;
    if (packet == 0)
    {
        u = {300.0L, 400.0L, 500.0L};
        weight = 3.25L;
    }
    else
    {
        u = {-300.0L, -400.0L, -500.0L};
        weight = 0.125L;
    }
    long double const speed = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    long double const mu = std::abs(u[2]) / speed;
    double const probability = warpx::radiation::PacketDiffusionInterfaceProbability(
        kOpacity / 8.0, mu);
    long double const energy =
        weight * PhysConst::m_e * PhysConst::c * speed;
    momentum[0] = probability * energy;
    momentum[1] = (1.0L - static_cast<long double>(probability)) * energy;
    return momentum;
}

void
CheckPrivatePositions (WarpXParticleContainer::Base& photons)
{
    bool found_shallow = false;
    bool found_exact_endpoint = false;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        long const tile_count = tile.numParticles();
        if (tile_count == 0)
        {
            continue;
        }
        auto const& soa = tile.GetStructOfArrays();
        amrex::Vector<amrex::ParticleReal> z(tile_count);
        amrex::Vector<amrex::ParticleReal> uz(tile_count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
                         soa.GetRealData(PIdx::z).begin() + tile_count, z.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::uz).begin(),
                         soa.GetRealData(PIdx::uz).begin() + tile_count, uz.begin());
        amrex::Gpu::streamSynchronize();
        for (long i = 0; i < tile_count; ++i)
        {
            found_shallow = found_shallow ||
                            (std::abs(static_cast<double>(uz[i]) - 0.1) < 1.e-12);
            if (std::abs(static_cast<double>(uz[i]) - 0.1) < 1.e-12)
            {
                Require(z[i] > 0.105_prt && z[i] < 0.125_prt,
                        "Projected shallow ray crossed a D face too early.");
            }
            found_exact_endpoint = found_exact_endpoint ||
                                   (std::abs(static_cast<double>(uz[i]) - 1000.0) < 1.e-12);
            if (std::abs(static_cast<double>(uz[i]) - 1000.0) < 1.e-12)
            {
                Require(std::abs(static_cast<double>(z[i]) - 0.75) <=
                            64.0 * std::numeric_limits<double>::epsilon(),
                        "Exact-endpoint K/K ray did not finish on its face.");
            }
        }
        amrex::ignore_unused(key);
    }
    int flags[2] = {found_shallow ? 1 : 0, found_exact_endpoint ? 1 : 0};
    amrex::ParallelDescriptor::ReduceIntMax(flags, 2);
    Require(flags[0] == 1, "Projected shallow survivor was lost.");
    Require(flags[1] == 1, "Exact-endpoint survivor was lost.");
}

void
CheckNativeIds (WarpXParticleContainer::Base& photons)
{
    long const count = photons.TotalNumberOfParticles();
    Require(count == 36, "Committed native photon count is incorrect.");
    std::set<std::uint64_t> local_ids;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        long const tile_count = tile.numParticles();
        if (tile_count == 0)
        {
            continue;
        }
        auto const& ids = tile.GetStructOfArrays().GetIdCPUData();
        amrex::Vector<std::uint64_t> host_ids(tile_count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, ids.begin(), ids.end(), host_ids.begin());
        amrex::Gpu::streamSynchronize();
        for (long i = 0; i < tile_count; ++i)
        {
            std::uint64_t const value = host_ids[i];
            Require(amrex::ParticleIDWrapper{value}.is_valid(),
                    "Committed photon has an invalid native ID.");
            Require(local_ids.insert(value).second, "Duplicate native ID on one rank.");
        }
        amrex::ignore_unused(key);
    }
}

void
TestNonzeroDomainPeriodicLeakage ()
{
    amrex::Box const domain(amrex::IntVect(10), amrex::IntVect(17));
    amrex::RealBox const real_box(2.0, 3.0);
    amrex::Array<int, 1> const periodic{{1}};
    amrex::Geometry const geometry(domain, real_box, amrex::CoordSys::cartesian, periodic);
    amrex::BoxArray cells(domain);
    cells.maxSize(4);
    amrex::DistributionMapping const distribution(amrex::Vector<int>{
        0, amrex::ParallelDescriptor::NProcs() > 1 ? 1 : 0});
    amrex::iMultiFab mask(cells, distribution, 1, 1);
    amrex::MultiFab opacity(cells, distribution, 1, 1);
    amrex::BoxArray const nodes = amrex::convert(cells, amrex::IntVect::TheUnitVector());
    amrex::MultiFab leakage(nodes, distribution, 1, 1);
    InitializeMaterials(mask, opacity, leakage, geometry, {12, 13, 17});

    constexpr double seam_energy = 4.5e-13;
    for (amrex::MFIter iterator(leakage); iterator.isValid(); ++iterator)
    {
        auto leakage_array = leakage.array(iterator);
        amrex::ParallelFor(iterator.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                leakage_array(i, j, k) = i == 18 ? seam_energy : 0.0;
            });
    }
    leakage.SumBoundary(geometry.periodicity());
    leakage.OverrideSync(geometry.periodicity());

    EmissionPlan plan;
    using warpx::radiation::BuildPacketDiffusionLeakageEmissionPlan;
    Require(BuildPacketDiffusionLeakageEmissionPlan(
                leakage, mask, geometry, kLeakagePhotonEnergy, plan),
            "Nonzero-index periodic leakage builder failed.");
    Require(plan.inventory[0] > 0.0, "Periodic seam leakage plan is empty.");
    CheckQuadrature(plan, seam_energy);
    long local_in_k = 0;
    long local_outside = 0;
    for (amrex::Long i = 0; i < plan.z.size(); ++i)
    {
        bool const in_upper_k = plan.z[i] >= 2.0_prt && plan.z[i] < 2.125_prt;
        bool const in_lower_k = plan.z[i] >= 2.0_prt && plan.z[i] < 2.125_prt;
        local_in_k += (in_upper_k || in_lower_k) ? 1 : 0;
        local_outside += (plan.z[i] < 2.0_prt || plan.z[i] >= 3.0_prt) ? 1 : 0;
    }
    amrex::ParallelDescriptor::ReduceLongSum(local_in_k);
    amrex::ParallelDescriptor::ReduceLongSum(local_outside);
    Require(local_in_k == 8, "Periodic seam birth was not placed in K.");
    Require(local_outside == 0, "Periodic seam birth escaped the domain.");
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto& live = simulation.GetPartContainer().GetParticleContainerFromName("photons");
        auto const& geometry = simulation.Geom(0);
        amrex::BoxArray const cells = live.ParticleBoxArray(0);
        amrex::DistributionMapping const distribution = live.ParticleDistributionMap(0);
        amrex::iMultiFab mask(cells, distribution, 1, 1);
        amrex::MultiFab opacity(cells, distribution, 1, 1);
        amrex::BoxArray const faces_ba =
            amrex::convert(cells, amrex::IntVect::TheUnitVector());
        amrex::MultiFab admissions(faces_ba, distribution, 1, 1);
        InitializeMaterials(mask, opacity, admissions, geometry, {1, 2});
        Require(cells.size() == 2, "Native fixture did not create two owned grids.");

        EmissionPlan empty_leakage_plan;
        amrex::MultiFab empty_leakage(faces_ba, distribution, 1, 1);
        empty_leakage.setVal(0.0);
        Require(warpx::radiation::BuildPacketDiffusionLeakageEmissionPlan(
                    empty_leakage, mask, geometry, kLeakagePhotonEnergy, empty_leakage_plan),
                "Empty leakage builder failed.");
        Require(empty_leakage_plan.x.empty(), "Zero leakage produced photons.");

        long const before_rejection = ParticleType::NextID();
        {
            warpx::radiation::PhotonPacketTransaction transaction(live);
            EmissionPlan invalid;
            if (amrex::ParallelDescriptor::MyProc() == 0)
            {
                invalid.x.push_back(0.0_prt);
                invalid.y.push_back(0.0_prt);
                invalid.z.push_back(0.1875_prt);
                invalid.ux.push_back(0.0_prt);
                invalid.uy.push_back(0.0_prt);
                invalid.uz.push_back(1000.0_prt);
                invalid.weight.push_back(1.0_prt);
            }
            Require(warpx::radiation::detail::StageEmissionPlan(transaction, invalid),
                    "Staging initial-D rejection packet failed.");
            warpx::radiation::PacketDiffusionInterfacePhotonReport report;
            EmissionPlan returns;
            Require(!warpx::radiation::AdvancePacketDiffusionCandidatePhotons(
                        transaction.Particles(), mask, opacity, geometry, kDt, admissions,
                        returns, report),
                    "An initial-D photon was accepted by the native interface.");
            Require(transaction.Particles().TotalNumberOfParticles() == 1,
                    "Numerical rejection changed the private candidate count.");
            Require(live.TotalNumberOfParticles() == 0,
                    "Numerical rejection changed live photons.");
        }
        Require(ParticleType::NextID() == before_rejection + 1,
                "Numerical rejection consumed native photon IDs.");

        long const before_empty = ParticleType::NextID();
        {
            warpx::radiation::PhotonPacketTransaction transaction(live);
            warpx::radiation::PacketDiffusionInterfacePhotonReport report;
            EmissionPlan returns;
            amrex::MultiFab crossing_inventory(faces_ba, distribution, 8, 1);
            Require(warpx::radiation::AdvancePacketDiffusionCandidatePhotons(
                        transaction.Particles(), mask, opacity, geometry, kDt, admissions,
                        returns, report, {}, &crossing_inventory),
                    "Empty native interface trial failed.");
            Require(report.initial_inventory[0] == 0.0 && returns.x.empty(),
                    "Empty native interface trial was not empty.");
        }
        Require(ParticleType::NextID() == before_empty + 1,
                "Empty native interface trial consumed native photon IDs.");

        TestInventoryCounterexample();

        std::array<long double, 2> const split0 = ExpectedInterfaceSplit(0);
        std::array<long double, 2> const split1 = ExpectedInterfaceSplit(1);
        long const before_commit = ParticleType::NextID();
        long expected_local_births = 0;
        {
            warpx::radiation::PhotonPacketTransaction transaction(live);
            EmissionPlan initial = InitialNativePlan();
            Require(warpx::radiation::detail::StageEmissionPlan(transaction, initial),
                    "Staging initial native packets failed.");
            bool inventory_valid = false;
            InventoryOracle const initial_oracle =
                IndependentPhotonInventory(transaction.Particles(), inventory_valid);
            Require(inventory_valid, "Independent initial native inventory is invalid.");

            warpx::radiation::PacketDiffusionInterfacePhotonReport report;
            EmissionPlan returns;
            amrex::MultiFab crossing_inventory(faces_ba, distribution, 8, 1);
            Require(warpx::radiation::AdvancePacketDiffusionCandidatePhotons(
                        transaction.Particles(), mask, opacity, geometry, kDt, admissions,
                        returns, report, {}, &crossing_inventory),
                    "Native packet interface advance failed.");
            Require(report.incident_packets == 2,
                    "Native packet interface incident count is incorrect.");
            Require(report.transported_packets == 4,
                    "Native packet interface survivor count is incorrect.");
            Require(report.reflected_rays == 16,
                    "Native packet interface reflected-ray count is incorrect.");
            Require(transaction.Particles().TotalNumberOfParticles() == 4,
                    "Invalid incident packets were not removed from the private candidate.");
            CheckPrivatePositions(transaction.Particles());
            InventoryOracle const survivor_oracle =
                IndependentPhotonInventory(transaction.Particles(), inventory_valid);
            Require(inventory_valid, "Independent surviving native inventory is invalid.");
            InventoryOracle const returns_oracle =
                IndependentPlanInventory(returns, inventory_valid);
            Require(inventory_valid, "Independent reflected-plan inventory is invalid.");
            RequireProductionInventory("initial", report.initial_inventory, initial_oracle);
            RequireProductionInventory("survivors", report.final_inventory, survivor_oracle);
            RequireProductionInventory("reflected plan", returns.inventory, returns_oracle);
            for (int n = 0; n < 4; ++n) {
                auto const incident = OwnerFaceSums(crossing_inventory, geometry, n);
                auto const reflected = OwnerFaceSums(crossing_inventory, geometry, n+4);
                double incident_sum = 0;
                double reflected_sum = 0;
                for (auto const& [face, value] : incident) {
                    incident_sum += value;
                    Require(face == 1 || face == 3 || value == 0,
                        "Incident angular inventory landed on a noninterface face.");
                }
                for (auto const& [face, value] : reflected) {
                    reflected_sum += value;
                    Require(face == 1 || face == 3 || value == 0,
                        "Return angular inventory landed on a noninterface face.");
                }
                double const scale = report.initial_inventory[0];
                Require(std::abs(incident_sum - (report.initial_inventory[n] -
                    report.final_inventory[n])) <= 1.e-12*scale,
                    "Unique-face incident inventory differs from removed native photons.");
                Require(std::abs(reflected_sum - returns.inventory[n]) <= 1.e-12*scale,
                    "Unique-face return inventory differs from actual reflected photons.");
            }

            double const admitted_scale = static_cast<double>(split0[0] + split1[0]);
            Require(std::abs(report.admitted_energy - admitted_scale) <=
                        1.e-12 * admitted_scale,
                    "Native packet admitted energy differs from scalar reference.");
            double const reflected_scale = static_cast<double>(split0[1] + split1[1]);
            Require(std::abs(report.reflected_energy - reflected_scale) <=
                        1.e-12 * reflected_scale,
                    "Native packet reflected energy differs from scalar reference.");
            Require(std::abs(report.initial_inventory[0] - report.final_inventory[0] -
                             report.reflected_energy - report.admitted_energy) <=
                        1.e-12 * report.initial_inventory[0],
                    "Native packet energy split does not conserve energy.");

            auto const owner_sums = OwnerFaceSums(admissions, geometry);
            Require(std::count_if(owner_sums.begin(), owner_sums.end(),
                                  [] (auto const& entry) { return entry.second != 0.0; }) == 2,
                    "Admissions were deposited on wrong faces.");
            Require(std::abs(owner_sums.at(1) - static_cast<double>(split0[0])) <=
                        1.e-12 * static_cast<double>(split0[0]),
                    "Positive-direction admission has wrong owner sum.");
            Require(std::abs(owner_sums.at(3) - static_cast<double>(split1[0])) <=
                        1.e-12 * static_cast<double>(split1[0]),
                    "Negative-direction admission has wrong owner sum.");

            amrex::MultiFab leakage(faces_ba, distribution, 1, 1);
            leakage.setVal(0.0);
            constexpr double leakage_energies[2] = {2.4e-13, 3.7e-13};
            for (amrex::MFIter iterator(leakage); iterator.isValid(); ++iterator)
            {
                auto leakage_array = leakage.array(iterator);
                amrex::ParallelFor(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        leakage_array(i, j, k) =
                            i == 1 ? leakage_energies[0] :
                                     (i == 3 ? leakage_energies[1] : 0.0);
                    });
            }
            leakage.SumBoundary(geometry.periodicity());
            leakage.OverrideSync(geometry.periodicity());
            EmissionPlan leakage_plan;
            amrex::MultiFab outgoing_inventory(faces_ba, distribution, 4, 1);
            Require(warpx::radiation::BuildPacketDiffusionLeakageEmissionPlan(
                        leakage, mask, geometry, kLeakagePhotonEnergy, leakage_plan,
                        &outgoing_inventory),
                    "Native leakage plan builder failed.");
            InventoryOracle const leakage_oracle =
                IndependentPlanInventory(leakage_plan, inventory_valid);
            Require(inventory_valid, "Independent leakage-plan inventory is invalid.");
            RequireProductionInventory("leakage plan", leakage_plan.inventory, leakage_oracle);
            for (int n = 0; n < 4; ++n) {
                auto const outgoing = OwnerFaceSums(outgoing_inventory, geometry, n);
                double sum = 0;
                for (auto const& [face, value] : outgoing) {
                    sum += value;
                    Require(face == 1 || face == 3 || value == 0,
                        "Leakage angular inventory landed on a noninterface face.");
                }
                Require(std::abs(sum - leakage_plan.inventory[n]) <=
                    1.e-12*leakage_plan.inventory[0],
                    "Unique-face leakage inventory differs from native emission plan.");
            }
            CheckQuadrature(leakage_plan, leakage_energies[0] + leakage_energies[1],
                (2.0 / 3.0) * (leakage_energies[1] - leakage_energies[0]) /
                    (leakage_energies[0] + leakage_energies[1]));

            Require(warpx::radiation::detail::StageEmissionPlan(transaction, returns),
                    "Staging reflected native photons failed.");
            Require(warpx::radiation::detail::StageEmissionPlan(transaction, leakage_plan),
                    "Staging leakage native photons failed.");
            amrex::GpuArray<amrex::Real, 4> after_staging{};
            Require(warpx::radiation::detail::PhotonInventory(
                        transaction.Particles(), after_staging),
                    "Actual staged native inventory is invalid.");
            InventoryOracle const staged_oracle =
                IndependentPhotonInventory(transaction.Particles(), inventory_valid);
            Require(inventory_valid, "Independent staged native inventory is invalid.");
            InventoryOracle const expected_staged = AddInventories(
                AddInventories(survivor_oracle, returns_oracle), leakage_oracle);
            RequireProductionInventory("staged", after_staging, staged_oracle);
            RequireIndependentInventory("accepted staged photons", staged_oracle,
                                        expected_staged);
            Require(transaction.ValidateLive(), "Native transaction validation failed.");
            for (auto const& [key, tile] : transaction.Particles().GetParticles(0))
            {
                expected_local_births += tile.numParticles();
                amrex::ignore_unused(key);
            }
            transaction.Commit();
            CheckNativeIds(live);
        }
        Require(ParticleType::NextID() == before_commit + expected_local_births + 1,
                "Native commit consumed the wrong number of particle IDs.");

        TestNonzeroDomainPeriodicLeakage();
        amrex::Print() << "Native packet diffusion interface: events, conservation, IDs, and "
                          "periodic leakage pass.\n";
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
    return 0;
}

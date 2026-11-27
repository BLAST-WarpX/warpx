/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "PacketDiffusionInterfacePhotons.H"

#include "MovingLeakageAngularQuadrature.H"
#include "PacketDiffusionInterface.H"
#include "PhotonFourMomentum.H"
#include "RZMomentGeometry.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleUtil.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace warpx::radiation
{
namespace
{
using namespace amrex::literals;

using Base = WarpXParticleContainer::Base;
using EmissionPlan = detail::EmissionPlan;

constexpr int kD = 1;
constexpr int kK = 0;

struct ParticleRecord
{
    std::pair<int, int> tile{};
    long index = 0;
    std::uint64_t idcpu = 0;
    amrex::ParticleReal z = 0.0_prt;
    amrex::ParticleReal w = 0.0_prt;
    amrex::ParticleReal ux = 0.0_prt;
    amrex::ParticleReal uy = 0.0_prt;
    amrex::ParticleReal uz = 0.0_prt;
};

struct FaceEvent
{
    int face = 0;
    double energy = 0.0;
    amrex::GpuArray<amrex::Real, 8> inventory{};
};

/** Diagnostic inventory must use the actual stored native photon, not its
 * nominal construction energy or unrounded direction. */
amrex::GpuArray<amrex::Real, 4>
storedInventory (amrex::ParticleReal weight, amrex::ParticleReal ux,
                 amrex::ParticleReal uy, amrex::ParticleReal uz)
{
    long double const factor = static_cast<long double>(weight) * PhysConst::m_e * PhysConst::c;
    long double const speed = std::sqrt(static_cast<long double>(ux)*ux +
        static_cast<long double>(uy)*uy + static_cast<long double>(uz)*uz);
    return {static_cast<amrex::Real>(factor*speed), static_cast<amrex::Real>(factor*ux),
        static_cast<amrex::Real>(factor*uy), static_cast<amrex::Real>(factor*uz)};
}

struct FaceLedgerTrial
{
    amrex::MultiFab* output;
    bool accepted = false;
    bool clearable = false;
    explicit FaceLedgerTrial (amrex::MultiFab* value) : output(value) {}
    ~FaceLedgerTrial ()
    {
        if (!accepted && clearable) { output->setVal(0); }
    }
    bool compatible (amrex::MultiFab const& faces, int components)
    {
        int low = output ? 1 : 0;
        int high = low;
        amrex::ParallelDescriptor::ReduceIntMin(low);
        amrex::ParallelDescriptor::ReduceIntMax(high);
        bool valid = low == high;
        if (output) {
            valid = valid && output->ok() && faces.ok() && output != &faces &&
                output->boxArray() == faces.boxArray() &&
                output->DistributionMap() == faces.DistributionMap() &&
                output->ixType() == faces.ixType() && output->nComp() == components &&
                output->nGrowVect() == faces.nGrowVect();
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        clearable = output && output != &faces && output->ok();
        if (clearable) { output->setVal(0); }
        return valid;
    }
};

void
clearPlan (EmissionPlan& plan)
{
    plan.x.clear();
    plan.y.clear();
    plan.z.clear();
    plan.ux.clear();
    plan.uy.clear();
    plan.uz.clear();
    plan.weight.clear();
    plan.logical_draw_keys.clear();
    plan.inventory = {};
}

void
removeInvalidParticles (Base& photons)
{
    for (auto& [key, tile] : photons.GetParticles(0))
    {
        amrex::removeInvalidParticles(tile);
        amrex::ignore_unused(key);
    }
}

bool
collectiveAnd (bool local)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(local);
    return local;
}

bool
finitePositive (double value)
{
    return std::isfinite(value) && value > 0.0;
}

bool
finiteNonnegative (double value)
{
    return std::isfinite(value) && value >= 0.0;
}

int
cellIndex (amrex::Geometry const& geometry, amrex::ParticleReal z)
{
    amrex::Real const fraction = (static_cast<amrex::Real>(z) - geometry.ProbLo(0)) *
                                  geometry.InvCellSize(0);
    long const index = static_cast<long>(std::floor(fraction));
    int const domain_lo = geometry.Domain().smallEnd(0);
    int const ncells = geometry.Domain().length(0);
    long const wrapped = domain_lo +
                         (index % static_cast<long>(ncells) +
                          static_cast<long>(ncells)) %
                             static_cast<long>(ncells);
    return static_cast<int>(wrapped);
}

amrex::Real
wrapCoordinate (amrex::Geometry const& geometry, amrex::Real z)
{
    amrex::Real const lo = geometry.ProbLo(0);
    amrex::Real const length = geometry.ProbHi(0) - lo;
    while (z < lo)
    {
        z += length;
    }
    while (z >= geometry.ProbHi(0))
    {
        z -= length;
    }
    return z;
}

amrex::Real
faceCoordinate (amrex::Geometry const& geometry, int face)
{
    return geometry.ProbLo(0) +
           static_cast<amrex::Real>(face - geometry.Domain().smallEnd(0)) *
               geometry.CellSize(0);
}

amrex::Real
periodicSafeBirthCoordinate (amrex::Geometry const& geometry, int face, int direction)
{
    amrex::Real const coordinate = faceCoordinate(geometry, face);
    int const domain_hi = geometry.Domain().bigEnd(0) + 1;
    if (face == geometry.Domain().smallEnd(0) && direction < 0)
    {
        return std::nextafter(geometry.ProbHi(0), geometry.ProbLo(0));
    }
    if (face == domain_hi && direction > 0)
    {
        return std::nextafter(geometry.ProbLo(0), geometry.ProbHi(0));
    }
    return std::nextafter(coordinate, coordinate + static_cast<amrex::Real>(direction));
}

std::vector<ParticleRecord>
readParticles (Base& photons)
{
    std::vector<ParticleRecord> records;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        long const count = tile.numParticles();
        if (count == 0)
        {
            continue;
        }
        auto const& soa = tile.GetStructOfArrays();
        amrex::Vector<amrex::ParticleReal> z(count);
        amrex::Vector<amrex::ParticleReal> w(count);
        amrex::Vector<amrex::ParticleReal> ux(count);
        amrex::Vector<amrex::ParticleReal> uy(count);
        amrex::Vector<amrex::ParticleReal> uz(count);
        amrex::Vector<std::uint64_t> idcpu(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
                         soa.GetRealData(PIdx::z).begin() + count, z.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::w).begin(),
                         soa.GetRealData(PIdx::w).begin() + count, w.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::ux).begin(),
                         soa.GetRealData(PIdx::ux).begin() + count, ux.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::uy).begin(),
                         soa.GetRealData(PIdx::uy).begin() + count, uy.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::uz).begin(),
                         soa.GetRealData(PIdx::uz).begin() + count, uz.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                         soa.GetIdCPUData().begin() + count, idcpu.begin());
        amrex::Gpu::streamSynchronize();
        for (long i = 0; i < count; ++i)
        {
            records.push_back(ParticleRecord{key, i, idcpu[i], z[i], w[i], ux[i], uy[i], uz[i]});
        }
    }
    return records;
}

bool
validParticleRecord (ParticleRecord const& record)
{
    return record.idcpu != amrex::ParticleIdCpus::Invalid &&
           std::isfinite(record.z) && std::isfinite(record.w) && record.w > 0.0_prt &&
           std::isfinite(record.ux) && std::isfinite(record.uy) && std::isfinite(record.uz) &&
           (std::abs(record.ux) > 0.0_prt || std::abs(record.uy) > 0.0_prt ||
            std::abs(record.uz) > 0.0_prt);
}

bool
cellIsD (amrex::iMultiFab const& host_mask, int grid, int cell)
{
    return host_mask.array(grid)(cell, 0, 0, 0) == kD;
}

double
cellOpacity (amrex::MultiFab const& host_opacity, int grid, int cell)
{
    return static_cast<double>(host_opacity.array(grid)(cell, 0, 0, 0));
}

bool
addReflectedRays (EmissionPlan& plan, amrex::Geometry const& geometry,
                  ParticleRecord const& record, int uncanonical_face, int sign,
                  double reflected_energy, double remaining_time)
{
    if (!(reflected_energy > 0.0))
    {
        return true;
    }
    long double const ux = record.ux;
    long double const uy = record.uy;
    long double const uz = record.uz;
    long double const speed_squared =
        ux * ux + uy * uy + uz * uz;
    if (!(speed_squared > 0.0L))
    {
        return false;
    }
    long double const speed = std::sqrt(speed_squared);
    long double const one_photon_energy = PhysConst::m_e * PhysConst::c * speed;
    if (!finitePositive(static_cast<double>(one_photon_energy)))
    {
        return false;
    }

    amrex::Real const outward_normal = -static_cast<amrex::Real>(sign);
    amrex::Real const birth_coordinate =
        periodicSafeBirthCoordinate(geometry, uncanonical_face,
                                    static_cast<int>(outward_normal));

    constexpr long double sqrt6 = 2.4494897427831780981972840747058913919659474806567L;
    constexpr std::array<long double, 2> mus = {
        (6.0L - sqrt6) / 10.0L, (6.0L + sqrt6) / 10.0L};
    constexpr std::array<long double, 2> weights = {
        (9.0L - sqrt6) / 18.0L, (9.0L + sqrt6) / 18.0L};
    constexpr std::array<long double, 4> cosines = {1.0L, 0.0L, -1.0L, 0.0L};
    constexpr std::array<long double, 4> sines = {0.0L, 1.0L, 0.0L, -1.0L};

    for (int node = 0; node < 2; ++node)
    {
        long double const transverse = std::sqrt(1.0L - mus[node] * mus[node]);
        for (int azimuth = 0; azimuth < 4; ++azimuth)
        {
            long double const ray_energy = reflected_energy * weights[node] / 4.0L;
            long double const proposed_weight = ray_energy / one_photon_energy;
            if (!(ray_energy > 0.0L) || !(proposed_weight > 0.0L))
            {
                return false;
            }
            long double const nx = transverse * cosines[azimuth];
            long double const ny = transverse * sines[azimuth];
            long double const nz = mus[node] * static_cast<long double>(outward_normal);

            amrex::Real const final_coordinate = wrapCoordinate(
                geometry,
                birth_coordinate +
                    static_cast<amrex::Real>(nz) * PhysConst::c *
                        static_cast<amrex::Real>(remaining_time));
            if (!std::isfinite(final_coordinate))
            {
                return false;
            }
            plan.x.push_back(0.0_prt);
            plan.y.push_back(0.0_prt);
            plan.z.push_back(static_cast<amrex::ParticleReal>(final_coordinate));
            amrex::ParticleReal const stored_ux = static_cast<amrex::ParticleReal>(speed * nx);
            amrex::ParticleReal const stored_uy = static_cast<amrex::ParticleReal>(speed * ny);
            amrex::ParticleReal const stored_uz = static_cast<amrex::ParticleReal>(speed * nz);
            amrex::ParticleReal const stored_weight =
                static_cast<amrex::ParticleReal>(proposed_weight);
            if (!std::isfinite(stored_ux) || !std::isfinite(stored_uy) ||
                !std::isfinite(stored_uz) || !std::isfinite(stored_weight) ||
                stored_weight <= 0.0_prt)
            {
                return false;
            }

            plan.ux.push_back(stored_ux);
            plan.uy.push_back(stored_uy);
            plan.uz.push_back(stored_uz);
            plan.weight.push_back(stored_weight);

            auto const actual = storedInventory(stored_weight, stored_ux, stored_uy, stored_uz);
            for (int n = 0; n < 4; ++n) { plan.inventory[n] += actual[n]; }
        }
    }
    return true;
}

void
writeParticleUpdates (Base& photons, std::vector<ParticleRecord> const& records)
{
    auto& tiles = photons.GetParticles(0);
    struct UpdateIndex
    {
        std::pair<int, int> tile;
        long particle_index;
        std::size_t record_index;
    };
    std::vector<UpdateIndex> indices;
    for (ParticleRecord const& record : records)
    {
        indices.push_back(UpdateIndex{
            record.tile, record.index,
            static_cast<std::size_t>(std::distance(records.data(), &record))});
    }
    std::sort(indices.begin(), indices.end(), [] (UpdateIndex const& lhs, UpdateIndex const& rhs) {
        return std::tie(lhs.tile, lhs.particle_index) <
               std::tie(rhs.tile, rhs.particle_index);
    });
    std::size_t cursor = 0;
    while (cursor < indices.size())
    {
        auto const key = indices[cursor].tile;
        std::size_t last = cursor;
        while (last < indices.size() && indices[last].tile == key)
        {
            ++last;
        }
        auto found = tiles.find(key);
        if (found == tiles.end())
        {
            amrex::Abort("Private photon tile disappeared during interface transport.");
        }
        auto& soa = found->second.GetStructOfArrays();
        long const count = found->second.numParticles();
        amrex::Vector<amrex::ParticleReal> z(count);
        amrex::Vector<amrex::ParticleReal> w(count);
        amrex::Vector<std::uint64_t> idcpu(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
                         soa.GetRealData(PIdx::z).begin() + count, z.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::w).begin(),
                         soa.GetRealData(PIdx::w).begin() + count, w.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                         soa.GetIdCPUData().begin() + count, idcpu.begin());
        amrex::Gpu::streamSynchronize();
        for (std::size_t item = cursor; item < last; ++item)
        {
            long const index = indices[item].particle_index;
            ParticleRecord const& record = records[indices[item].record_index];
            z[index] = record.z;
            w[index] = record.w;
            idcpu[index] = record.idcpu;
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, z.begin(), z.end(),
                         soa.GetRealData(PIdx::z).begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, w.begin(), w.end(),
                         soa.GetRealData(PIdx::w).begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, idcpu.begin(), idcpu.end(),
                         soa.GetIdCPUData().begin());
        amrex::Gpu::streamSynchronize();
        cursor = last;
    }
}

bool
validateFieldTopology (Base& photons, amrex::iMultiFab const& mask,
                       amrex::MultiFab const& opacity, amrex::MultiFab& faces,
                       amrex::Geometry const& geometry, amrex::Real dt,
                       PacketDiffusionInterfacePhotonControls const& controls)
{
#if !defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(photons, mask, opacity, faces, geometry, dt, controls);
    bool valid = false;
#else
    amrex::IntVect const unit = amrex::IntVect::TheUnitVector();
    bool const periodic = geometry.isAllPeriodic();
    bool valid = geometry.Coord() == amrex::CoordSys::cartesian && periodic &&
                 geometry.Domain().length(0) > 1 &&
                 finitePositive(controls.marshak_extrapolation_lambda);
    valid = valid && mask.ok() && opacity.ok() && faces.ok() &&
            photons.finestLevel() == 0 &&
            photons.ParticleBoxArray(0) == mask.boxArray() &&
            photons.ParticleDistributionMap(0) == mask.DistributionMap() &&
            mask.ixType().cellCentered() && mask.nComp() == 1 &&
            mask.nGrowVect().allGE(unit) && opacity.boxArray() == mask.boxArray() &&
            opacity.DistributionMap() == mask.DistributionMap() && opacity.nComp() == 1 &&
            opacity.nGrowVect().allGE(unit) &&
            faces.boxArray() == amrex::convert(mask.boxArray(), unit) &&
            faces.DistributionMap() == mask.DistributionMap() &&
    faces.ixType()[0] == amrex::IndexType::NODE && faces.nComp() == 1 &&
            faces.nGrowVect().allGE(unit) && std::isfinite(dt) && dt > 0.0 &&
            PhysConst::c * dt <=
                0.5 * geometry.CellSize(0) +
                8.0 * geometry.CellSize(0) * std::numeric_limits<amrex::Real>::epsilon();
#endif
    return collectiveAnd(valid);
}

bool
validateMaterialFields (amrex::iMultiFab const& mask, amrex::MultiFab const& opacity,
                        amrex::Geometry const& geometry,
                        PacketDiffusionInterfacePhotonControls const& controls)
{
    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    amrex::iMultiFab host_mask(mask.boxArray(), mask.DistributionMap(), 1, mask.nGrowVect(),
                               host_info);
    amrex::MultiFab host_opacity(opacity.boxArray(), opacity.DistributionMap(), 1,
                                 opacity.nGrowVect(), host_info);
    amrex::dtoh_memcpy(host_mask, mask);
    amrex::dtoh_memcpy(host_opacity, opacity);
    amrex::Gpu::streamSynchronize();

    bool valid = true;
    for (amrex::MFIter iterator(host_mask); iterator.isValid() && valid; ++iterator)
    {
        auto const mask_array = host_mask.const_array(iterator);
        auto const opacity_array = host_opacity.const_array(iterator);
        auto const box = iterator.fabbox();
        for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
        {
            int const value = mask_array(i, 0, 0, 0);
            double const chi = static_cast<double>(opacity_array(i, 0, 0, 0));
            if ((value != kD && value != kK) || !std::isfinite(chi) || chi < 0.0 ||
                (value == kD && chi == 0.0) ||
                (value == kK && chi != 0.0))
            {
                valid = false;
                break;
            }
        }
        for (int i = box.smallEnd(0); valid && i < box.bigEnd(0); ++i)
        {
            int const left = mask_array(i, 0, 0, 0);
            int const right = mask_array(i + 1, 0, 0, 0);
            bool const interface = (left == kD && right == kK) ||
                                   (left == kK && right == kD);
            if (!interface)
            {
                continue;
            }
            int const d_cell = left == kD ? i : i + 1;
            double const tau = static_cast<double>(opacity_array(d_cell, 0, 0, 0)) *
                               geometry.CellSize(0);
            double const probability = PacketDiffusionInterfaceProbability(
                tau, 1.0, controls.marshak_extrapolation_lambda);
            if (!finitePositive(tau) || !finitePositive(probability) || probability > 1.0)
            {
                valid = false;
                break;
            }
        }
    }
    return collectiveAnd(valid);
}

void
addFaceEventLocally (amrex::MultiFab& host_faces, FaceEvent const& event, bool& valid)
{
    for (amrex::MFIter iterator(host_faces); iterator.isValid(); ++iterator)
    {
        amrex::Box const box = iterator.fabbox();
        if (box.contains(amrex::IntVect(event.face)))
        {
        host_faces.array(iterator)(event.face, 0, 0, 0) +=
            static_cast<amrex::Real>(event.energy);
            return;
        }
    }
    valid = false;
}

} // namespace

bool
AdvancePacketDiffusionCandidatePhotons (
    Base& private_photons, amrex::iMultiFab const& diffusion_mask, amrex::MultiFab const& opacity,
    amrex::Geometry const& geometry, amrex::Real dt, amrex::MultiFab& face_admissions,
    EmissionPlan& diffuse_returns, PacketDiffusionInterfacePhotonReport& report,
    PacketDiffusionInterfacePhotonControls const& controls, amrex::MultiFab* face_inventory)
{
    report = {};
    clearPlan(diffuse_returns);
    FaceLedgerTrial ledger(face_inventory);
    if (!ledger.compatible(face_admissions, 8)) {
        if (face_admissions.ok()) { face_admissions.setVal(0); }
        return false;
    }

    bool const topology_valid = validateFieldTopology(
        private_photons, diffusion_mask, opacity, face_admissions, geometry, dt, controls);
    if (!topology_valid)
    {
        if (face_admissions.ok())
        {
            face_admissions.setVal(0.0);
        }
        return false;
    }

    face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
    if (!validateMaterialFields(diffusion_mask, opacity, geometry, controls))
    {
        face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
        clearPlan(diffuse_returns);
        return false;
    }

    bool valid = detail::PhotonInventory(private_photons, report.initial_inventory);
    if (!collectiveAnd(valid))
    {
        face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
        clearPlan(diffuse_returns);
        report = {};
        return false;
    }

    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    amrex::iMultiFab host_mask(diffusion_mask.boxArray(), diffusion_mask.DistributionMap(), 1,
                               diffusion_mask.nGrowVect(), host_info);
    amrex::MultiFab host_opacity(opacity.boxArray(), opacity.DistributionMap(), 1,
                                 opacity.nGrowVect(), host_info);
    amrex::MultiFab host_faces(face_admissions.boxArray(), face_admissions.DistributionMap(), 1,
                               face_admissions.nGrowVect(), host_info);
    amrex::MultiFab host_inventory;
    if (face_inventory) {
        host_inventory.define(face_inventory->boxArray(), face_inventory->DistributionMap(),
            8, face_inventory->nGrowVect(), host_info);
        host_inventory.setVal(0);
    }
    amrex::dtoh_memcpy(host_mask, diffusion_mask);
    amrex::dtoh_memcpy(host_opacity, opacity);
    host_faces.setVal(0.0);
    amrex::Gpu::streamSynchronize();

    std::vector<ParticleRecord> records = readParticles(private_photons);
    std::vector<FaceEvent> face_events;
    long incident_packets = 0;
    long transported_packets = 0;
    long reflected_rays = 0;
    double admitted_energy = 0.0;
    double reflected_energy = 0.0;

    for (ParticleRecord& record : records)
    {
        if (!validParticleRecord(record))
        {
            valid = false;
            break;
        }
        int const cell = cellIndex(geometry, record.z);
        if (cellIsD(host_mask, record.tile.first, cell))
        {
            valid = false;
            break;
        }

        long double const speed_squared = static_cast<long double>(record.ux) * record.ux +
                                          static_cast<long double>(record.uy) * record.uy +
                                          static_cast<long double>(record.uz) * record.uz;
        long double const speed = std::sqrt(speed_squared);
        int const sign = record.uz > 0.0_prt ? 1 : (record.uz < 0.0_prt ? -1 : 0);
        if (sign == 0)
        {
            ++transported_packets;
            continue;
        }

        int const next_face = sign > 0 ? cell + 1 : cell;
        amrex::Real const face_coordinate = faceCoordinate(geometry, next_face);
        amrex::Real const distance_to_face =
            sign > 0 ? face_coordinate - static_cast<amrex::Real>(record.z)
                     : static_cast<amrex::Real>(record.z) - face_coordinate;
        amrex::Real const distance_tolerance =
            32.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
            (std::abs(face_coordinate) + std::abs(static_cast<amrex::Real>(record.z)) +
             geometry.CellSize(0));
        if (distance_to_face < -distance_tolerance)
        {
            valid = false;
            break;
        }
        amrex::Real const distance = std::max(distance_to_face, amrex::Real(0.0));
        long double const absolute_mu = std::abs(static_cast<long double>(record.uz) / speed);
        amrex::Real const z_displacement =
            static_cast<amrex::Real>(absolute_mu) * PhysConst::c * dt;
        if (distance > z_displacement + distance_tolerance)
        {
            record.z = static_cast<amrex::ParticleReal>(wrapCoordinate(
                geometry, static_cast<amrex::Real>(record.z) +
                              static_cast<amrex::Real>(record.uz / speed) * PhysConst::c * dt));
            ++transported_packets;
            continue;
        }

        int const adjacent = cell + sign;
        if (!cellIsD(host_mask, record.tile.first, adjacent))
        {
            record.z = static_cast<amrex::ParticleReal>(wrapCoordinate(
                geometry, static_cast<amrex::Real>(record.z) +
                              static_cast<amrex::Real>(record.uz / speed) * PhysConst::c * dt));
            ++transported_packets;
            continue;
        }

        double const tau = cellOpacity(host_opacity, record.tile.first, adjacent) *
                           geometry.CellSize(0);
        double const probability = PacketDiffusionInterfaceProbability(
            tau, static_cast<double>(absolute_mu), controls.marshak_extrapolation_lambda);
        long double const energy = static_cast<long double>(record.w) * PhysConst::m_e *
                                   PhysConst::c * speed;
        long double const admitted = static_cast<long double>(probability) * energy;
        long double const reflected =
            (1.0L - static_cast<long double>(probability)) * energy;
        if (!finitePositive(tau) || !finiteNonnegative(absolute_mu) || absolute_mu > 1.0 ||
            !finitePositive(probability) || probability > 1.0 || !finitePositive(energy) ||
            !finiteNonnegative(admitted) || !finiteNonnegative(reflected) ||
            admitted + reflected > (1.0L + 16.0L * std::numeric_limits<double>::epsilon()) * energy)
        {
            valid = false;
            break;
        }

        long double const remaining_path = static_cast<long double>(dt) -
            static_cast<long double>(distance) / (absolute_mu * PhysConst::c);
        long double const time_roundoff =
            static_cast<long double>(distance_tolerance) / (absolute_mu * PhysConst::c);
        if (!std::isfinite(remaining_path) || remaining_path < -time_roundoff ||
            remaining_path > static_cast<long double>(dt) *
                                 (1.0L + 64.0L * std::numeric_limits<double>::epsilon()))
        {
            valid = false;
            break;
        }
        double const remaining_time =
            std::max(0.0, std::min(static_cast<double>(remaining_path), dt));
        auto const first_return = diffuse_returns.weight.size();
        if (!addReflectedRays(diffuse_returns, geometry, record, next_face, sign,
                              static_cast<double>(reflected), remaining_time))
        {
            valid = false;
            break;
        }
        int const reflected_from_packet = static_cast<int>(reflected > 0.0L) * 8;
        reflected_rays += reflected_from_packet;

        FaceEvent event{next_face, static_cast<double>(admitted)};
        auto const incoming = storedInventory(record.w, record.ux, record.uy, record.uz);
        for (int n = 0; n < 4; ++n) { event.inventory[n] = incoming[n]; }
        for (auto r = first_return; r < diffuse_returns.weight.size(); ++r) {
            auto const outgoing = storedInventory(diffuse_returns.weight[r],
                diffuse_returns.ux[r], diffuse_returns.uy[r], diffuse_returns.uz[r]);
            for (int n = 0; n < 4; ++n) { event.inventory[n+4] += outgoing[n]; }
        }
        face_events.push_back(event);
        admitted_energy += static_cast<double>(admitted);
        reflected_energy += static_cast<double>(reflected);
        ++incident_packets;
        record.idcpu = amrex::ParticleIdCpus::Invalid;
        record.w = 0.0_prt;
    }

    valid = valid && diffuse_returns.x.size() == diffuse_returns.y.size() &&
            diffuse_returns.x.size() == diffuse_returns.z.size() &&
            diffuse_returns.x.size() == diffuse_returns.ux.size() &&
            diffuse_returns.x.size() == diffuse_returns.uy.size() &&
            diffuse_returns.x.size() == diffuse_returns.uz.size() &&
            diffuse_returns.x.size() == diffuse_returns.weight.size();
    if (!collectiveAnd(valid))
    {
        face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
        clearPlan(diffuse_returns);
        report = {};
        return false;
    }

    for (FaceEvent const& event : face_events)
    {
        addFaceEventLocally(host_faces, event, valid);
        if (face_inventory) {
            bool deposited = false;
            for (amrex::MFIter mfi(host_inventory); mfi.isValid(); ++mfi) {
                if (mfi.fabbox().contains(amrex::IntVect(event.face))) {
                    auto const values = host_inventory.array(mfi);
                    for (int n = 0; n < 8; ++n) { values(event.face,0,0,n) += event.inventory[n]; }
                    deposited = true;
                    break;
                }
            }
            valid = valid && deposited;
        }
    }
    if (!collectiveAnd(valid))
    {
        face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
        clearPlan(diffuse_returns);
        report = {};
        return false;
    }
    amrex::htod_memcpy(face_admissions, host_faces);
    amrex::Gpu::streamSynchronize();
    face_admissions.SumBoundary(geometry.periodicity());
    face_admissions.OverrideSync(geometry.periodicity());
    if (face_inventory) {
        amrex::htod_memcpy(*face_inventory, host_inventory);
        amrex::Gpu::streamSynchronize();
        face_inventory->SumBoundary(geometry.periodicity());
        face_inventory->OverrideSync(geometry.periodicity());
    }
    writeParticleUpdates(private_photons, records);
    removeInvalidParticles(private_photons);
    private_photons.Redistribute();
    valid = detail::PhotonInventory(private_photons, report.final_inventory);
    if (!collectiveAnd(valid))
       {
        face_admissions.setVal(0.0, 0, 1, face_admissions.nGrowVect());
        clearPlan(diffuse_returns);
        report = {};
        return false;
    }

    amrex::ParallelDescriptor::ReduceLongSum(incident_packets);
    amrex::ParallelDescriptor::ReduceLongSum(transported_packets);
    amrex::ParallelDescriptor::ReduceLongSum(reflected_rays);
    amrex::ParallelDescriptor::ReduceRealSum(admitted_energy);
    amrex::ParallelDescriptor::ReduceRealSum(reflected_energy);
    amrex::ParallelDescriptor::ReduceRealSum(diffuse_returns.inventory.data(), 4);
    report.incident_packets = incident_packets;
    report.transported_packets = transported_packets;
    report.reflected_rays = reflected_rays;
    report.admitted_energy = admitted_energy;
    report.reflected_energy = reflected_energy;
    report.reflected_inventory = diffuse_returns.inventory;
    valid = !face_inventory || face_inventory->is_finite();
    for (auto value : report.reflected_inventory) { valid = valid && std::isfinite(value); }
    valid = valid && std::isfinite(admitted_energy) && std::isfinite(reflected_energy);
    if (!collectiveAnd(valid)) {
        face_admissions.setVal(0);
        clearPlan(diffuse_returns);
        report = {};
        return false;
    }
    ledger.accepted = true;
    return true;
}

namespace
{
bool
BuildLeakageEmissionPlan (
    amrex::MultiFab const& accepted_face_leakage, amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry, amrex::Real photon_energy, EmissionPlan& leakage_plan,
    amrex::MultiFab* face_inventory, amrex::MultiFab const* work_velocity,
    int mesh_direction, bool rz, LogicalPhotonBirthKeyContext const* key_context)
{
    clearPlan(leakage_plan);
    FaceLedgerTrial ledger(face_inventory);
    int minimum_geometry = rz ? 1 : 0;
    int maximum_geometry = minimum_geometry;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_geometry);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_geometry);
    int const inventory_components = rz ? 5 : 4;
    if (!ledger.compatible(accepted_face_leakage, inventory_components) ||
        minimum_geometry != maximum_geometry) { return false; }
    int minimum_mode = work_velocity ? 1 : 0;
    int maximum_mode = minimum_mode;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_mode);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_mode);
    if (minimum_mode != maximum_mode) { return false; }
    int minimum_direction = mesh_direction;
    int maximum_direction = mesh_direction;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_direction);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_direction);
    if (minimum_direction != maximum_direction || mesh_direction < 0 ||
        mesh_direction >= AMREX_SPACEDIM) { return false; }
    int minimum_keys = key_context ? 1 : 0;
    int maximum_keys = minimum_keys;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_keys);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_keys);
    if (minimum_keys != maximum_keys) { return false; }
    if (key_context) {
        amrex::Long minimum[4] = {
            static_cast<amrex::Long>(static_cast<std::uint32_t>(key_context->seed)),
            static_cast<amrex::Long>(key_context->seed >> 32U),
            static_cast<amrex::Long>(static_cast<std::uint32_t>(key_context->interval)),
            static_cast<amrex::Long>(key_context->interval >> 32U)};
        amrex::Long maximum[4] = {minimum[0], minimum[1], minimum[2], minimum[3]};
        amrex::ParallelDescriptor::ReduceLongMin(minimum, 4);
        amrex::ParallelDescriptor::ReduceLongMax(maximum, 4);
        for (int n = 0; n < 4; ++n) {
            if (minimum[n] != maximum[n]) { return false; }
        }
    }
    amrex::IntVect const unit = amrex::IntVect::TheUnitVector();
    auto const face_type = amrex::IntVect::TheDimensionVector(mesh_direction);
    int const physical_axis = AMREX_SPACEDIM == 1 ? 2 :
        (AMREX_SPACEDIM == 2 && mesh_direction == 1 ? 2 : mesh_direction);
#if defined(WARPX_DIM_RZ)
    bool const geometry_valid = rz &&
        (geometry.Coord() == amrex::CoordSys::cartesian || geometry.Coord() == amrex::CoordSys::RZ) &&
        !geometry.isPeriodic(0) && geometry.ProbLo(0) >= 0;
#else
    bool const geometry_valid = !rz && geometry.Coord() == amrex::CoordSys::cartesian;
#endif
    bool valid = std::isfinite(photon_energy) && photon_energy > 0.0 &&
                 geometry_valid &&
                 diffusion_mask.ok() && accepted_face_leakage.ok() &&
                 diffusion_mask.ixType().cellCentered() && diffusion_mask.nComp() == 1 &&
                 diffusion_mask.nGrowVect().allGE(unit) &&
                 accepted_face_leakage.boxArray() ==
                     amrex::convert(diffusion_mask.boxArray(), face_type) &&
                 accepted_face_leakage.DistributionMap() == diffusion_mask.DistributionMap() &&
                 accepted_face_leakage.ixType().toIntVect() == face_type &&
                 accepted_face_leakage.nComp() == 1;
    if (work_velocity) {
        valid = valid && work_velocity->ok() && work_velocity->nComp() == 3 &&
            work_velocity->ixType().cellCentered() &&
            work_velocity->boxArray() == diffusion_mask.boxArray() &&
            work_velocity->DistributionMap() == diffusion_mask.DistributionMap();
    }
    if (!collectiveAnd(valid))
    {
        clearPlan(leakage_plan);
        return false;
    }

    amrex::MFInfo const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    amrex::iMultiFab host_mask(diffusion_mask.boxArray(), diffusion_mask.DistributionMap(), 1,
                               diffusion_mask.nGrowVect(), host_info);
    amrex::MultiFab host_leakage(accepted_face_leakage.boxArray(),
                                 accepted_face_leakage.DistributionMap(), 1,
                                 accepted_face_leakage.nGrowVect(), host_info);
    amrex::dtoh_memcpy(host_mask, diffusion_mask);
    amrex::dtoh_memcpy(host_leakage, accepted_face_leakage);
    amrex::MultiFab host_inventory;
    if (face_inventory) {
        host_inventory.define(face_inventory->boxArray(), face_inventory->DistributionMap(),
            inventory_components, face_inventory->nGrowVect(), host_info);
        host_inventory.setVal(0);
    }
    amrex::MultiFab host_velocity;
    if (work_velocity) {
        amrex::MultiFab velocity(work_velocity->boxArray(), work_velocity->DistributionMap(), 3, 1);
        amrex::MultiFab::Copy(velocity, *work_velocity, 0, 0, 3, 0);
        velocity.FillBoundary(geometry.periodicity());
        host_velocity.define(velocity.boxArray(), velocity.DistributionMap(), 3, 1, host_info);
        amrex::dtoh_memcpy(host_velocity, velocity);
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter mfi(host_velocity); mfi.isValid(); ++mfi) {
            auto const v = host_velocity.const_array(mfi);
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const& iv = cell();
                std::array<double,3> const beta{v(iv,0)/PhysConst::c,
                    v(iv,1)/PhysConst::c, v(iv,2)/PhysConst::c};
                valid = ComputeMovingLeakageAngularQuadrature(beta, 1).valid && valid;
            }
        }
        if (!collectiveAnd(valid)) { return false; }
    }
    amrex::Gpu::streamSynchronize();
    auto const owner = host_leakage.OwnerMask(geometry.periodicity());
    amrex::iMultiFab host_owner(owner->boxArray(), owner->DistributionMap(), 1, 0, host_info);
    amrex::dtoh_memcpy(host_owner, *owner);
    amrex::Gpu::streamSynchronize();

    for (amrex::MFIter iterator(host_leakage); iterator.isValid() && valid; ++iterator)
    {
        auto const leakage = host_leakage.const_array(iterator);
        auto const mask = host_mask.const_array(iterator);
        auto const owner_array = host_owner.const_array(iterator);
        for (amrex::BoxIterator face_it(iterator.validbox()); face_it.ok(); ++face_it)
        {
            auto const& face = face_it();
            if (owner_array(face, 0) == 0)
            {
                continue;
            }
            double const energy = static_cast<double>(leakage(face, 0));
            if (!geometry.isPeriodic(mesh_direction) &&
                (face[mesh_direction] == geometry.Domain().smallEnd(mesh_direction) ||
                 face[mesh_direction] == geometry.Domain().bigEnd(mesh_direction)+1)) {
                // A physical reflecting wall cannot emit a leakage photon. Do
                // not consult unspecified physical mask/velocity ghosts.
                if (energy != 0.0) { valid = false; break; }
                continue;
            }
            // Both indices are deliberately unwrapped.  At a periodic endpoint,
            // the out-of-domain ghost contains its lower-domain neighbor.
            auto const left = face - face_type;
            auto const right = face;
            int const left_material = mask(left, 0);
            int const right_material = mask(right, 0);
            bool const interface = (left_material == kD && right_material == kK) ||
                                   (left_material == kK && right_material == kD);
            if ((left_material != kD && left_material != kK) ||
                (right_material != kD && right_material != kK) ||
                !finiteNonnegative(energy) || (!interface && energy != 0.0))
            {
                valid = false;
                break;
            }
            if (energy == 0.0)
            {
                continue;
            }

            bool const k_on_right = right_material == kK;
            int const sign = k_on_right ? 1 : -1;
            std::array<amrex::ParticleReal, 3> birth{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int const physical = AMREX_SPACEDIM == 1 ? 2 :
                    (AMREX_SPACEDIM == 2 && d == 1 ? 2 : d);
                auto coordinate = geometry.ProbLo(d) +
                    (static_cast<amrex::Real>(face[d]-geometry.Domain().smallEnd(d)) +
                     (d == mesh_direction ? 0.0 : 0.5))*geometry.CellSize(d);
                if (d == mesh_direction) {
                    if (face[d] == geometry.Domain().smallEnd(d) && sign < 0) {
                        coordinate = std::nextafter(geometry.ProbHi(d), geometry.ProbLo(d));
                    } else if (face[d] == geometry.Domain().bigEnd(d)+1 && sign > 0) {
                        coordinate = std::nextafter(geometry.ProbLo(d), geometry.ProbHi(d));
                    } else {
                        coordinate = std::nextafter(coordinate, coordinate + sign);
                    }
                }
                birth[physical] = static_cast<amrex::ParticleReal>(coordinate);
                if (d == mesh_direction && geometry.isPeriodic(d)) {
                    // GPU Redistribute always enforces this conservative
                    // native interval. A one-ULP inward endpoint birth may
                    // otherwise wrap even though its ordinary cell is valid.
                    birth[physical] = std::clamp(birth[physical],
                        geometry.ProbLoArrayInParticleReal()[d],
                        geometry.ProbHiArrayInParticleReal()[d]);
                }
            }
#if defined(WARPX_DIM_RZ)
            RZMomentMetric const metric(geometry);
            if (rz && mesh_direction == 1) {
                // Uniform axial intensity integrates over r dr. A single
                // birth at sqrt(<r^2>) preserves the leading rigid-rotation
                // angular flux, proportional to Omega*<r^2>, whereas a birth
                // at <r> would substitute <r>^2 and lose angular momentum.
                auto const lower = metric.Radius(face[0]);
                auto const upper = metric.Radius(face[0]+1);
                birth[0] = static_cast<amrex::ParticleReal>(
                    std::sqrt((lower*lower+upper*upper)/2));
            }
            if (rz) {
                // A physical nextafter at z=0 can disappear when the native
                // locator subtracts a nonzero ProbLo. Resolve the K-side in
                // that same cell-index arithmetic, including periodic aliases.
                int target = face[mesh_direction]-(sign < 0 ? 1 : 0);
                int const lo = geometry.Domain().smallEnd(mesh_direction);
                int const hi = geometry.Domain().bigEnd(mesh_direction);
                if (target < lo) { target = hi; }
                if (target > hi) { target = lo; }
                auto& coordinate = birth[physical_axis];
                auto located = [&]() {
                    auto const stored = mesh_direction == 0 ?
                        std::sqrt(coordinate*coordinate) : coordinate;
                    return lo+static_cast<int>(std::floor(
                        (stored-geometry.ProbLo(mesh_direction))*
                        geometry.InvCellSize(mesh_direction)));
                };
                if (located() != target) {
                    amrex::Real const boundary = target-lo+(sign < 0 ? 1 : 0);
                    auto const inside = std::nextafter(boundary, boundary+sign);
                    coordinate = static_cast<amrex::ParticleReal>(
                        geometry.ProbLo(mesh_direction)+inside*geometry.CellSize(mesh_direction));
                    for (int attempt = 0; attempt < 8 && located() != target; ++attempt) {
                        coordinate = std::nextafter(coordinate,
                            sign > 0 ? std::numeric_limits<amrex::ParticleReal>::infinity() :
                                       -std::numeric_limits<amrex::ParticleReal>::infinity());
                    }
                    if (located() != target) { valid = false; break; }
                }
            }
#endif
            // Native RZ particle insertion stores sqrt(x*x+y*y), not x.
            auto const native_radius = static_cast<amrex::ParticleReal>(
                std::sqrt(birth[0]*birth[0]+birth[1]*birth[1]));
            long double const speed =
                static_cast<long double>(photon_energy) / (PhysConst::m_e * PhysConst::c);
            if (!finitePositive(static_cast<double>(speed)))
            {
                valid = false;
                break;
            }

            MovingLeakageAngularQuadrature moving;
            if (work_velocity) {
                auto const v = host_velocity.const_array(iterator);
                auto const diffusion_cell = k_on_right ? left : right;
                std::array<double,3> beta{v(diffusion_cell,0)/PhysConst::c,
                    v(diffusion_cell,1)/PhysConst::c, v(diffusion_cell,2)/PhysConst::c};
#if defined(WARPX_DIM_RZ)
                if (rz) {
                    beta[1] *= native_radius/metric.Angular(diffusion_cell[0]).mean_radius;
                }
#endif
                // Keep the old arithmetic and ray storage exactly at zero velocity.
                if (beta[0] != 0 || beta[1] != 0 || beta[2] != 0) {
                    moving = ComputeMovingLeakageAngularQuadrature(beta, sign, physical_axis);
                    if (!moving.valid) { valid = false; break; }
                }
            }
            constexpr long double sqrt6 =
                2.4494897427831780981972840747058913919659474806567L;
            constexpr std::array<long double, 2> mus = {
                (6.0L - sqrt6) / 10.0L, (6.0L + sqrt6) / 10.0L};
            constexpr std::array<long double, 2> weights = {
                (9.0L - sqrt6) / 18.0L, (9.0L + sqrt6) / 18.0L};
            for (int node = 0; node < 2; ++node)
            {
                long double const transverse = std::sqrt(1.0L - mus[node] * mus[node]);
                for (int azimuth = 0; azimuth < 4; ++azimuth)
                {
                    long double const ray_energy = moving.valid ?
                        energy*moving.fraction[4*node+azimuth] : energy * weights[node] / 4.0L;
                    long double const ray_weight =
                        ray_energy / static_cast<long double>(photon_energy);
                    if (!(ray_energy > 0.0L) || !(ray_weight > 0.0L))
                    {
                        valid = false;
                        break;
                    }
                    long double const nx = transverse * (azimuth == 0 ? 1.0L :
                                                          (azimuth == 2 ? -1.0L : 0.0L));
                    long double const ny = transverse * (azimuth == 1 ? 1.0L :
                                                          (azimuth == 3 ? -1.0L : 0.0L));
                    long double const nz = mus[node] * sign;
                    std::array<long double, 3> direction{nx, ny, nz};
                    if (physical_axis != 2) { std::swap(direction[physical_axis], direction[2]); }
                    amrex::ParticleReal const stored_z = birth[2];
                    amrex::ParticleReal const stored_ux =
                        static_cast<amrex::ParticleReal>(speed * direction[0]);
                    amrex::ParticleReal const stored_uy =
                        static_cast<amrex::ParticleReal>(speed * direction[1]);
                    amrex::ParticleReal const stored_uz =
                        static_cast<amrex::ParticleReal>(speed * direction[2]);
                    amrex::ParticleReal const stored_weight =
                        static_cast<amrex::ParticleReal>(ray_weight);
                    if (!std::isfinite(birth[0]) || !std::isfinite(birth[1]) ||
                        !std::isfinite(stored_z) || !std::isfinite(stored_ux) ||
                        !std::isfinite(stored_uy) || !std::isfinite(stored_uz) ||
                        !std::isfinite(stored_weight) || stored_weight <= 0.0_prt)
                    {
                        valid = false;
                        break;
                    }
                    if (key_context) {
                        std::uint64_t key = 0;
                        if (!MakePhotonBirthDrawKey(geometry, face, mesh_direction,
                                sign, 4*node+azimuth, *key_context, key)) {
                            valid = false;
                            break;
                        }
                        leakage_plan.logical_draw_keys.push_back(key);
                    }
                    leakage_plan.x.push_back(birth[0]);
                    leakage_plan.y.push_back(birth[1]);
                    leakage_plan.z.push_back(stored_z);
                    leakage_plan.ux.push_back(stored_ux);
                    leakage_plan.uy.push_back(stored_uy);
                    leakage_plan.uz.push_back(stored_uz);
                    leakage_plan.weight.push_back(stored_weight);

                    auto const actual = storedInventory(stored_weight, stored_ux, stored_uy, stored_uz);
                    for (int n = 0; n < 4; ++n) {
                        leakage_plan.inventory[n] += actual[n];
                        if (face_inventory) { host_inventory.array(iterator)(face,n) += actual[n]; }
                    }
                    if (rz && face_inventory) {
                        long double const angular = static_cast<long double>(stored_weight)*
                            PhysConst::m_e*PhysConst::c*native_radius*stored_uy;
                        host_inventory.array(iterator)(face,4) += static_cast<amrex::Real>(angular);
                    }
                }
                if (!valid)
                {
                    break;
                }
            }
            if (!valid)
            {
                break;
            }
        }
    }

    if (!collectiveAnd(valid))
    {
        clearPlan(leakage_plan);
        return false;
    }
    amrex::ParallelDescriptor::ReduceRealSum(leakage_plan.inventory.data(), 4);
    if (face_inventory) {
        amrex::htod_memcpy(*face_inventory, host_inventory);
        amrex::Gpu::streamSynchronize();
        face_inventory->OverrideSync(geometry.periodicity());
    }
    valid = !face_inventory || face_inventory->is_finite();
    for (auto value : leakage_plan.inventory) { valid = valid && std::isfinite(value); }
    if (!collectiveAnd(valid)) { clearPlan(leakage_plan); return false; }
    ledger.accepted = true;
    return true;
}
} // namespace

bool
BuildPacketDiffusionLeakageEmissionPlan (
    amrex::MultiFab const& leakage, amrex::iMultiFab const& mask,
    amrex::Geometry const& geometry, amrex::Real photon_energy, EmissionPlan& plan,
    amrex::MultiFab* inventory, amrex::MultiFab const* velocity, int direction,
    LogicalPhotonBirthKeyContext const* key_context)
{
    return BuildLeakageEmissionPlan(leakage, mask, geometry, photon_energy, plan,
                                   inventory, velocity, direction, false, key_context);
}

bool
BuildRZPacketDiffusionLeakageEmissionPlan (
    amrex::MultiFab const& leakage, amrex::iMultiFab const& mask,
    amrex::Geometry const& geometry, amrex::Real photon_energy, EmissionPlan& plan,
    amrex::MultiFab* inventory, amrex::MultiFab const* velocity, int direction,
    LogicalPhotonBirthKeyContext const* key_context)
{
    return BuildLeakageEmissionPlan(leakage, mask, geometry, photon_energy, plan,
                                   inventory, velocity, direction, true, key_context);
}

} // namespace warpx::radiation

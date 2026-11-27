/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CollidedDiffusionAdmission.H"

#include "Particles/Pusher/GetAndSetPosition.H"
#include "PhotonFourMomentum.H"
#include "RZMomentGeometry.H"
#include "Utils/TextMsg.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace warpx::radiation
{
namespace
{
using Base = WarpXParticleContainer::Base;
constexpr std::array<char const*,3> names{"radiation_diffusion_census_ux0",
    "radiation_diffusion_census_uy0", "radiation_diffusion_census_uz0"};

bool All (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

CollidedDiffusionAdmissionReport Fail (char const* reason)
{
    CollidedDiffusionAdmissionReport report;
    report.reason = reason;
    return report;
}

bool Schema (Base const& photons, CollidedDiffusionAdmissionComponents const& components)
{
    bool const cartesian =
#if defined(WARPX_DIM_1D_Z)
        AMREX_SPACEDIM == 1
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        AMREX_SPACEDIM == 2
#elif defined(WARPX_DIM_3D)
        AMREX_SPACEDIM == 3
#else
        false
#endif
        ;
    bool valid = photons.finestLevel() == 0 && cartesian &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53;
    auto const& registered = photons.GetRealSoANames();
    for (int d = 0; d < 3; ++d) {
        int lo = components.initial_u[d], hi = lo;
        amrex::ParallelDescriptor::ReduceIntMin(lo);
        amrex::ParallelDescriptor::ReduceIntMax(hi);
        valid = valid && lo == hi && lo >= PIdx::nattribs &&
            lo < static_cast<int>(registered.size());
        if (lo >= 0 && lo < static_cast<int>(registered.size())) {
            valid = valid && registered[lo] == names[d];
        }
    }
    return All(valid);
}

struct CartesianMesh
{
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> lower{}, upper{}, inverse_cell_size{};
    amrex::IntVect first, last;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
CartesianCellForGeometry (CartesianMesh const& geometry,
    amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z,
    amrex::GpuArray<int,3>& cell)
{
    if (!amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
        !amrex::Math::isfinite(z)) { return false; }
    amrex::ParticleReal coordinates[3]{x,y,z};
#if defined(WARPX_DIM_RZ)
    coordinates[0] = std::sqrt(x*x+y*y);
#endif
#if defined(WARPX_DIM_1D_Z)
    int const physical_to_mesh[3] = {2,-1,-1};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    int const physical_to_mesh[3] = {0,2,-1};
#elif defined(WARPX_DIM_3D)
    int const physical_to_mesh[3] = {0,1,2};
#else
    int const physical_to_mesh[3] = {-1,-1,-1};
#endif
    cell = {0,0,0};
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        int const physical = physical_to_mesh[axis];
        if (physical < 0) { return false; }
        amrex::ParticleReal const coordinate = coordinates[physical];
        amrex::Real const lower = geometry.lower[axis];
        amrex::Real const upper = geometry.upper[axis];
        amrex::Real const inverse_cell_size = geometry.inverse_cell_size[axis];
        if (coordinate < lower || coordinate >= upper ||
            !amrex::Math::isfinite(inverse_cell_size) || inverse_cell_size <= 0) {
            return false;
        }
        amrex::Real const offset = std::floor(
            static_cast<amrex::Real>(coordinate-lower)*inverse_cell_size);
        amrex::Real const index = offset + static_cast<amrex::Real>(geometry.first[axis]);
        if (!amrex::Math::isfinite(index) || index < geometry.first[axis] ||
            index > geometry.last[axis]) { return false; }
        cell[axis] = static_cast<int>(index);
    }
    return true;
}

amrex::GpuArray<amrex::ParticleReal*,3> ScratchPointers (
    WarpXParIter& it, CollidedDiffusionAdmissionComponents const& components)
{
    amrex::GpuArray<amrex::ParticleReal*,3> scratch{};
    for (int d=0; d<3; ++d) {
        scratch[d]=it.GetStructOfArrays().GetRealData(components.initial_u[d]).dataPtr();
    }
    return scratch;
}

bool Overlaps (amrex::MultiFab const& a, amrex::MultiFab const& b)
{
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto const& af = a[mfi]; auto const& bf = b[mfi];
        auto const ap = reinterpret_cast<std::uintptr_t>(af.dataPtr());
        auto const bp = reinterpret_cast<std::uintptr_t>(bf.dataPtr());
        auto const as = static_cast<std::uintptr_t>(af.box().numPts())*af.nComp()*sizeof(amrex::Real);
        auto const bs = static_cast<std::uintptr_t>(bf.box().numPts())*bf.nComp()*sizeof(amrex::Real);
        auto const maximum = std::numeric_limits<std::uintptr_t>::max();
        if (!ap || !bp || as > maximum-ap || bs > maximum-bp ||
            (ap < bp+bs && bp < ap+as)) { return true; }
    }
    return false;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
PhysicalBackground (amrex::Real e, amrex::Real x, amrex::Real y, amrex::Real z)
{
    if (!amrex::Math::isfinite(e) || e < 0 || !amrex::Math::isfinite(x) ||
        !amrex::Math::isfinite(y) || !amrex::Math::isfinite(z)) { return false; }
    if (e == 0) { return x == 0 && y == 0 && z == 0; }
    amrex::Real const norm = std::sqrt((x/e)*(x/e)+(y/e)*(y/e)+(z/e)*(z/e));
    return amrex::Math::isfinite(norm) &&
        norm <= 1+32*std::numeric_limits<amrex::Real>::epsilon();
}
}

CollidedDiffusionAdmissionComponents
RegisterCollidedDiffusionAdmissionState (WarpXParticleContainer& species)
{
    auto const& existing = species.GetRealSoANames();
    bool clear = species.AmIA<PhysicalSpecies::photon>();
    for (auto const* name : names) {
        clear = clear && std::find(existing.begin(), existing.end(), name) == existing.end();
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(All(clear),
        "Census registration requires photon species and unregistered census attributes");
    CollidedDiffusionAdmissionComponents components;
    for (int d = 0; d < 3; ++d) {
        species.AddRealComp(names[d], 1);
        components.initial_u[d] = species.GetRealCompIndex(names[d]);
    }
    return components;
}

CollidedDiffusionAdmissionReport SnapshotCollidedDiffusionAdmission (
    Base& photons, CollidedDiffusionAdmissionComponents const& components)
{
    if (!Schema(photons, components)) { return Fail("Invalid census schema, level, or precision"); }
    amrex::Gpu::DeviceScalar<int> invalid(0);
    amrex::Gpu::DeviceScalar<amrex::Long> count(0);
    auto* bad = invalid.dataPtr(); auto* live = count.dataPtr();
    for (WarpXParIter it(photons,0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const position = GetParticlePosition<PIdx>(it);
        amrex::For(it.numParticles(), [=] AMREX_GPU_DEVICE(long n) {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return; }
            auto const packet = PacketConversionFourMomentumFromPhoton(p.m_rdata[PIdx::w][n],
                p.m_rdata[PIdx::ux][n], p.m_rdata[PIdx::uy][n], p.m_rdata[PIdx::uz][n]);
            amrex::ParticleReal x=0,y=0,z=0; position(n,x,y,z);
            if (!packet.valid || !amrex::Math::isfinite(x) || !amrex::Math::isfinite(y) ||
                !amrex::Math::isfinite(z)) { amrex::HostDevice::Atomic::Add(bad,1); return; }
            amrex::HostDevice::Atomic::Add(live,amrex::Long(1));
        });
    }
    if (!All(invalid.dataValue() == 0)) { return Fail("Invalid photon before census snapshot"); }
    for (WarpXParIter it(photons,0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const scratch = ScratchPointers(it,components);
        amrex::For(it.numParticles(), [=] AMREX_GPU_DEVICE(long n) {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return; }
            scratch[0][n] = p.m_rdata[PIdx::ux][n];
            scratch[1][n] = p.m_rdata[PIdx::uy][n];
            scratch[2][n] = p.m_rdata[PIdx::uz][n];
        });
    }
    CollidedDiffusionAdmissionReport report;
    report.retained = count.dataValue();
    amrex::ParallelDescriptor::ReduceLongSum(report.retained);
    amrex::Gpu::streamSynchronize();
    report.valid = true;
    return report;
}

CollidedDiffusionAdmissionReport TryAdmitCollidedPhotons (
    Base& photons, CollidedDiffusionAdmissionComponents const& components,
    amrex::MultiFab& energy, amrex::MultiFab& momentum, amrex::iMultiFab const& bulk,
    amrex::Geometry const& geometry, CollidedDiffusionAdmissionControls const& controls)
{
    if (!Schema(photons, components)) { return Fail("Invalid census schema, level, or precision"); }
    amrex::Real lower[2]{controls.max_flux_fraction,controls.max_pressure_anisotropy};
    amrex::Real upper[2]{lower[0],lower[1]};
    bool valid = std::isfinite(lower[0]) && lower[0] >= 0 && lower[0] <= amrex::Real(1)/3 &&
        std::isfinite(lower[1]) && lower[1] > 0 && lower[1] <= amrex::Real(0.1);
    if (!All(valid)) { return Fail("Invalid admission controls"); }
    amrex::ParallelDescriptor::ReduceRealMin(lower,2);
    amrex::ParallelDescriptor::ReduceRealMax(upper,2);
    if (!All(lower[0] == upper[0] && lower[1] == upper[1])) {
        return Fail("Admission controls differ between ranks");
    }
    auto const& boxes = energy.boxArray(); auto const& dm = energy.DistributionMap();
    auto const& particle_geometry = photons.Geom(0);
    bool const supported_geometry =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        !geometry.isPeriodic(0) &&
        geometry.ProbLo(0) >= 0;
#else
        geometry.Coord() == 0;
#endif
    valid = supported_geometry && geometry.Domain().ok() &&
        particle_geometry.Coord() == geometry.Coord() &&
        particle_geometry.Domain() == geometry.Domain();
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        valid = valid && geometry.Domain().length(axis) > 0 &&
            particle_geometry.isPeriodic(axis) == geometry.isPeriodic(axis) &&
            std::isfinite(geometry.ProbLo(axis)) &&
            std::isfinite(geometry.ProbHi(axis)) &&
            geometry.ProbHi(axis) > geometry.ProbLo(axis) &&
            std::isfinite(geometry.CellSize(axis)) && geometry.CellSize(axis) > 0 &&
            std::isfinite(geometry.InvCellSize(axis)) && geometry.InvCellSize(axis) > 0 &&
            particle_geometry.ProbLo(axis) == geometry.ProbLo(axis) &&
            particle_geometry.ProbHi(axis) == geometry.ProbHi(axis) &&
            particle_geometry.CellSize(axis) == geometry.CellSize(axis);
    }
    valid = valid && boxes.ok() && boxes.isDisjoint() &&
        boxes.contains(geometry.Domain()) && boxes.numPts() == geometry.Domain().numPts() &&
        energy.nComp() == 1 && energy.ixType().cellCentered() && momentum.nComp() == 3 &&
        momentum.ixType().cellCentered() && momentum.boxArray() == boxes &&
        momentum.DistributionMap() == dm && bulk.nComp() == 1 && bulk.ixType().cellCentered() &&
        bulk.boxArray() == boxes && bulk.DistributionMap() == dm &&
        photons.ParticleBoxArray(0) == boxes && photons.ParticleDistributionMap(0) == dm;
    if (!All(valid)) { return Fail("Invalid admission geometry or layout"); }
    constexpr int geometry_count = 5*AMREX_SPACEDIM;
    std::array<amrex::Real,geometry_count> geometry_min{}, geometry_max{};
    CartesianMesh mesh;
    mesh.first = geometry.Domain().smallEnd();
    mesh.last = geometry.Domain().bigEnd();
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        mesh.lower[axis] = geometry.ProbLo(axis);
        mesh.upper[axis] = geometry.ProbHi(axis);
        mesh.inverse_cell_size[axis] = geometry.InvCellSize(axis);
        int const base = 5*axis;
        geometry_min[base+0] = geometry.ProbLo(axis);
        geometry_min[base+1] = geometry.ProbHi(axis);
        geometry_min[base+2] = geometry.CellSize(axis);
        geometry_min[base+3] = static_cast<amrex::Real>(geometry.Domain().smallEnd(axis));
        geometry_min[base+4] = static_cast<amrex::Real>(geometry.Domain().bigEnd(axis));
    }
    geometry_max = geometry_min;
    amrex::ParallelDescriptor::ReduceRealMin(geometry_min.data(),geometry_count);
    amrex::ParallelDescriptor::ReduceRealMax(geometry_max.data(),geometry_count);
    for (int d=0; d<geometry_count; ++d) {
        if (geometry_min[d] != geometry_max[d]) { return Fail("Admission geometries differ"); }
    }
    if (!All(!Overlaps(energy,momentum))) { return Fail("Admission fields overlap storage"); }

    amrex::MultiFab cohort(boxes,dm,10,0); cohort.setVal(0);
    amrex::Gpu::DeviceScalar<int> invalid(0);
    amrex::Gpu::DeviceScalar<amrex::Long> live_count(0), eligible_count(0);
    auto* bad = invalid.dataPtr(); auto* live = live_count.dataPtr();
    auto* eligible = eligible_count.dataPtr();
#if defined(WARPX_DIM_RZ)
    RZMomentMetric const metric(geometry);
#endif
    for (WarpXParIter it(photons,0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const scratch = ScratchPointers(it,components);
        auto const position = GetParticlePosition<PIdx>(it);
        auto const owner_box = it.tilebox();
        auto const owner_lo = owner_box.smallEnd();
        auto const owner_hi = owner_box.bigEnd();
        auto const deposited = cohort.array(it); auto const mask = bulk.const_array(it);
        amrex::For(it.numParticles(), [=] AMREX_GPU_DEVICE(long n) {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return; }
            amrex::ParticleReal x=0,y=0,z=0; position(n,x,y,z);
            auto const packet = PacketConversionFourMomentumFromPhoton(p.m_rdata[PIdx::w][n],
                p.m_rdata[PIdx::ux][n],p.m_rdata[PIdx::uy][n],p.m_rdata[PIdx::uz][n]);
            amrex::GpuArray<int,3> cell{};
            bool finite = packet.valid && CartesianCellForGeometry(mesh,x,y,z,cell);
            for (int d=0; d<3; ++d) {
                finite = finite && amrex::Math::isfinite(scratch[d][n]);
            }
            if (!finite) { amrex::HostDevice::Atomic::Add(bad,1); return; }
            bool owned = true;
            for (int d=0; d<AMREX_SPACEDIM; ++d) {
                owned = owned && cell[d] >= owner_lo[d] && cell[d] <= owner_hi[d];
            }
            if (!owned) { amrex::HostDevice::Atomic::Add(bad,1); return; }
            int const i = cell[0], j = cell[1], k = cell[2];
            amrex::HostDevice::Atomic::Add(live,amrex::Long(1));
            bool const changed = p.m_rdata[PIdx::ux][n] != scratch[0][n] ||
                p.m_rdata[PIdx::uy][n] != scratch[1][n] ||
                p.m_rdata[PIdx::uz][n] != scratch[2][n];
            if (!changed || mask(i,j,k) != 1) { return; }
            amrex::HostDevice::Atomic::Add(eligible,amrex::Long(1));
            auto transferred = packet.value;
#if defined(WARPX_DIM_RZ)
            auto const theta = p.m_rdata[PIdx::theta][n];
            auto const ct = std::cos(theta), st = std::sin(theta);
            transferred[1] = ct*packet.value[1]+st*packet.value[2];
            auto const physical_theta = -st*packet.value[1]+ct*packet.value[2];
            transferred[2] = (x*packet.value[2]-y*packet.value[1])/
                metric.Angular(i).mean_radius;
#endif
            for (int d=0; d<4; ++d) {
                amrex::HostDevice::Atomic::Add(&deposited(i,j,k,d),transferred[d]);
            }
            amrex::Real const nx=transferred[1]/packet.value[0];
#if defined(WARPX_DIM_RZ)
            amrex::Real const ny=physical_theta/packet.value[0];
#else
            amrex::Real const ny=transferred[2]/packet.value[0];
#endif
            amrex::Real const nz=packet.value[3]/packet.value[0];
            amrex::Real const moments[6]{nx*nx,ny*ny,nz*nz,nx*ny,nx*nz,ny*nz};
            for (int d=0; d<6; ++d) {
                amrex::HostDevice::Atomic::Add(&deposited(i,j,k,d+4),
                    packet.value[0]*moments[d]);
            }
        });
    }
    if (!All(invalid.dataValue() == 0)) { return Fail("Invalid or unowned census photon"); }
    amrex::MultiFab next_e(boxes,dm,1,0), next_q(boxes,dm,3,0);
    amrex::iMultiFab accepted(boxes,dm,1,0); accepted.setVal(0);
    CollidedDiffusionAdmissionReport report;
    report.admitted_four_inventory.define(boxes,dm,4,0);
    report.admitted_four_inventory.setVal(0);
    amrex::ReduceOps<amrex::ReduceOpMax> ops; amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(energy); it.isValid(); ++it) {
        auto const e=energy.const_array(it), q=momentum.const_array(it), a=cohort.const_array(it);
        auto const mask=bulk.const_array(it); auto const accept=accepted.array(it);
        auto const en=next_e.array(it), qn=next_q.array(it);
        auto const transferred=report.admitted_four_inventory.array(it);
        ops.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->amrex::GpuTuple<int> {
            bool bad_cell = (mask(i,j,k) != 0 && mask(i,j,k) != 1) ||
                !PhysicalBackground(e(i,j,k),q(i,j,k,0),q(i,j,k,1),q(i,j,k,2));
            for (int d=0; d<10; ++d) { bad_cell = bad_cell || !amrex::Math::isfinite(a(i,j,k,d)); }
            if (bad_cell) { return {1}; }
            CollidedDiffusionCohortMoments m; m.background_energy=e(i,j,k); m.cohort_energy=a(i,j,k,0);
            for (int d=0; d<3; ++d) { m.background_momentum[d]=q(i,j,k,d); m.cohort_momentum[d]=a(i,j,k,d+1); }
            for (int d=0; d<6; ++d) { m.cohort_pressure[d]=a(i,j,k,d+4); }
            bool const take = mask(i,j,k)==1 && ShouldAdmitCollidedDiffusionCohort(m,controls);
            en(i,j,k)=e(i,j,k)+(take ? a(i,j,k,0) : 0);
            for (int d=0; d<3; ++d) { qn(i,j,k,d)=q(i,j,k,d)+(take ? a(i,j,k,d+1) : 0); }
            if (!PhysicalBackground(en(i,j,k),qn(i,j,k,0),qn(i,j,k,1),qn(i,j,k,2))) { return {1}; }
            accept(i,j,k)=take ? 1 : 0;
            for (int d=0; d<4; ++d) { transferred(i,j,k,d)=take ? a(i,j,k,d) : 0; }
            return {0};
        });
    }
    int local_bad=amrex::get<0>(data.value());
    if (energy.local_size()==0) { local_bad=0; }
    if (!All(local_bad==0)) { return Fail("Invalid background, mask, or cohort reduction"); }
    for (int d=0; d<4; ++d) {
        report.actual_admitted_four_inventory[d]=report.admitted_four_inventory.sum(d);
        if (!std::isfinite(report.actual_admitted_four_inventory[d])) {
            return Fail("Nonfinite global admission inventory");
        }
    }
    report.retained=live_count.dataValue(); report.eligible=eligible_count.dataValue();
    amrex::ParallelDescriptor::ReduceLongSum(report.retained);
    amrex::ParallelDescriptor::ReduceLongSum(report.eligible);
    // No recoverable numerical rejection after this private commit boundary.
    amrex::MultiFab::Copy(energy,next_e,0,0,1,0);
    amrex::MultiFab::Copy(momentum,next_q,0,0,3,0);
    amrex::Gpu::DeviceScalar<amrex::Long> removed(0); auto* count=removed.dataPtr();
    for (WarpXParIter it(photons,0); it.isValid(); ++it) {
        auto const p=it.GetParticleTile().getParticleTileData();
        auto const scratch=ScratchPointers(it,components);
        auto const position=GetParticlePosition<PIdx>(it); auto const accept=accepted.const_array(it);
        auto const owner_box = it.tilebox();
        auto const owner_lo=owner_box.smallEnd(), owner_hi=owner_box.bigEnd();
        amrex::For(it.numParticles(),[=] AMREX_GPU_DEVICE(long n) {
            if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid()) { return; }
            bool const changed=p.m_rdata[PIdx::ux][n]!=scratch[0][n] ||
                p.m_rdata[PIdx::uy][n]!=scratch[1][n] ||
                p.m_rdata[PIdx::uz][n]!=scratch[2][n];
            if (!changed) { return; }
            amrex::ParticleReal x=0,y=0,z=0; position(n,x,y,z);
            amrex::GpuArray<int,3> cell{};
            if (!CartesianCellForGeometry(mesh,x,y,z,cell)) { return; }
            bool owned = true;
            for (int d=0; d<AMREX_SPACEDIM; ++d) {
                owned = owned && cell[d] >= owner_lo[d] && cell[d] <= owner_hi[d];
            }
            if (!owned || accept(cell[0],cell[1],cell[2])==0) { return; }
            p.m_idcpu[n]=amrex::ParticleIdCpus::Invalid; p.m_rdata[PIdx::w][n]=0;
            amrex::HostDevice::Atomic::Add(count,amrex::Long(1));
        });
    }
    report.admitted=removed.dataValue();
    amrex::ParallelDescriptor::ReduceLongSum(report.admitted);
    report.retained-=report.admitted;
    for (auto& entry : photons.GetParticles(0)) { amrex::removeInvalidParticles(entry.second); }
    photons.Redistribute();
    report.valid=true;
    return report;
}
}

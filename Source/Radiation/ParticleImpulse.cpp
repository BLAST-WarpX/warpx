/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleImpulse.H"

#include "MaterialKineticWork.H"
#include "ParticleCellShape.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace warpx::radiation
{
    namespace
    {
        AMREX_GPU_HOST_DEVICE
        amrex::GpuArray<int, 3> FoldReflectingCell (
            amrex::GpuArray<int, 3> cell, bool reflecting, int lo, int hi) noexcept
        {
            // Even scalar cell extension: -1 -> 0 at a nodal material wall.
            // Radiation assignment uses this same map for every force component,
            // not the component-dependent PEC electric-field parity.
            if (reflecting) {
                if (cell[0] < lo) { cell[0] = 2 * lo - 1 - cell[0]; }
                else if (cell[0] > hi) { cell[0] = 2 * hi + 1 - cell[0]; }
            }
            return cell;
        }

        std::array<std::string, 4>
        AttributeNames (std::string const& path)
        {
            std::string const prefix = "radiation_impulse_" + path;
            return {prefix + "_ux", prefix + "_uy", prefix + "_uz", prefix + "_work"};
        }
    }

    void
    RegisterParticleImpulseState (WarpXParticleContainer& species, std::string const& path)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!path.empty(), "Radiation impulse path must be named.");
        auto const names = AttributeNames(path);
        for (auto const& name : names)
        {
            auto const& existing = species.GetRealSoANames();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::find(existing.begin(), existing.end(), name) == existing.end(),
                "Duplicate/reserved radiation impulse particle attribute: " + name);
            species.AddRealComp(name, /*comm=*/1);
        }
    }

    amrex::GpuArray<amrex::Real, 4>
    ParticleImpulseInventory (MultiParticleContainer& particles,
                              std::vector<std::string> const& species_names,
                              std::string const& path)
    {
        auto const attributes = AttributeNames(path);
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (auto const& name : species_names)
        {
            auto& species = particles.GetParticleContainerFromName(name);
            auto const mass = static_cast<amrex::Real>(species.getMass());
            amrex::GpuArray<int, 4> components{};
            for (int d = 0; d < 4; ++d) { components[d] = species.GetRealCompIndex(attributes[d]); }
            for (WarpXParIter iterator(species, 0); iterator.isValid(); ++iterator)
            {
                auto const* weight = iterator.GetStructOfArrays().GetRealData(PIdx::w).dataPtr();
                amrex::GpuArray<amrex::ParticleReal const*, 4> carry{};
                for (int d = 0; d < 4; ++d)
                {
                    carry[d] = iterator.GetStructOfArrays().GetRealData(components[d]).dataPtr();
                }
                ops.eval(iterator.numParticles(), data, [=] AMREX_GPU_DEVICE (long ip) -> Tuple
                {
                    auto const weighted_mass = weight[ip] * mass;
                    return {weighted_mass * carry[0][ip], weighted_mass * carry[1][ip],
                            weighted_mass * carry[2][ip], weighted_mass * carry[3][ip]};
                });
            }
        }
        auto const totals = data.value();
        amrex::GpuArray<amrex::Real, 4> result{amrex::get<0>(totals), amrex::get<1>(totals),
                                             amrex::get<2>(totals), amrex::get<3>(totals)};
        amrex::ParallelDescriptor::ReduceRealSum(result.data(), 4);
        return result;
    }

    struct ParticleImpulseMaterial::Impl
    {
        struct Species
        {
            struct Baseline
            {
                std::pair<int, int> key;
                long count = 0;
                std::vector<int> components;
                amrex::Gpu::DeviceVector<amrex::ParticleReal> values;
            };
            WarpXParticleContainer* live = nullptr;
            WarpXParticleContainer::Base* particles = nullptr;
            std::unique_ptr<WarpXParticleContainer::Base> owned;
            amrex::Real mass = 0;
            std::vector<Baseline> baseline;
        };
        std::vector<Species> species;
        bool cloned = false;
        bool committed = false;
    };

    ParticleImpulseMaterial::ParticleImpulseMaterial (
        MultiParticleContainer& particles, std::vector<std::string> const& species_names)
        : ParticleImpulseMaterial(particles, species_names, true) {}

    ParticleImpulseMaterial::ParticleImpulseMaterial (
        MultiParticleContainer& particles, std::vector<std::string> const& species_names,
        bool clone)
        : m_impl(std::make_unique<Impl>())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::GetInstance().finestLevel() == 0,
            "Private radiation material intervals require level zero.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!species_names.empty(), "No material impulse species.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::set<std::string>(species_names.begin(), species_names.end()).size() ==
                species_names.size(), "Duplicate material impulse species.");
        m_impl->cloned = clone;
        for (auto const& name : species_names) {
            Impl::Species entry;
            entry.live = &particles.GetParticleContainerFromName(name);
            entry.mass = static_cast<amrex::Real>(entry.live->getMass());
            if (clone) {
                entry.owned = std::make_unique<WarpXParticleContainer::Base>(
                    entry.live->make_alike<>());
                // make_alike copies the SoA schema, not a polymorphic arena.
                // Match WarpX's live device-capable allocation before copying.
                entry.owned->SetArena(entry.live->arena());
                entry.owned->copyParticles(*entry.live, true);
                entry.particles = entry.owned.get();
                std::vector<int> components{PIdx::ux, PIdx::uy, PIdx::uz};
                auto const& names = entry.live->GetRealSoANames();
                for (int d = 0; d < static_cast<int>(names.size()); ++d) {
                    if (names[d].starts_with("radiation_impulse_")) { components.push_back(d); }
                }
                for (auto const& [key, tile] : entry.live->GetParticles(0)) {
                    if (tile.numParticles() == 0) { continue; }
                    Impl::Species::Baseline saved;
                    saved.key = key;
                    saved.count = tile.numParticles();
                    saved.components = components;
                    saved.values.resize(saved.count * components.size());
                    for (std::size_t d = 0; d < components.size(); ++d) {
                        auto const& src = tile.GetStructOfArrays().GetRealData(components[d]);
                        amrex::Gpu::copy(amrex::Gpu::deviceToDevice, src.begin(),
                            src.begin() + saved.count, saved.values.begin() + d * saved.count);
                    }
                    entry.baseline.push_back(std::move(saved));
                }
            } else { entry.particles = entry.live; }
            m_impl->species.push_back(std::move(entry));
        }
    }

    ParticleImpulseMaterial::~ParticleImpulseMaterial () = default;

    bool
    ParticleImpulseMaterial::Commit ()
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->cloned && !m_impl->committed,
            "Only an uncommitted private material state may commit.");
        struct Copy
        {
            WarpXParticleContainer::ParticleTileType* live;
            WarpXParticleContainer::ParticleTileType* trial;
            std::vector<int> components;
        };
        std::vector<Copy> copies;
        amrex::Gpu::DeviceScalar<int> invalid(0);
        auto* invalid_ptr = invalid.dataPtr();
        int invalid_layout = 0;
        for (auto& species : m_impl->species) {
            auto const& names = species.particles->GetRealSoANames();
            if (names != species.live->GetRealSoANames() ||
                species.particles->ParticleBoxArray(0) != species.live->ParticleBoxArray(0) ||
                species.particles->ParticleDistributionMap(0) !=
                    species.live->ParticleDistributionMap(0)) {
                invalid_layout = 1;
                continue;
            }
            for (auto const& [key, live] : species.live->GetParticles(0)) {
                if (live.numParticles() == 0) { continue; }
                auto found = species.particles->GetParticles(0).find(key);
                if (found == species.particles->GetParticles(0).end() ||
                    found->second.numParticles() != live.numParticles()) {
                    invalid_layout = 1;
                }
            }
            for (auto const& baseline : species.baseline) {
                auto found = species.live->GetParticles(0).find(baseline.key);
                if (found == species.live->GetParticles(0).end() ||
                    found->second.numParticles() != baseline.count) {
                    invalid_layout = 1;
                    continue;
                }
                for (std::size_t d = 0; d < baseline.components.size(); ++d) {
                    auto const* original = baseline.values.dataPtr() + d * baseline.count;
                    auto const* current = found->second.GetStructOfArrays()
                        .GetRealData(baseline.components[d]).dataPtr();
                    amrex::For(baseline.count, [=] AMREX_GPU_DEVICE (long p) {
                        if (current[p] != original[p]) {
                            amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        }
                    });
                }
            }
            std::vector<int> components{PIdx::ux, PIdx::uy, PIdx::uz};
            for (int d = 0; d < static_cast<int>(names.size()); ++d) {
                if (names[d].starts_with("radiation_impulse_")) { components.push_back(d); }
            }
            for (auto& [key, trial] : species.particles->GetParticles(0)) {
                if (trial.numParticles() == 0) { continue; }
                auto found = species.live->GetParticles(0).find(key);
                if (found == species.live->GetParticles(0).end() ||
                    found->second.numParticles() != trial.numParticles()) {
                    invalid_layout = 1;
                    continue;
                }
                auto& live = found->second;
                auto const before = live.getParticleTileData();
                auto const after = trial.getParticleTileData();
                amrex::For(trial.numParticles(), [=] AMREX_GPU_DEVICE (long p) {
                    bool mismatch = before.m_idcpu[p] != after.m_idcpu[p]
                        || before.m_rdata[PIdx::w][p] != after.m_rdata[PIdx::w][p];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        mismatch = mismatch || before.m_rdata[d][p] != after.m_rdata[d][p];
                    }
                    if (mismatch) { amrex::HostDevice::Atomic::Add(invalid_ptr, 1); }
                });
                copies.push_back({&live, &trial, components});
            }
        }
        invalid_layout = amrex::max(invalid_layout, invalid.dataValue());
        amrex::ParallelDescriptor::ReduceIntMax(invalid_layout);
        if (invalid_layout != 0) { return false; }
        for (auto const& copy : copies) {
            for (auto component : copy.components) {
                auto const& src = copy.trial->GetStructOfArrays().GetRealData(component);
                auto& dst = copy.live->GetStructOfArrays().GetRealData(component);
                amrex::Gpu::copy(amrex::Gpu::deviceToDevice, src.begin(), src.end(), dst.begin());
            }
        }
        amrex::Gpu::streamSynchronize();
        m_impl->committed = true;
        return true;
    }

    struct ParticleImpulseTransaction::Impl
    {
        struct Tile
        {
            WarpXParticleContainer::ParticleTileType* particles = nullptr;
            amrex::GpuArray<int, 4> components{};
            amrex::Gpu::DeviceVector<MaterialImpulseCandidate> candidate;
        };
        std::vector<Tile> tiles;
        amrex::MultiFab requested_work;
        amrex::MultiFab actual_work;
        amrex::MultiFab actual_impulse;
        amrex::MultiFab energy_carry_change;
        amrex::MultiFab momentum_carry_change;
        amrex::MultiFab numerical_energy_residual;
        amrex::MultiFab work_partition_residual;
        amrex::MultiFab mass;
        amrex::MultiFab work_velocity;
        bool has_work_velocity = false;
        bool valid = false;
        bool committed = false;
    };

    ParticleImpulseTransaction::ParticleImpulseTransaction () : m_impl(std::make_unique<Impl>()) {}
    ParticleImpulseTransaction::~ParticleImpulseTransaction () = default;

    bool
    ParticleImpulseTransaction::Stage (
        MultiParticleContainer& particles, std::vector<std::string> const& species_names,
        std::string const& path, amrex::MultiFab const& cell_impulse, bool need_work_velocity,
        ParticleImpulseAssignment assignment)
    {
        ParticleImpulseMaterial material(particles, species_names, false);
        return Stage(material, path, cell_impulse, need_work_velocity, assignment);
    }

    bool
    ParticleImpulseTransaction::Stage (
        ParticleImpulseMaterial& material, std::string const& path,
        amrex::MultiFab const& cell_impulse, bool need_work_velocity,
        ParticleImpulseAssignment assignment)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            cell_impulse.nComp() == 3 && cell_impulse.ixType().cellCentered(),
            "Particle impulse staging requires a cell-centered vector.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!material.m_impl->committed,
            "Cannot stage another source on an already committed material interval.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::GetInstance().finestLevel() == 0 &&
                std::numeric_limits<amrex::Real>::digits >= 53 &&
                std::numeric_limits<amrex::ParticleReal>::digits >= 53,
            "Particle-owned radiation impulse currently requires level zero and double precision.");
#if defined(WARPX_DIM_RSPHERE)
        amrex::Abort("Particle-owned radiation impulse is not implemented in RSPHERE.");
#endif
        // Re-staging explicitly discards the previous uncommitted candidate.
        m_impl = std::make_unique<Impl>();
        bool const shaped = assignment != ParticleImpulseAssignment::NearestCell;
        int const shape_order = WarpX::nox;
        int const shape_ghosts = shaped ? (shape_order + 2) / 2 : 0;
        auto const& geometry = WarpX::GetInstance().Geom(0);
        bool const reflecting = assignment == ParticleImpulseAssignment::ReflectingNodalCellAverage;
        int const domain_lo = geometry.Domain().smallEnd(0);
        int const domain_hi = geometry.Domain().bigEnd(0);
        if (reflecting) {
#if defined(WARPX_DIM_1D_Z)
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!geometry.isPeriodic(0)
                && geometry.Coord() == 0 && geometry.Domain().length(0) >= shape_ghosts
                && WarpX::field_boundary_lo[0] == FieldBoundaryType::PEC
                && WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC
                && WarpX::particle_boundary_lo[0] == ParticleBoundaryType::Reflecting
                && WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting,
                "Reflecting nodal assignment requires 1D PEC/reflecting faces "
                "and stencil support.");
#else
            amrex::Abort("Reflecting nodal radiation assignment currently supports 1D only.");
#endif
        }
        if (shaped)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.Coord() == 0 &&
                (geometry.isAllPeriodic() || reflecting),
                "Nodal radiation force assignment requires periodic Cartesian geometry.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shape_order >= 1 && shape_order <= 4 &&
                (assignment == ParticleImpulseAssignment::NativeNodalCellAverage || reflecting ||
                 (assignment == ParticleImpulseAssignment::LinearNodalCellAverage && shape_order == 1)),
                "Nodal radiation force assignment requires matching native particle shapes 1-4; "
                "the explicit linear option requires particle_shape=1.");
        }
        auto const& ba = cell_impulse.boxArray();
        auto const& dm = cell_impulse.DistributionMap();
        for (auto* field : {&m_impl->actual_work, &m_impl->work_partition_residual,
                            &m_impl->energy_carry_change, &m_impl->numerical_energy_residual})
        {
            field->define(ba, dm, 1, 0);
            field->setVal(0);
        }
        m_impl->requested_work.define(ba, dm, 1, shape_ghosts);
        m_impl->requested_work.setVal(0);
        m_impl->actual_impulse.define(ba, dm, 3, 0);
        m_impl->actual_impulse.setVal(0);
        m_impl->momentum_carry_change.define(ba, dm, 3, 0);
        m_impl->momentum_carry_change.setVal(0);
        m_impl->mass.define(ba, dm, 1, shape_ghosts);
        auto& cell_mass = m_impl->mass;
        cell_mass.setVal(0);
        m_impl->has_work_velocity = need_work_velocity;
        if (need_work_velocity)
        {
            m_impl->work_velocity.define(ba, dm, 3, shape_ghosts);
            m_impl->work_velocity.setVal(0);
        }
        amrex::MultiFab ghost_impulse;
        if (shaped)
        {
            ghost_impulse.define(ba, dm, 3, shape_ghosts);
            amrex::MultiFab::Copy(ghost_impulse, cell_impulse, 0, 0, 3, 0);
            ghost_impulse.FillBoundary(geometry.periodicity());
        }
        auto const plo = geometry.ProbLoArray();
        auto const phi = geometry.ProbHiArray();
        auto const dxi = geometry.InvCellSizeArray();
        amrex::Gpu::DeviceScalar<int> invalid(0);
        int* const invalid_ptr = invalid.dataPtr();
        amrex::MFItInfo info;
        // No OpenMP region around staging: tiles own candidate allocations and
        // metadata. GPU scatter still uses atomic updates; CPU For is not SIMD.
        if (amrex::Gpu::notInLaunchRegion())
        {
            info.EnableTiling(WarpXParticleContainer::tile_size);
        }
        for (auto const& entry : material.m_impl->species)
        {
            auto& species = *entry.particles;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species.ParticleBoxArray(0) == ba && species.ParticleDistributionMap(0) == dm,
                "Particle impulse field must use the material particle grid layout.");
            auto const mass = entry.mass;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0, "Material rest mass must be positive.");
            for (WarpXParIter iterator(species, 0, info); iterator.isValid(); ++iterator)
            {
                auto const data = iterator.GetParticleTile().getParticleTileData();
                auto const mass_field = cell_mass.array(iterator);
                amrex::For(iterator.numParticles(), [=] AMREX_GPU_DEVICE (long ip)
                {
                    auto const p = WarpXParticleContainer::ParticleType(data, ip);
                    if (reflecting && (!amrex::Math::isfinite(p.pos(0)) ||
                        p.pos(0) < plo[0] || p.pos(0) > phi[0])) {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
                    auto owner = amrex::getParticleCell(p, plo, dxi);
                    if (reflecting) { owner[0] = amrex::min(owner[0], domain_hi); }
                    auto const [i, j, k] = owner.dim3();
                    if (shaped)
                    {
                        ParticleCellShape<AMREX_SPACEDIM> shape;
                        shape.center = {i, j, k};
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            shape.fraction[d] = (p.pos(d) - plo[d]) * dxi[d] - shape.center[d];
                        }
                        shape.Initialize(shape_order);
                        for (int s = 0; s < shape.size; ++s)
                        {
                            auto const cell = FoldReflectingCell(
                                shape.Cell(s), reflecting, domain_lo, domain_hi);
                            amrex::Gpu::Atomic::AddNoRet(&mass_field(cell[0], cell[1], cell[2]),
                                shape.Weight(s) * data.m_rdata[PIdx::w][ip] * mass);
                        }
                        return;
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &mass_field(i, j, k), data.m_rdata[PIdx::w][ip] * mass);
                });
            }
        }
        if (reflecting) {
            int invalid_positions = invalid.dataValue();
            amrex::ParallelDescriptor::ReduceIntMax(invalid_positions);
            if (invalid_positions != 0) { return false; }
        }
        if (shaped)
        {
            cell_mass.SumBoundary(geometry.periodicity());
            cell_mass.FillBoundary(geometry.periodicity());
        }
        for (amrex::MFIter iterator(cell_mass); iterator.isValid(); ++iterator)
        {
            auto const mass = cell_mass.const_array(iterator);
            auto const impulse = cell_impulse.const_array(iterator);
            amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                bool valid = mass(i, j, k) >= 0 && amrex::Math::isfinite(mass(i, j, k));
                for (int d = 0; d < 3; ++d)
                {
                    valid = valid && amrex::Math::isfinite(impulse(i, j, k, d))
                        && (impulse(i, j, k, d) == 0 || mass(i, j, k) > 0);
                }
                if (!valid) { amrex::HostDevice::Atomic::Add(invalid_ptr, 1); }
            });
        }
        int invalid_count = invalid.dataValue();
        amrex::ParallelDescriptor::ReduceIntMax(invalid_count);
        if (invalid_count != 0) { return false; }

        auto const attributes = AttributeNames(path);
        for (auto const& entry : material.m_impl->species)
        {
            auto& species = *entry.particles;
            auto const species_mass = entry.mass;
            amrex::GpuArray<int, 4> components{};
            for (int d = 0; d < 4; ++d) { components[d] = species.GetRealCompIndex(attributes[d]); }
            for (WarpXParIter iterator(species, 0, info); iterator.isValid(); ++iterator)
            {
                Impl::Tile staged;
                staged.particles = &iterator.GetParticleTile();
                staged.components = components;
                auto const np = iterator.numParticles();
                staged.candidate.resize(np);
                auto* candidate = staged.candidate.dataPtr();
                auto const data = staged.particles->getParticleTileData();
                auto const mass = cell_mass.const_array(iterator);
                auto const impulse = shaped ? ghost_impulse.const_array(iterator)
                    : cell_impulse.const_array(iterator);
                auto const requested_work = m_impl->requested_work.array(iterator);
                auto const actual_work = m_impl->actual_work.array(iterator);
                auto const actual_impulse = m_impl->actual_impulse.array(iterator);
                auto const energy_change = m_impl->energy_carry_change.array(iterator);
                auto const momentum_change = m_impl->momentum_carry_change.array(iterator);
                auto const residual = m_impl->numerical_energy_residual.array(iterator);
                auto const partition_residual = m_impl->work_partition_residual.array(iterator);
                amrex::Array4<amrex::Real> work_velocity;
                if (need_work_velocity) { work_velocity = m_impl->work_velocity.array(iterator); }
                amrex::GpuArray<amrex::ParticleReal const*, 4> carry{};
                for (int d = 0; d < 4; ++d)
                {
                    carry[d] = staged.particles->GetStructOfArrays()
                                   .GetRealData(components[d]).dataPtr();
                }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                auto const position = GetParticlePosition<PIdx>(iterator);
#endif
                amrex::For(np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    auto const p = WarpXParticleContainer::ParticleType(data, ip);
                    auto owner = amrex::getParticleCell(p, plo, dxi);
                    if (reflecting) { owner[0] = amrex::min(owner[0], domain_hi); }
                    auto const [i, j, k] = owner.dim3();
                    auto const weight_mass = data.m_rdata[PIdx::w][ip] * species_mass;
                    if (!(weight_mass > 0) || !(mass(i, j, k) > 0))
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
                    amrex::GpuArray<amrex::ParticleReal, 3> velocity{};
                    amrex::GpuArray<amrex::Real, 3> old_carry{};
                    amrex::GpuArray<amrex::Real, 3> increment{};
                    ParticleCellShape<AMREX_SPACEDIM> shape;
                    shape.center = {i, j, k};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        shape.fraction[d] = (p.pos(d) - plo[d]) * dxi[d] - shape.center[d];
                    }
                    if (shaped) { shape.Initialize(shape_order); }
                    for (int d = 0; d < 3; ++d)
                    {
                        velocity[d] = data.m_rdata[PIdx::ux + d][ip];
                        old_carry[d] = carry[d][ip];
                        if (!shaped) { increment[d] = impulse(i, j, k, d) / mass(i, j, k); }
                    }
                    if (shaped)
                    {
                        for (int s = 0; s < shape.size; ++s)
                        {
                            auto const cell = FoldReflectingCell(
                                shape.Cell(s), reflecting, domain_lo, domain_hi);
                            auto const w = shape.Weight(s);
                            if (w == 0) { continue; }
                            auto const m = mass(cell[0], cell[1], cell[2]);
                            for (int d = 0; d < 3; ++d)
                            {
                                increment[d] += w * impulse(cell[0], cell[1], cell[2], d) / m;
                            }
                        }
                    }
                    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
                    amrex::Real old_momentum_scale = PhysConst::c;
                    amrex::Real old_kinetic_scale = PhysConst::c2;
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const u = static_cast<amrex::Real>(velocity[d]);
                        old_momentum_scale = amrex::max(old_momentum_scale, std::abs(u));
                        old_kinetic_scale += u * u;
                    }
                    bool old_bounded = amrex::Math::isfinite(old_kinetic_scale)
                        && std::abs(carry[3][ip]) <= 256 * epsilon * old_kinetic_scale;
                    for (int d = 0; d < 3; ++d)
                    {
                        old_bounded = old_bounded && std::abs(old_carry[d])
                            <= 256 * epsilon * old_momentum_scale;
                    }
                    if (!old_bounded)
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    amrex::ParticleReal radius, theta, z;
                    position.AsStored(ip, radius, theta, z);
                    amrex::ignore_unused(radius, z);
                    auto const cosine = std::cos(theta);
                    auto const sine = std::sin(theta);
                    auto const radial = increment[0];
                    auto const azimuthal = increment[1];
                    increment[0] = radial * cosine - azimuthal * sine;
                    increment[1] = radial * sine + azimuthal * cosine;
#endif
                    candidate[ip] = EvaluateMaterialImpulseCandidate(
                        velocity, old_carry, carry[3][ip], increment);
                    auto const& result = candidate[ip];
                    amrex::Real momentum_scale = PhysConst::c;
                    amrex::Real kinetic_scale = PhysConst::c2;
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const old_u = static_cast<amrex::Real>(velocity[d]);
                        auto const new_u = static_cast<amrex::Real>(result.velocity[d]);
                        momentum_scale = amrex::max(momentum_scale,
                            amrex::max(std::abs(old_u), std::abs(new_u)));
                        kinetic_scale += old_u * old_u + new_u * new_u;
                    }
                    bool bounded = result.valid && amrex::Math::isfinite(kinetic_scale)
                        && std::abs(result.energy_carry) <= 256 * epsilon * kinetic_scale;
                    for (int d = 0; d < 3; ++d)
                    {
                        bounded = bounded && std::abs(result.momentum_carry[d])
                            <= 256 * epsilon * momentum_scale;
                    }
                    if (!bounded)
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
                    if (need_work_velocity || shaped)
                    {
                        amrex::Real old_squared = 0;
                        amrex::Real trial_squared = 0;
                        for (int d = 0; d < 3; ++d)
                        {
                            auto const u = static_cast<amrex::Real>(velocity[d]);
                            auto const trial = u + increment[d];
                            old_squared += u * u;
                            trial_squared += trial * trial;
                        }
                        auto const denominator = std::sqrt(1 + old_squared / PhysConst::c2)
                            + std::sqrt(1 + trial_squared / PhysConst::c2);
                        amrex::GpuArray<amrex::Real, 3> secant{};
                        for (int d = 0; d < 3; ++d)
                        {
                            secant[d] = (2 * static_cast<amrex::Real>(velocity[d]) + increment[d])
                                / denominator;
                        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                        auto const x = secant[0];
                        auto const y = secant[1];
                        secant[0] = x * cosine + y * sine;
                        secant[1] = -x * sine + y * cosine;
#endif
                        if (shaped)
                        {
                            amrex::Real partition = 0;
                            amrex::Real scale = std::abs(result.requested_work);
                            for (int s = 0; s < shape.size; ++s)
                            {
                                auto const cell = FoldReflectingCell(
                                    shape.Cell(s), reflecting, domain_lo, domain_hi);
                                auto const w = shape.Weight(s);
                                if (w == 0) { continue; }
                                auto const ratio = w / mass(cell[0], cell[1], cell[2]);
                                amrex::Real work = 0;
                                for (int d = 0; d < 3; ++d)
                                {
                                    auto const term = ratio * impulse(cell[0], cell[1], cell[2], d)
                                        * secant[d];
                                    work += term;
                                    scale += std::abs(term);
                                    if (need_work_velocity)
                                    {
                                        amrex::Gpu::Atomic::AddNoRet(
                                            &work_velocity(cell[0], cell[1], cell[2], d),
                                            weight_mass * ratio * secant[d]);
                                    }
                                }
                                partition += work;
                                amrex::Gpu::Atomic::AddNoRet(
                                    &requested_work(cell[0], cell[1], cell[2]), weight_mass * work);
                            }
                            auto const difference = partition - result.requested_work;
                            if (!amrex::Math::isfinite(scale) ||
                                std::abs(difference) > 256 * epsilon * scale)
                            {
                                amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                            }
                            amrex::Gpu::Atomic::AddNoRet(&partition_residual(i, j, k),
                                weight_mass * difference);
                        }
                        else
                        {
                            for (int d = 0; d < 3; ++d)
                            {
                                amrex::Gpu::Atomic::AddNoRet(&work_velocity(i, j, k, d),
                                    weight_mass / mass(i, j, k) * secant[d]);
                            }
                        }
                    }
                    amrex::GpuArray<amrex::Real, 3> applied{};
                    amrex::GpuArray<amrex::Real, 3> carry_change{};
                    for (int d = 0; d < 3; ++d)
                    {
                        applied[d] = static_cast<amrex::Real>(result.velocity[d])
                            - static_cast<amrex::Real>(velocity[d]);
                        carry_change[d] = result.momentum_carry[d] - old_carry[d];
                    }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    auto const x = applied[0];
                    auto const y = applied[1];
                    applied[0] = x * cosine + y * sine;
                    applied[1] = -x * sine + y * cosine;
                    auto const carry_x = carry_change[0];
                    auto const carry_y = carry_change[1];
                    carry_change[0] = carry_x * cosine + carry_y * sine;
                    carry_change[1] = -carry_x * sine + carry_y * cosine;
#endif
                    for (int d = 0; d < 3; ++d)
                    {
                        amrex::Gpu::Atomic::AddNoRet(
                            &actual_impulse(i, j, k, d), weight_mass * applied[d]);
                        amrex::Gpu::Atomic::AddNoRet(
                            &momentum_change(i, j, k, d), weight_mass * carry_change[d]);
                    }
                    if (!shaped)
                    {
                        amrex::Gpu::Atomic::AddNoRet(
                            &requested_work(i, j, k), weight_mass * result.requested_work);
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &actual_work(i, j, k), weight_mass * result.actual_work);
                    amrex::Gpu::Atomic::AddNoRet(
                        &energy_change(i, j, k),
                        weight_mass * (result.energy_carry - carry[3][ip]));
                    amrex::Gpu::Atomic::AddNoRet(&residual(i, j, k), weight_mass *
                        (result.requested_work - result.actual_work
                         - (result.energy_carry - carry[3][ip])));
                });
                m_impl->tiles.push_back(std::move(staged));
            }
        }
        if (shaped)
        {
            m_impl->requested_work.SumBoundary(geometry.periodicity());
            if (need_work_velocity) { m_impl->work_velocity.SumBoundary(geometry.periodicity()); }
        }
        // Individual finite contributions can still overflow an aggregate.
        for (auto const* field : {&m_impl->requested_work, &m_impl->actual_work,
                                  &m_impl->energy_carry_change, &m_impl->actual_impulse,
                                  &m_impl->momentum_carry_change,
                                  &m_impl->numerical_energy_residual,
                                  &m_impl->work_partition_residual})
        {
            for (amrex::MFIter iterator(*field); iterator.isValid(); ++iterator)
            {
                auto const values = field->const_array(iterator);
                amrex::For(iterator.validbox(), field->nComp(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (!amrex::Math::isfinite(values(i, j, k, n)))
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                    }
                });
            }
        }
        invalid_count = invalid.dataValue();
        if (need_work_velocity && !m_impl->work_velocity.is_finite()) { invalid_count = 1; }
        amrex::ParallelDescriptor::ReduceIntMax(invalid_count);
        m_impl->valid = invalid_count == 0;
        return m_impl->valid;
    }

    void
    ParticleImpulseTransaction::Commit ()
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->valid && !m_impl->committed,
                                         "Only a valid uncommitted particle impulse may commit.");
        for (auto& tile : m_impl->tiles)
        {
            auto const data = tile.particles->getParticleTileData();
            auto const* candidate = tile.candidate.dataPtr();
            amrex::GpuArray<amrex::ParticleReal*, 4> carry{};
            for (int d = 0; d < 4; ++d)
            {
                carry[d] = tile.particles->GetStructOfArrays()
                               .GetRealData(tile.components[d]).dataPtr();
            }
            amrex::ParallelFor(static_cast<long>(tile.candidate.size()),
                [=] AMREX_GPU_DEVICE (long ip)
            {
                for (int d = 0; d < 3; ++d)
                {
                    data.m_rdata[PIdx::ux + d][ip] = candidate[ip].velocity[d];
                    carry[d][ip] = candidate[ip].momentum_carry[d];
                }
                carry[3][ip] = candidate[ip].energy_carry;
            });
        }
        amrex::Gpu::streamSynchronize();
        m_impl->committed = true;
    }

    amrex::MultiFab const& ParticleImpulseTransaction::RequestedWork () const
    { return m_impl->requested_work; }
    amrex::MultiFab const& ParticleImpulseTransaction::ActualWork () const
    { return m_impl->actual_work; }
    amrex::MultiFab const& ParticleImpulseTransaction::ActualImpulse () const
    { return m_impl->actual_impulse; }
    amrex::MultiFab const& ParticleImpulseTransaction::EnergyCarryChange () const
    { return m_impl->energy_carry_change; }
    amrex::MultiFab const& ParticleImpulseTransaction::MomentumCarryChange () const
    { return m_impl->momentum_carry_change; }
    amrex::MultiFab const& ParticleImpulseTransaction::NumericalEnergyResidual () const
    { return m_impl->numerical_energy_residual; }
    amrex::MultiFab const& ParticleImpulseTransaction::WorkPartitionResidual () const
    { return m_impl->work_partition_residual; }
    amrex::MultiFab const& ParticleImpulseTransaction::MaterialMass () const
    { return m_impl->mass; }
    amrex::MultiFab const& ParticleImpulseTransaction::WorkVelocity () const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->has_work_velocity,
            "Work velocity must be requested when staging the particle impulse.");
        return m_impl->work_velocity;
    }
}

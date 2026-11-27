/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleImpulse.H"

#include "MaterialKineticWork.H"
#include "ParticleCellShape.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "RZMomentGeometry.H"
#include "RZParticleCellShape.H"
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
        // Particle and field precision can differ. Complete each contribution
        // in its expression type, then convert once to the ledger's storage
        // type; AMReX atomics require identical pointer and value types.
        template <typename T>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void AddFieldReal (amrex::Real* destination, T value) noexcept
        {
            amrex::Gpu::Atomic::AddNoRet(destination, static_cast<amrex::Real>(value));
        }

        // Reference ordering for replay-sensitive private source deposits.
        // Each GPU tile retains the existing per-particle arithmetic but uses
        // one thread to visit its particles in index order. Other tiles still
        // use their own streams/FABs; CPU and unselected callers retain For.
        template <typename F>
        void ForParticleContributions (long count, bool deterministic, F const& f)
        {
#ifdef AMREX_USE_GPU
            if (deterministic && amrex::Gpu::inLaunchRegion())
            {
                if (count > 0)
                {
                    amrex::For(1L, [=] AMREX_GPU_DEVICE (long)
                    {
                        for (long particle = 0; particle < count; ++particle)
                        {
                            f(particle);
                        }
                    });
                }
                return;
            }
#endif
            amrex::ignore_unused(deterministic);
            amrex::For(count, f);
        }

#if defined(WARPX_DIM_RZ)
        // NVCC requires the enclosing function of an extended device lambda
        // to have public access. Keep this kernel file-local, not a new API.
        void
        DepositRZActualRate (
            WarpXParticleContainer::ParticleTileType& particles,
            MaterialImpulseCandidate const* actual, long count, bool candidate,
            amrex::Real rest_mass, amrex::Array4<amrex::Real const> mass,
            amrex::Array4<amrex::Real const> inertia,
            amrex::Array4<amrex::Real> rate, amrex::Geometry const& geometry,
            bool angular_overlap, bool native_axis, bool reflecting_radial_density,
            bool reflecting_axial_density, bool deterministic)
        {
            auto const data = particles.getParticleTileData();
            auto const lower = geometry.ProbLoArray(),
                       inverse = geometry.InvCellSizeArray();
            RZMomentMetric const metric(geometry);
            int const radial_hi = geometry.Domain().bigEnd(0);
            int const axial_hi = geometry.Domain().bigEnd(1);
            int const shape_order = WarpX::nox;
            // CPU scatter is NOT SIMD; device scatter uses the same native
            // accumulator as Stage. Input sampling ignores pending carries.
            ForParticleContributions(
                count, deterministic,
                [=] AMREX_GPU_DEVICE(long p)
                {
                    auto const particle =
                        WarpXParticleContainer::ParticleType(data, p);
                    auto const owner =
                        amrex::getParticleCell(particle, lower, inverse);
                    auto const [i, j, k] = owner.dim3();
                    auto const weight = data.m_rdata[PIdx::w][p] * rest_mass;
                    auto const radius = data.m_rdata[PIdx::r][p];
                    auto const theta = data.m_rdata[PIdx::theta][p];
                    auto const lever = radius / metric.Angular(i).mean_radius;
                    amrex::Real squared = 0;
                    amrex::Real velocity[3];
                    for (int d = 0; d < 3; ++d)
                    {
                        velocity[d] = candidate
                                          ? static_cast<amrex::Real>(
                                                actual[p].velocity[d])
                                          : static_cast<amrex::Real>(
                                                data.m_rdata[PIdx::ux + d][p]);
                        squared += velocity[d] * velocity[d];
                    }
                    auto const gamma = std::sqrt(1 + squared / PhysConst::c2);
                    auto const ux = velocity[0] / gamma;
                    auto const uy = velocity[1] / gamma;
                    auto const uz = velocity[2] / gamma;
                    auto const cosine = std::cos(theta), sine = std::sin(theta);
                    if (angular_overlap)
                    {
                        ParticleCellShape<2> shape;
                        shape.center = {i, j, k};
                        for (int d = 0; d < 2; ++d) {
                            shape.fraction[d] = (particle.pos(d) - lower[d]) * inverse[d] -
                                shape.center[d];
                        }
                        InitializeRZNativeCellAverage(shape, metric, radial_hi,
                                                     native_axis, shape_order,
                                                     reflecting_radial_density,
                                                     reflecting_axial_density, axial_hi);
                        amrex::Real const cylindrical[3] = {
                            ux * cosine + uy * sine, -ux * sine + uy * cosine, uz};
                        for (int s = 0; s < shape.size; ++s)
                        {
                            auto const w = shape.Weight(s);
                            if (w == 0) { continue; }
                            auto const cell = shape.Cell(s);
                            for (int d = 0; d < 3; ++d)
                            {
                                auto ratio = w / mass(cell[0], cell[1], cell[2]);
                                if (d == 1) {
                                    auto const moment = inertia(cell[0], cell[1], cell[2]);
                                    ratio = moment > 0 ? w * radius /
                                        metric.Angular(cell[0]).mean_radius / moment : 0;
                                }
                                AddFieldReal(&rate(cell[0], cell[1], cell[2], d),
                                             weight * ratio * cylindrical[d]);
                            }
                        }
                        return;
                    }
                    AddFieldReal(&rate(i, j, k, 0),
                                 weight / mass(i, j, k) *
                                     (ux * cosine + uy * sine));
                    AddFieldReal(&rate(i, j, k, 1),
                                 inertia(i, j, k) > 0
                                     ? weight * lever / inertia(i, j, k) *
                                           (-ux * sine + uy * cosine)
                                     : 0);
                    AddFieldReal(&rate(i, j, k, 2),
                                 weight / mass(i, j, k) * uz);
                });
        }
#endif

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

        amrex::Real
        CandidateMaximumBeta (MaterialImpulseCandidate const* candidates, long count)
        {
            amrex::ReduceOps<amrex::ReduceOpMax> ops;
            amrex::ReduceData<amrex::Real> data(ops);
            using Tuple = typename decltype(data)::Type;
            ops.eval(count, data, [=] AMREX_GPU_DEVICE(long p) -> Tuple {
                amrex::Real norm = 0;
                for (auto const value : candidates[p].velocity)
                {
                    norm += value * value;
                }
                return {amrex::Math::isfinite(norm) ? norm / (PhysConst::c2 + norm)
                    : std::numeric_limits<amrex::Real>::infinity()};
            });
            return amrex::get<0>(data.value());
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
                              std::string const& path, bool local_cylindrical)
    {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!local_cylindrical,
            "Local cylindrical carry inventory requires RZ or RCYLINDER.");
#endif
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
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                auto const* theta = iterator.GetStructOfArrays().GetRealData(PIdx::theta).dataPtr();
#endif
                amrex::GpuArray<amrex::ParticleReal const*, 4> carry{};
                for (int d = 0; d < 4; ++d)
                {
                    carry[d] = iterator.GetStructOfArrays().GetRealData(components[d]).dataPtr();
                }
                ops.eval(iterator.numParticles(), data, [=] AMREX_GPU_DEVICE (long ip) -> Tuple
                {
                    auto const weighted_mass = weight[ip] * mass;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    if (local_cylindrical) {
                        auto const angle = static_cast<amrex::Real>(theta[ip]);
                        auto const cosine = std::cos(angle);
                        auto const sine = std::sin(angle);
                        return {weighted_mass * (carry[0][ip] * cosine + carry[1][ip] * sine),
                                weighted_mass * (-carry[0][ip] * sine + carry[1][ip] * cosine),
                                weighted_mass * carry[2][ip], weighted_mass * carry[3][ip]};
                    }
#endif
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
            std::string m_name;
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
            entry.m_name = name;
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

    ParticleImpulseMaterial::ParticleImpulseMaterial (
        std::unique_ptr<Impl> state)
        : m_impl(std::move(state))
    {
    }

    WarpXParticleContainer::Base const*
    ParticleImpulseMaterial::CandidateSpecies (std::string const& species_name) const
    {
        if (!m_impl || !m_impl->cloned || m_impl->committed) { return nullptr; }
        for (auto const& species : m_impl->species) {
            if (species.m_name == species_name) { return species.owned.get(); }
        }
        return nullptr;
    }

    bool
    ParticleImpulseMaterial::ValidateLive () const
    {
        bool valid = m_impl && m_impl->cloned && !m_impl->committed &&
                     !m_impl->species.empty();
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            return false;
        }
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
#if defined(WARPX_DIM_RZ)
                    mismatch = mismatch || before.m_rdata[PIdx::theta][p] !=
                                               after.m_rdata[PIdx::theta][p];
#endif
                    if (mismatch) { amrex::HostDevice::Atomic::Add(invalid_ptr, 1); }
                });
            }
        }
        invalid_layout = amrex::max(invalid_layout, invalid.dataValue());
        amrex::ParallelDescriptor::ReduceIntMax(invalid_layout);
        if (invalid_layout != 0) { return false; }
        return true;
    }

    std::vector<ParticleImpulseMaterial::SpeciesView>
    ParticleImpulseMaterial::CandidateSpecies () const
    {
        std::vector<SpeciesView> views;
        if (!m_impl || !m_impl->cloned || m_impl->committed)
        {
            return views;
        }
        for (auto const& species : m_impl->species)
        {
            views.push_back(
                {species.m_name, species.mass, species.owned.get()});
        }
        return views;
    }

    bool
    ParticleImpulseMaterial::TryCloneCandidate (
        std::unique_ptr<ParticleImpulseMaterial>& output) const
    {
        if (!ValidateLive())
        {
            return false;
        }
        auto state = std::make_unique<Impl>();
        state->cloned = true;
        for (auto const& original : m_impl->species)
        {
            Impl::Species species;
            species.live = original.live;
            species.mass = original.mass;
            species.m_name = original.m_name;
            species.owned = std::make_unique<WarpXParticleContainer::Base>(
                original.owned->make_alike<>());
            species.owned->SetArena(original.owned->arena());
            species.owned->copyParticles(*original.owned, true);
            species.particles = species.owned.get();
            for (auto const& old : original.baseline)
            {
                Impl::Species::Baseline baseline;
                baseline.key = old.key;
                baseline.count = old.count;
                baseline.components = old.components;
                baseline.values.resize(old.values.size());
                amrex::Gpu::copy(amrex::Gpu::deviceToDevice, old.values.begin(),
                                 old.values.end(), baseline.values.begin());
                species.baseline.push_back(std::move(baseline));
            }
            state->species.push_back(std::move(species));
        }
        amrex::Gpu::streamSynchronize();
        output.reset(new ParticleImpulseMaterial(std::move(state)));
        return true;
    }

    bool
    ParticleImpulseMaterial::Commit ()
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_impl->cloned && !m_impl->committed,
            "Only an uncommitted private material state may commit.");
        if (!ValidateLive())
        {
            return false;
        }
        for (auto& species : m_impl->species)
        {
            std::vector<int> components{PIdx::ux, PIdx::uy, PIdx::uz};
            auto const& names = species.particles->GetRealSoANames();
            for (int d = 0; d < static_cast<int>(names.size()); ++d)
            {
                if (names[d].starts_with("radiation_impulse_"))
                {
                    components.push_back(d);
                }
            }
            for (auto const& [key, trial] : species.particles->GetParticles(0))
            {
                if (trial.numParticles() == 0)
                {
                    continue;
                }
                auto& live = species.live->GetParticles(0).at(key);
                for (auto component : components)
                {
                    auto const& src =
                        trial.GetStructOfArrays().GetRealData(component);
                    auto& dst = live.GetStructOfArrays().GetRealData(component);
                    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, src.begin(),
                                     src.end(), dst.begin());
                }
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
            int grid = -1;
            amrex::Real rest_mass = 0;
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
        amrex::MultiFab work_accounting_bound;
        amrex::MultiFab work_accounting_particle_count;
        amrex::MultiFab requested_projection_bound;
        amrex::MultiFab requested_projection_count;
        amrex::MultiFab mass;
        amrex::MultiFab work_velocity;
        amrex::MultiFab angular_inertia;
        amrex::MultiFab conjugate_impulse;
        amrex::MultiFab conjugate_carry_change;
        amrex::MultiFab conjugate_rotation_scale;
        amrex::MultiFab conjugate_roundoff_bound;
        bool angular = false;
        bool angular_overlap = false;
        bool deterministic_overlap = false;
        bool reflecting_radial_density = false;
        bool reflecting_axial_density = false;
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
        ParticleImpulseAssignment assignment, bool need_source_projection_bound)
    {
        ParticleImpulseMaterial material(particles, species_names, false);
        return Stage(material, path, cell_impulse, need_work_velocity, assignment,
                     need_source_projection_bound);
    }

    bool
    ParticleImpulseTransaction::Stage (
        ParticleImpulseMaterial& material, std::string const& path,
        amrex::MultiFab const& cell_impulse, bool need_work_velocity,
        ParticleImpulseAssignment assignment, bool need_source_projection_bound)
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
        bool const reflecting_axial_density =
            assignment == ParticleImpulseAssignment::RZAngularReflectingClosedNodalCellAverage;
        bool const reflecting_radial_density = reflecting_axial_density ||
            assignment == ParticleImpulseAssignment::RZAngularReflectingRadialNodalCellAverage;
        bool const angular_overlap = reflecting_radial_density ||
            assignment == ParticleImpulseAssignment::RZAngularNodalCellAverage;
        bool const cartesian_overlap =
            assignment == ParticleImpulseAssignment::ConservativeNodalCellAverage;
        bool const bounded_overlap = angular_overlap || cartesian_overlap;
        bool const projection_requested = need_source_projection_bound &&
                                          bounded_overlap && need_work_velocity;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!need_source_projection_bound || projection_requested,
            "Source work projection bounds require conservative overlap and work velocity.");
        bool const angular = assignment == ParticleImpulseAssignment::RZAngularConservative ||
            angular_overlap;
        bool const shaped = angular_overlap ||
            (assignment != ParticleImpulseAssignment::NearestCell && !angular);
        bool const nearest = assignment == ParticleImpulseAssignment::NearestCell;
        m_impl->angular = angular;
        m_impl->angular_overlap = angular_overlap;
        // The angular P1 source reconstructs its material frame from these
        // overlapping deposits. Rejected intervals must replay the same
        // accepted ledger, including its compensated low bits. Order particle
        // and ghost contributions without changing per-particle arithmetic or bounds.
        m_impl->deterministic_overlap = projection_requested;
        m_impl->reflecting_radial_density = reflecting_radial_density;
        m_impl->reflecting_axial_density = reflecting_axial_density;
#if !defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!angular, "Angular particle impulse assignment requires RZ.");
#endif
        int const shape_order = WarpX::nox;
        int support_entries = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { support_entries *= shape_order + 2; }
        // <=64 operations per stencil entry cover native shape formation,
        // normalized component increments and source/secant dot partitions.
        // Keep the original linear-RZ floor, not a residual-fitted constant.
        auto const overlap_fixed_operations = amrex::max(1024, 64 * support_entries + 64);
        int const shape_ghosts = shaped ? (shape_order + 2) / 2 : 0;
        auto const& geometry = WarpX::GetInstance().Geom(0);
#if defined(WARPX_DIM_RZ)
        RZMomentMetric const angular_metric(geometry);
        int const axial_hi = geometry.Domain().bigEnd(1);
#endif
        [[maybe_unused]] bool const native_axis = WarpX::GetInstance().verboncoeurAxisCorrection();
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
        if (angular_overlap)
        {
#if defined(WARPX_DIM_RZ)
            // WarpX's RZ build also uses Cartesian AMReX geometry metadata;
            // the native deposit and this operator apply the cylindrical
            // measure explicitly, independently of that metadata flag.
            bool const reflecting_z = reflecting_axial_density && !geometry.isPeriodic(1) &&
                geometry.Domain().length(0) >= shape_ghosts &&
                geometry.Domain().length(1) >= shape_ghosts &&
                WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
                WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
                WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
                WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
                WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
                WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
                WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shape_order >= 1 && shape_order <= 3 &&
                (geometry.Coord() == 0 || geometry.Coord() == 1) &&
                geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
                (reflecting_axial_density ? reflecting_z : geometry.isPeriodic(1)) &&
                WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting,
                "RZ nodal angular assignment requires native shapes 1-3, an axis, "
                "reflecting radial material and periodic z, or the explicit "
                "closed-nodal policy with PEC/reflecting endcaps and stencil support.");
#endif
        }
        else if (shaped)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.Coord() == 0 &&
                (geometry.isAllPeriodic() || reflecting),
                "Nodal radiation force assignment requires periodic Cartesian geometry.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shape_order >= 1 && shape_order <= 4 &&
                (assignment == ParticleImpulseAssignment::NativeNodalCellAverage || reflecting ||
                 (cartesian_overlap && shape_order <= 3) ||
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
        if (projection_requested)
        {
            m_impl->requested_projection_bound.define(ba, dm, 1, shape_ghosts);
            m_impl->requested_projection_count.define(ba, dm, 1, shape_ghosts);
            m_impl->requested_projection_bound.setVal(0);
            m_impl->requested_projection_count.setVal(0);
        }
        m_impl->work_accounting_bound.define(ba, dm, 1, 0);
        m_impl->work_accounting_bound.setVal(0);
        m_impl->work_accounting_particle_count.define(ba, dm, 1, 0);
        m_impl->work_accounting_particle_count.setVal(0);
        m_impl->actual_impulse.define(ba, dm, 3, 0);
        m_impl->actual_impulse.setVal(0);
        m_impl->momentum_carry_change.define(ba, dm, 3, 0);
        m_impl->momentum_carry_change.setVal(0);
        m_impl->mass.define(ba, dm, 1, shape_ghosts);
        auto& cell_mass = m_impl->mass;
        cell_mass.setVal(0);
        if (angular) {
            m_impl->angular_inertia.define(ba, dm, 1, shape_ghosts);
            m_impl->conjugate_impulse.define(ba, dm, 3, 0);
            m_impl->conjugate_carry_change.define(ba, dm, 3, 0);
            m_impl->conjugate_rotation_scale.define(ba, dm, 3, 0);
            m_impl->conjugate_roundoff_bound.define(ba, dm, 3, 0);
            m_impl->angular_inertia.setVal(0);
            m_impl->conjugate_impulse.setVal(0);
            m_impl->conjugate_carry_change.setVal(0);
            m_impl->conjugate_rotation_scale.setVal(0);
            m_impl->conjugate_roundoff_bound.setVal(0);
        }
        else if (cartesian_overlap) {
            m_impl->conjugate_rotation_scale.define(ba, dm, 3, 0);
            m_impl->conjugate_roundoff_bound.define(ba, dm, 3, 0);
            m_impl->conjugate_rotation_scale.setVal(0);
            m_impl->conjugate_roundoff_bound.setVal(0);
        }
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
        amrex::Long source_particle_count = 0;
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
                if (bounded_overlap) { source_particle_count += iterator.numParticles(); }
                auto const data = iterator.GetParticleTile().getParticleTileData();
                auto const mass_field = cell_mass.array(iterator);
                [[maybe_unused]] auto const inertia_field = angular ? m_impl->angular_inertia.array(iterator)
                    : amrex::Array4<amrex::Real>{};
                ForParticleContributions(iterator.numParticles(), m_impl->deterministic_overlap,
                    [=] AMREX_GPU_DEVICE (long ip)
                {
                    auto const p = WarpXParticleContainer::ParticleType(data, ip);
                    if (angular || cartesian_overlap) {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (!amrex::Math::isfinite(p.pos(d)) || p.pos(d) < plo[d] ||
                                p.pos(d) >= phi[d]) {
                                amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                                return;
                            }
                        }
                    }
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
#if defined(WARPX_DIM_RZ)
                        if (angular_overlap) {
                            InitializeRZNativeCellAverage(shape, angular_metric, domain_hi,
                                                         native_axis, shape_order,
                                                         reflecting_radial_density,
                                                         reflecting_axial_density, axial_hi);
                        }
#endif
                        for (int s = 0; s < shape.size; ++s)
                        {
                            auto const w = shape.Weight(s);
                            if (w == 0) { continue; }
                            auto const cell = FoldReflectingCell(
                                shape.Cell(s), reflecting, domain_lo, domain_hi);
                            AddFieldReal(&mass_field(cell[0], cell[1], cell[2]),
                                w * data.m_rdata[PIdx::w][ip] * mass);
#if defined(WARPX_DIM_RZ)
                            if (angular_overlap) {
                                auto const lever = data.m_rdata[PIdx::r][ip] /
                                    angular_metric.Angular(cell[0]).mean_radius;
                                AddFieldReal(&inertia_field(cell[0], cell[1], cell[2]),
                                    w * data.m_rdata[PIdx::w][ip] * mass * lever * lever);
                            }
#endif
                        }
                        return;
                    }
                    AddFieldReal(
                        &mass_field(i, j, k), data.m_rdata[PIdx::w][ip] * mass);
#if defined(WARPX_DIM_RZ)
                    if (angular) {
                        auto const lever = data.m_rdata[PIdx::r][ip] / angular_metric.Angular(i).mean_radius;
                        AddFieldReal(&inertia_field(i, j, k),
                            data.m_rdata[PIdx::w][ip] * mass * lever * lever);
                    }
#endif
                });
            }
        }
        if (reflecting || angular || cartesian_overlap) {
            int invalid_positions = invalid.dataValue();
            amrex::ParallelDescriptor::ReduceIntMax(invalid_positions);
            if (invalid_positions != 0) { return false; }
        }
        if (shaped)
        {
            cell_mass.SumBoundary(geometry.periodicity(), m_impl->deterministic_overlap);
            cell_mass.FillBoundary(geometry.periodicity());
            if (angular_overlap) {
                m_impl->angular_inertia.SumBoundary(
                    geometry.periodicity(), m_impl->deterministic_overlap);
                m_impl->angular_inertia.FillBoundary(geometry.periodicity());
            }
        }
        if (bounded_overlap) { amrex::ParallelDescriptor::ReduceLongSum(source_particle_count); }
        for (amrex::MFIter iterator(cell_mass); iterator.isValid(); ++iterator)
        {
            auto const mass = cell_mass.const_array(iterator);
            auto const impulse = cell_impulse.const_array(iterator);
            auto const inertia = angular ? m_impl->angular_inertia.const_array(iterator)
                : amrex::Array4<amrex::Real const>{};
            amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                bool valid = mass(i, j, k) >= 0 && amrex::Math::isfinite(mass(i, j, k));
                for (int d = 0; d < 3; ++d)
                {
                    valid = valid && amrex::Math::isfinite(impulse(i, j, k, d))
                        && (impulse(i, j, k, d) == 0 || mass(i, j, k) > 0);
                }
                if (angular) {
                    valid = valid && inertia(i, j, k) >= 0 && amrex::Math::isfinite(inertia(i, j, k))
                        && (impulse(i, j, k, 1) == 0 || inertia(i, j, k) > 0);
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
                staged.grid = iterator.index();
                staged.rest_mass = species_mass;
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
                auto const inertia = angular ? m_impl->angular_inertia.const_array(iterator)
                    : amrex::Array4<amrex::Real const>{};
                auto const conjugate = angular ? m_impl->conjugate_impulse.array(iterator)
                    : amrex::Array4<amrex::Real>{};
                auto const conjugate_carry = angular ? m_impl->conjugate_carry_change.array(iterator)
                    : amrex::Array4<amrex::Real>{};
                auto const rotation_scale = angular || cartesian_overlap ?
                    m_impl->conjugate_rotation_scale.array(iterator) : amrex::Array4<amrex::Real>{};
                auto const residual = m_impl->numerical_energy_residual.array(iterator);
                auto const partition_residual = m_impl->work_partition_residual.array(iterator);
                auto const accounting_scale = m_impl->work_accounting_bound.array(iterator);
                auto const accounting_count =
                    m_impl->work_accounting_particle_count.array(iterator);
                amrex::Array4<amrex::Real> work_velocity;
                if (need_work_velocity) { work_velocity = m_impl->work_velocity.array(iterator); }
                amrex::Array4<amrex::Real> projection_scale, projection_count;
                if (projection_requested)
                {
                    projection_scale = m_impl->requested_projection_bound.array(iterator);
                    projection_count = m_impl->requested_projection_count.array(iterator);
                }
                amrex::GpuArray<amrex::ParticleReal const*, 4> carry{};
                for (int d = 0; d < 4; ++d)
                {
                    carry[d] = staged.particles->GetStructOfArrays()
                                   .GetRealData(components[d]).dataPtr();
                }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                auto const position = GetParticlePosition<PIdx>(iterator);
#endif
                ForParticleContributions(np, m_impl->deterministic_overlap,
                    [=] AMREX_GPU_DEVICE (long ip)
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
#if defined(WARPX_DIM_RZ)
                    amrex::Real const angular_lever = angular
                        ? data.m_rdata[PIdx::r][ip] / angular_metric.Angular(i).mean_radius : 1;
#else
                    amrex::Real const angular_lever = 1;
#endif
                    ParticleCellShape<AMREX_SPACEDIM> shape;
                    shape.center = {i, j, k};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        shape.fraction[d] = (p.pos(d) - plo[d]) * dxi[d] - shape.center[d];
                    }
                    if (shaped) { shape.Initialize(shape_order); }
#if defined(WARPX_DIM_RZ)
                    if (angular_overlap) {
                        InitializeRZNativeCellAverage(shape, angular_metric, domain_hi,
                                                     native_axis, shape_order,
                                                     reflecting_radial_density,
                                                     reflecting_axial_density, axial_hi);
                    }
#endif
                    for (int d = 0; d < 3; ++d)
                    {
                        velocity[d] = data.m_rdata[PIdx::ux + d][ip];
                        old_carry[d] = carry[d][ip];
                        if (!shaped) { increment[d] = impulse(i, j, k, d) / mass(i, j, k); }
                    }
                    if (angular && !shaped) {
                        increment[1] = inertia(i, j, k) > 0
                            ? angular_lever * impulse(i, j, k, 1) / inertia(i, j, k) : 0;
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
                                auto ratio = w / m;
#if defined(WARPX_DIM_RZ)
                                if (angular_overlap && d == 1) {
                                    auto const moment = inertia(cell[0], cell[1], cell[2]);
                                    ratio = moment > 0 ? w * data.m_rdata[PIdx::r][ip] /
                                        angular_metric.Angular(cell[0]).mean_radius / moment : 0;
                                }
#endif
                                increment[d] += ratio * impulse(cell[0], cell[1], cell[2], d);
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
                    // Independent uncancelled rotation/work operands. These
                    // are retained only for the angular nearest-cell contract;
                    // no measured residual or shaped remap enters a bound.
                    amrex::Real angular_work_operands = 0;
                    amrex::Real partition_operands = 0;
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
                        if (angular)
                        {
                            auto const rotated_x = std::abs(radial * cosine) +
                                                   std::abs(azimuthal * sine);
                            auto const rotated_y = std::abs(radial * sine) +
                                                   std::abs(azimuthal * cosine);
                            // Bound both Cartesian work and its cylindrical
                            // conjugate expression before cancellation. In
                            // particular a transverse zero-work kick still
                            // has its actual rotation conditioning scale.
                            angular_work_operands =
                                rotated_x * (2 * std::abs(velocity[0]) + rotated_x) / denominator +
                                rotated_y * (2 * std::abs(velocity[1]) + rotated_y) / denominator +
                                std::abs(increment[2]) *
                                    (2 * std::abs(velocity[2]) + std::abs(increment[2])) /
                                    denominator +
                                std::abs(radial) *
                                    (std::abs(x * cosine) + std::abs(y * sine)) +
                                std::abs(azimuthal) *
                                    (std::abs(x * sine) + std::abs(y * cosine)) +
                                std::abs(increment[2] * secant[2]);
                        }
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
                                amrex::Real source_operands = 0;
                                for (int d = 0; d < 3; ++d)
                                {
                                    auto component_ratio = ratio;
#if defined(WARPX_DIM_RZ)
                                    if (angular_overlap && d == 1) {
                                        auto const moment = inertia(cell[0], cell[1], cell[2]);
                                        component_ratio = moment > 0
                                            ? w * data.m_rdata[PIdx::r][ip] /
                                                angular_metric.Angular(cell[0]).mean_radius / moment
                                            : 0;
                                    }
#endif
                                    auto const term = component_ratio *
                                        impulse(cell[0], cell[1], cell[2], d)
                                        * secant[d];
                                    work += term;
                                    scale += std::abs(term);
                                    source_operands += std::abs(term);
                                    if (need_work_velocity)
                                    {
                                        AddFieldReal(
                                            &work_velocity(cell[0], cell[1], cell[2], d),
                                            weight_mass * component_ratio * secant[d]);
                                    }
                                }
                                partition += work;
                                AddFieldReal(
                                    &requested_work(cell[0], cell[1], cell[2]), weight_mass * work);
                                if (projection_requested)
                                {
                                    AddFieldReal(&projection_scale(cell[0], cell[1], cell[2]),
                                                 weight_mass * source_operands);
                                    AddFieldReal(&projection_count(cell[0], cell[1], cell[2]),
                                                 amrex::Real(1));
                                }
                            }
                            auto const difference = partition - result.requested_work;
                            partition_operands = scale;
                            auto const partition_operations = bounded_overlap
                                ? amrex::max(256, 16 * support_entries) : 256;
                            if (!amrex::Math::isfinite(scale) ||
                                std::abs(difference) > partition_operations * epsilon *
                                    (scale + (angular_overlap ? angular_work_operands : 0)))
                            {
                                amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                            }
                            AddFieldReal(&partition_residual(i, j, k),
                                weight_mass * difference);
                        }
                        else
                        {
                            for (int d = 0; d < 3; ++d)
                            {
                                auto const contribution = angular && d == 1
                                    ? (inertia(i, j, k) > 0
                                        ? weight_mass * angular_lever / inertia(i, j, k) * secant[d] : 0)
                                    : weight_mass / mass(i, j, k) * secant[d];
                                AddFieldReal(&work_velocity(i, j, k, d),
                                    contribution);
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
                    if (angular)
                    {
                        amrex::Real transverse_scale = 0;
                        for (int d = 0; d < 2; ++d)
                        {
                            transverse_scale += std::abs(static_cast<amrex::Real>(velocity[d])) +
                                std::abs(static_cast<amrex::Real>(result.velocity[d])) +
                                std::abs(increment[d]) + std::abs(old_carry[d]) +
                                std::abs(result.momentum_carry[d]);
                        }
                        auto const axial_scale =
                            std::abs(static_cast<amrex::Real>(velocity[2])) +
                            std::abs(static_cast<amrex::Real>(result.velocity[2])) +
                            std::abs(increment[2]) + std::abs(old_carry[2]) +
                            std::abs(result.momentum_carry[2]);
                        AddFieldReal(&rotation_scale(i, j, k, 0),
                                     weight_mass * transverse_scale);
                        AddFieldReal(&rotation_scale(i, j, k, 1),
                                     weight_mass * std::abs(angular_lever) *
                                         transverse_scale);
                        AddFieldReal(&rotation_scale(i, j, k, 2),
                                     weight_mass * axial_scale);
                    }
                    else if (cartesian_overlap)
                    {
                        for (int d = 0; d < 3; ++d) {
                            auto const operands = std::abs(static_cast<amrex::Real>(velocity[d])) +
                                std::abs(static_cast<amrex::Real>(result.velocity[d])) +
                                std::abs(increment[d]) + std::abs(old_carry[d]) +
                                std::abs(result.momentum_carry[d]);
                            AddFieldReal(&rotation_scale(i, j, k, d), weight_mass * operands);
                        }
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
                        AddFieldReal(
                            &actual_impulse(i, j, k, d), weight_mass * applied[d]);
                        AddFieldReal(
                            &momentum_change(i, j, k, d), weight_mass * carry_change[d]);
                        if (angular) {
                            auto const lever = d == 1 ? angular_lever : 1;
                            AddFieldReal(&conjugate(i, j, k, d), weight_mass * lever * applied[d]);
                            AddFieldReal(&conjugate_carry(i, j, k, d), weight_mass * lever * carry_change[d]);
                        }
                    }
                    if (!shaped)
                    {
                        AddFieldReal(
                            &requested_work(i, j, k), weight_mass * result.requested_work);
                    }
                    if (nearest || angular || cartesian_overlap)
                    {
                        amrex::Real const operand_scale = std::abs(result.requested_work) +
                            std::abs(result.actual_work) + std::abs(carry[3][ip]) +
                            std::abs(result.energy_carry) + angular_work_operands +
                            partition_operands;
                        amrex::Real const weighted_operand_scale =
                            static_cast<amrex::Real>(weight_mass) * operand_scale;
                        if (!amrex::Math::isfinite(operand_scale) ||
                            !amrex::Math::isfinite(weighted_operand_scale) ||
                            (operand_scale > 0 && weighted_operand_scale <
                                std::numeric_limits<amrex::Real>::min()))
                        {
                            amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                            return;
                        }
                        AddFieldReal(&accounting_scale(i, j, k), weighted_operand_scale);
                        AddFieldReal(&accounting_count(i, j, k), amrex::Real(1.0));
                    }
                    AddFieldReal(
                        &actual_work(i, j, k), weight_mass * result.actual_work);
                    AddFieldReal(
                        &energy_change(i, j, k),
                        weight_mass * (result.energy_carry - carry[3][ip]));
                    AddFieldReal(&residual(i, j, k), weight_mass *
                        (result.requested_work - result.actual_work
                         - (result.energy_carry - carry[3][ip])));
                });
                m_impl->tiles.push_back(std::move(staged));
            }
        }
        if (shaped)
        {
            m_impl->requested_work.SumBoundary(
                geometry.periodicity(), m_impl->deterministic_overlap);
            if (need_work_velocity) {
                m_impl->work_velocity.SumBoundary(
                    geometry.periodicity(), m_impl->deterministic_overlap);
            }
        }
        if (projection_requested)
        {
            m_impl->requested_projection_bound.SumBoundary(geometry.periodicity(), true);
            m_impl->requested_projection_count.SumBoundary(geometry.periodicity(), true);
            for (amrex::MFIter it(m_impl->requested_projection_bound); it.isValid(); ++it)
            {
                auto const bound = m_impl->requested_projection_bound.array(it);
                auto const count = m_impl->requested_projection_count.const_array(it);
                amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Three scalar work terms and three velocity reductions
                    // each have at most N source contributors.4N+64 bounds
                    // their two reduction orders, product reassociation,
                    // source dot and velocity<->beta conversion. Correct the
                    // positive norm's own rounding before rounding outward.
                    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
                    auto const neps = (4 * count(i, j, k) + 64) * epsilon;
                    auto const scale = bound(i, j, k);
                    if (!(count(i, j, k) >= 0 && neps < .125 && scale >= 0) ||
                        !amrex::Math::isfinite(scale + neps))
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
                    if (scale == 0) { bound(i, j, k) = 0; return; }
                    auto const evaluated = neps / (1 - neps) * scale;
                    bound(i, j, k) = std::nextafter(evaluated * (1 + 8 * epsilon),
                        std::numeric_limits<amrex::Real>::infinity());
                    if (!amrex::Math::isfinite(bound(i, j, k)) ||
                        evaluated < std::numeric_limits<amrex::Real>::min())
                    { amrex::HostDevice::Atomic::Add(invalid_ptr, 1); }
                });
            }
        }
        for (amrex::MFIter iterator(m_impl->work_accounting_bound);
             iterator.isValid(); ++iterator)
        {
            auto const bound = m_impl->work_accounting_bound.array(iterator);
            auto const count = m_impl->work_accounting_particle_count.const_array(iterator);
            amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const epsilon = std::numeric_limits<amrex::Real>::epsilon();
                // Computed-operand identity and deposition need gamma_(n+8).
                // Correcting the positive norm's gamma_(n+3) underestimate is
                // bounded by gamma_(2n+32); epsilon conservatively replaces u.
                // Angular conjugacy adds two basis rotations (8 products,
                // 4 sums), scalar/mass/inertia ratios and dot-product terms.
                // Reserve 32 more operations, using their uncancelled norms
                // accumulated above, not total kinetic/radiation energy.
                // Overlap includes the actual shape's source stencil,
                // two normalization reductions, both basis rotations and the
                // independent finite-work readback. Use the actual GLOBAL
                // particle count to cover neighboring-cell normalization; a
                // center-owned count cannot bound an overlapping source mass.
                amrex::Real const operations = bounded_overlap
                    ? 32 * static_cast<amrex::Real>(source_particle_count) + overlap_fixed_operations
                    : 2 * count(i, j, k) + (angular && need_work_velocity ? 64 : 32);
                amrex::Real const numerator = operations * epsilon;
                bool const safe = count(i, j, k) >= 0 &&
                    count(i, j, k) <= 1 / (32 * epsilon) &&
                    amrex::Math::isfinite(bound(i, j, k)) &&
                    numerator < 0.125 && bound(i, j, k) >= 0;
                if (!safe)
                {
                    amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                    return;
                }
                amrex::Real const gamma_n = numerator / (1 - numerator);
                amrex::Real const scale = bound(i, j, k);
                if (scale == 0) { return; }
                // Cover rounding in the bound evaluation itself, then round
                // outward. Do not claim a relative model for subnormal bounds.
                amrex::Real const evaluated = scale * gamma_n;
                bound(i, j, k) = std::nextafter(evaluated * (1 + 8 * epsilon),
                    std::numeric_limits<amrex::Real>::infinity());
                if (!amrex::Math::isfinite(bound(i, j, k)) ||
                    evaluated < std::numeric_limits<amrex::Real>::min())
                {
                    amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                }
            });
        }
        if (angular || cartesian_overlap)
        {
            for (amrex::MFIter iterator(m_impl->conjugate_roundoff_bound);
                 iterator.isValid(); ++iterator)
            {
                auto const bound = m_impl->conjugate_roundoff_bound.array(iterator);
                auto const scale = m_impl->conjugate_rotation_scale.const_array(iterator);
                auto const count = m_impl->work_accounting_particle_count.const_array(iterator);
                auto const source = cell_impulse.const_array(iterator);
                amrex::For(iterator.validbox(), 3,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int d)
                {
                    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
                    // Two mass/inertia reductions and four actual/carry
                    // momentum readbacks, each with at most N additions;
                    // 8N+128 also covers the per-particle Cartesian/RZ basis
                    // pair, candidate arithmetic and cell normalization.
                    auto const operations = bounded_overlap
                        ? 32 * static_cast<amrex::Real>(source_particle_count) + overlap_fixed_operations
                        : 8 * count(i, j, k) + 128;
                    auto const numerator = operations * epsilon;
                    auto operands = scale(i, j, k, d) + std::abs(source(i, j, k, d));
                    if (angular && d < 2)
                    {
                        operands += std::abs(source(i, j, k, 1 - d));
                    }
                    if (!(count(i, j, k) >= 0 &&
                          count(i, j, k) <= 1 / (32 * epsilon) &&
                          numerator < 0.125 && amrex::Math::isfinite(operands) &&
                          operands >= 0))
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                        return;
                    }
                    auto const gamma_n = numerator / (1 - numerator);
                    bound(i, j, k, d) = operands == 0 ? 0 :
                        std::nextafter(gamma_n * operands * (1 + 8 * epsilon),
                                       std::numeric_limits<amrex::Real>::infinity());
                    if (!amrex::Math::isfinite(bound(i, j, k, d)))
                    {
                        amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                    }
                });
            }
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
        if (angular && (!m_impl->conjugate_impulse.is_finite() ||
                        !m_impl->conjugate_carry_change.is_finite() ||
                        !m_impl->conjugate_roundoff_bound.is_finite())) { invalid_count = 1; }
        if (cartesian_overlap && !m_impl->conjugate_roundoff_bound.is_finite()) {
            invalid_count = 1;
        }
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
    amrex::MultiFab const& ParticleImpulseTransaction::ConjugateImpulse () const
    { return m_impl->angular ? m_impl->conjugate_impulse : m_impl->actual_impulse; }
    amrex::MultiFab const& ParticleImpulseTransaction::ConjugateMomentumCarryChange () const
    { return m_impl->angular ? m_impl->conjugate_carry_change : m_impl->momentum_carry_change; }
    amrex::MultiFab const& ParticleImpulseTransaction::RZConjugateMomentumRoundoffBound () const
    { return m_impl->conjugate_roundoff_bound; }
    amrex::MultiFab const& ParticleImpulseTransaction::EnergyCarryChange () const
    { return m_impl->energy_carry_change; }
    amrex::MultiFab const& ParticleImpulseTransaction::MomentumCarryChange () const
    { return m_impl->momentum_carry_change; }
    amrex::MultiFab const& ParticleImpulseTransaction::NumericalEnergyResidual () const
    { return m_impl->numerical_energy_residual; }
    amrex::MultiFab const& ParticleImpulseTransaction::WorkPartitionResidual () const
    { return m_impl->work_partition_residual; }
    amrex::MultiFab const& ParticleImpulseTransaction::MomentumAccountingRoundoffBound () const
    { return m_impl->conjugate_roundoff_bound; }
    amrex::MultiFab const& ParticleImpulseTransaction::WorkAccountingRoundoffBound () const
    { return m_impl->work_accounting_bound; }
    amrex::MultiFab const& ParticleImpulseTransaction::RequestedWorkProjectionRoundoffBound () const
    { return m_impl->requested_projection_bound; }
    amrex::MultiFab const& ParticleImpulseTransaction::MaterialMass () const
    { return m_impl->mass; }
    amrex::MultiFab const& ParticleImpulseTransaction::WorkVelocity () const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->has_work_velocity,
            "Work velocity must be requested when staging the particle impulse.");
        return m_impl->work_velocity;
    }

    bool
    ParticleImpulseTransaction::TryCandidateRateVelocity (
        amrex::MultiFab& output) const
    {
        return TryRateVelocity(output, true);
    }

    bool
    ParticleImpulseTransaction::TryInitialRateVelocity (
        amrex::MultiFab& output) const
    {
        return TryRateVelocity(output, false);
    }

    bool
    ParticleImpulseTransaction::TryRateVelocity (amrex::MultiFab& output,
                                                bool candidate) const
    {
        amrex::ignore_unused(candidate); // Non-RZ stubs reject before sampling.
        bool valid = m_impl->valid && !m_impl->committed && m_impl->angular;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            return false;
        }
        amrex::MultiFab rate(m_impl->mass.boxArray(),
                             m_impl->mass.DistributionMap(), 3, m_impl->mass.nGrowVect());
        rate.setVal(0);
#if defined(WARPX_DIM_RZ)
        auto const& geometry = WarpX::GetInstance().Geom(0);
        for (auto const& tile : m_impl->tiles)
        {
            DepositRZActualRate(*tile.particles, tile.candidate.dataPtr(),
                                static_cast<long>(tile.candidate.size()),
                                candidate, tile.rest_mass,
                                m_impl->mass.const_array(tile.grid),
                                m_impl->angular_inertia.const_array(tile.grid),
                                rate.array(tile.grid), geometry, m_impl->angular_overlap,
                                WarpX::GetInstance().verboncoeurAxisCorrection(),
                                m_impl->reflecting_radial_density,
                                m_impl->reflecting_axial_density,
                                m_impl->deterministic_overlap);
        }
        if (m_impl->angular_overlap) {
            rate.SumBoundary(geometry.periodicity(), m_impl->deterministic_overlap);
        }
        amrex::Gpu::streamSynchronize();
        if (!rate.is_finite())
        {
            return false;
        }
#endif
        if (!m_impl->angular_overlap) {
            output = std::move(rate);
            return true;
        }
        amrex::MultiFab valid_rate(rate.boxArray(), rate.DistributionMap(), 3, 0);
        amrex::MultiFab::Copy(valid_rate, rate, 0, 0, 3, 0);
        output = std::move(valid_rate);
        return true;
    }

    amrex::Real
    ParticleImpulseTransaction::MaximumBetaSquared () const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->valid, "Ion beta requires a valid staged impulse.");
        amrex::Real maximum = 0;
        for (auto const& tile : m_impl->tiles)
        {
            if (tile.candidate.empty())
            {
                continue;
            }
            maximum = amrex::max(maximum,
                CandidateMaximumBeta(tile.candidate.dataPtr(), tile.candidate.size()));
        }
        amrex::ParallelDescriptor::ReduceRealMax(maximum);
        return maximum;
    }
}

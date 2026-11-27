/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
// Integration draft for the frozen-position, static D/K source API. This is
// not a moving-medium transport test. Retained constitutive q is an explicit
// opt-in with an independent native-particle plus field momentum ledger.
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/PacketDiffusionNativeInterval.H"
#include "Radiation/ParticleImpulse.H"
#include "QuasiStaticNativeStressOracle.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    struct TileSnapshot
    {
        std::pair<int, int> key;
        std::size_t count = 0;
        std::vector<amrex::ParticleReal> reals;
        std::vector<int> integers;
        std::vector<std::uint64_t> ids;

        long double Value (int component, std::size_t particle) const
        {
            return reals[static_cast<std::size_t>(component)*count+particle];
        }
    };

    template <typename T>
    bool SameBits (std::vector<T> const& a, std::vector<T> const& b)
    {
        return a.size() == b.size() && (a.empty() ||
            std::memcmp(a.data(), b.data(), a.size()*sizeof(T)) == 0);
    }

    std::vector<TileSnapshot> Snapshot (WarpXParticleContainer const& particles)
    {
        std::vector<TileSnapshot> result;
        for (auto const& [key, tile] : particles.GetParticles(0))
        {
            TileSnapshot state;
            state.key = key;
            state.count = tile.numParticles();
            auto const& soa = tile.GetStructOfArrays();
            state.reals.resize(state.count*tile.NumRealComps());
            state.integers.resize(state.count*tile.NumIntComps());
            state.ids.resize(state.count);
            if (state.count != 0)
            {
                for (int c = 0; c < tile.NumRealComps(); ++c)
                {
                    auto const& values = soa.GetRealData(c);
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                        values.begin()+state.count, state.reals.data()+c*state.count);
                }
                for (int c = 0; c < tile.NumIntComps(); ++c)
                {
                    auto const& values = soa.GetIntData(c);
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                        values.begin()+state.count, state.integers.data()+c*state.count);
                }
                auto const& ids = soa.GetIdCPUData();
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, ids.begin(), ids.begin()+state.count,
                                 state.ids.begin());
            }
            result.push_back(std::move(state));
        }
        return result;
    }

    bool SameParticles (std::vector<TileSnapshot> const& a,
                        std::vector<TileSnapshot> const& b)
    {
        if (a.size() != b.size()) { return false; }
        for (std::size_t t = 0; t < a.size(); ++t)
        {
            if (a[t].key != b[t].key || a[t].count != b[t].count ||
                !SameBits(a[t].reals, b[t].reals) ||
                !SameBits(a[t].integers, b[t].integers) || !SameBits(a[t].ids, b[t].ids))
            {
                return false;
            }
        }
        return true;
    }

    // Checkpoint I/O may reorder particles. Compare every saved attribute by
    // native identity rather than assuming an in-memory tile ordering.
    bool SameCheckpointParticles (std::vector<TileSnapshot> const& a,
                                  std::vector<TileSnapshot> const& b)
    {
        using Record = std::pair<std::vector<amrex::ParticleReal>, std::vector<int>>;
        auto canonical = [] (std::vector<TileSnapshot> const& snapshot)
        {
            std::map<std::uint64_t, Record> records;
            for (auto const& tile : snapshot)
            {
                for (std::size_t p = 0; p < tile.count; ++p)
                {
                    Record record;
                    for (std::size_t c = 0; c < tile.reals.size()/tile.count; ++c)
                    {
                        record.first.push_back(tile.reals[c*tile.count+p]);
                    }
                    for (std::size_t c = 0; c < tile.integers.size()/tile.count; ++c)
                    {
                        record.second.push_back(tile.integers[c*tile.count+p]);
                    }
                    AMREX_ALWAYS_ASSERT(records.emplace(tile.ids[p], std::move(record)).second);
                }
            }
            return records;
        };
        auto const lhs = canonical(a);
        auto const rhs = canonical(b);
        if (lhs.size() != rhs.size()) { return false; }
        for (auto const& [id, record] : lhs)
        {
            auto const found = rhs.find(id);
            if (found == rhs.end() || !SameBits(record.first, found->second.first) ||
                !SameBits(record.second, found->second.second)) { return false; }
        }
        return true;
    }

    std::vector<amrex::Real> FieldSnapshot (amrex::MultiFab const& field)
    {
        std::vector<amrex::Real> values;
        for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
        {
            auto const& fab = field[iterator];
            auto const size = static_cast<std::size_t>(fab.box().numPts())*field.nComp();
            auto const offset = values.size();
            values.resize(offset+size);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(), fab.dataPtr()+size,
                             values.data()+offset);
        }
        return values;
    }

    void SumLongDouble (long double* values, int count)
    {
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
#else
        amrex::ignore_unused(values, count);
#endif
    }

    std::array<long double, 3> NativeTotalCp (
        WarpXParticleContainer const& ions, WarpXParticleContainer const& photons,
        std::string const& path)
    {
        std::array<long double, 3> result{};
        auto const& names = ions.GetRealSoANames();
        int carry[3]{};
        std::string const suffix[3] = {"_ux", "_uy", "_uz"};
        for (int d = 0; d < 3; ++d)
        {
            auto const found = std::find(names.begin(), names.end(),
                                         "radiation_impulse_"+path+suffix[d]);
            AMREX_ALWAYS_ASSERT(found != names.end());
            carry[d] = static_cast<int>(std::distance(names.begin(), found));
        }
        for (auto const& tile : Snapshot(ions))
        {
            for (std::size_t p = 0; p < tile.count; ++p)
            {
                for (int d = 0; d < 3; ++d)
                {
                    result[d] += tile.Value(PIdx::w,p)*ions.getMass()*PhysConst::c*
                        (tile.Value(PIdx::ux+d,p)+tile.Value(carry[d],p));
                }
            }
        }
        for (auto const& tile : Snapshot(photons))
        {
            for (std::size_t p = 0; p < tile.count; ++p)
            {
                for (int d = 0; d < 3; ++d)
                {
                    result[d] += tile.Value(PIdx::w,p)*PhysConst::m_e*PhysConst::c*
                        tile.Value(PIdx::ux+d,p);
                }
            }
        }
        SumLongDouble(result.data(), 3);
        return result;
    }

    // At a cosine-return/leakage D/K face, the packet first moment and
    // reconstructed diffusion pressure differ by (2/3-lambda)*Qout*n_D.
    // In 1D Qout*n_D = dt*F(left->right), regardless of which side is D.
    long double InterfaceClosureBudget (amrex::MultiFab const& flux,
        amrex::iMultiFab const& mask, amrex::Geometry const& geometry,
        amrex::Real dt, amrex::Real lambda)
    {
        auto const host_info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        amrex::MultiFab host_flux(flux.boxArray(), flux.DistributionMap(), 1,
                                  flux.nGrowVect(), host_info);
        amrex::iMultiFab host_mask(mask.boxArray(), mask.DistributionMap(), 1,
                                   mask.nGrowVect(), host_info);
        auto const owner = flux.OwnerMask(geometry.periodicity());
        amrex::iMultiFab host_owner(owner->boxArray(), owner->DistributionMap(), 1, 0, host_info);
        amrex::dtoh_memcpy(host_flux, flux);
        amrex::dtoh_memcpy(host_mask, mask);
        amrex::dtoh_memcpy(host_owner, *owner);
        amrex::Gpu::streamSynchronize();
        long double net = 0;
        for (amrex::MFIter iterator(host_flux); iterator.isValid(); ++iterator)
        {
            auto const current = host_flux.const_array(iterator);
            auto const region = host_mask.const_array(iterator);
            auto const owned = host_owner.const_array(iterator);
            for (int face = iterator.validbox().smallEnd(0);
                 face <= iterator.validbox().bigEnd(0); ++face)
            {
                if (owned(face,0,0) && region(face-1,0,0) != region(face,0,0))
                {
                    net += static_cast<long double>(dt)*current(face,0,0);
                }
            }
        }
        SumLongDouble(&net, 1);
        return (2.0L/3-lambda)*net;
    }

    void CheckInterfaceInventories (
        warpx::radiation::PacketDiffusionNativeIntervalResult const& result,
        amrex::iMultiFab const& mask, amrex::Geometry const& geometry, amrex::Real lambda)
    {
        if (!result.valid) {
            AMREX_ALWAYS_ASSERT(!result.interface_crossing_inventory.isDefined());
            AMREX_ALWAYS_ASSERT(!result.interface_leakage_inventory.isDefined());
            AMREX_ALWAYS_ASSERT(!result.interface_traction_model_defect.isDefined());
            return;
        }
        AMREX_ALWAYS_ASSERT(result.interface_crossing_inventory.nComp() == 8);
        AMREX_ALWAYS_ASSERT(result.interface_leakage_inventory.nComp() == 4);
        AMREX_ALWAYS_ASSERT(result.interface_traction_model_defect.nComp() == 3);
        auto copy = [] (auto const& source) {
            std::decay_t<decltype(source)> host(source.boxArray(), source.DistributionMap(),
                source.nComp(), source.nGrowVect(),
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
            amrex::dtoh_memcpy(host, source);
            return host;
        };
        auto cross = copy(result.interface_crossing_inventory);
        auto leakage = copy(result.interface_leakage_inventory);
        auto defect = copy(result.interface_traction_model_defect);
        auto regions = copy(mask);
        auto const owner = result.interface_traction_model_defect.OwnerMask(geometry.periodicity());
        auto owned = copy(*owner);
        amrex::Gpu::streamSynchronize();
        int faces = 0;
        long double total_leakage = 0;
        for (amrex::MFIter mfi(defect); mfi.isValid(); ++mfi) {
            auto const c = cross.const_array(mfi), l = leakage.const_array(mfi);
            auto const b = defect.const_array(mfi);
            auto const region = regions.const_array(mfi), unique = owned.const_array(mfi);
            for (int i = mfi.validbox().smallEnd(0); i <= mfi.validbox().bigEnd(0); ++i) {
                if (!unique(i,0,0)) { continue; }
                int const normal = region(i-1,0,0) - region(i,0,0);
                if (normal == 0) { continue; }
                ++faces;
                long double const incoming = c(i,0,0,0), returned = c(i,0,0,4);
                long double const outgoing = l(i,0,0,0), admitted = incoming-returned;
                long double const scale = std::abs(incoming)+std::abs(returned)+std::abs(outgoing);
                AMREX_ALWAYS_ASSERT(incoming >= 0 && returned >= 0 && outgoing >= 0);
                AMREX_ALWAYS_ASSERT(admitted >= -1.e-11L*scale);
                total_leakage += outgoing;
                for (int d = 0; d < 3; ++d) {
                    long double const expected = d == 2
                        ? (2.0L/3-lambda)*(outgoing-admitted)*normal : -c(i,0,0,d+1);
                    AMREX_ALWAYS_ASSERT(std::abs(b(i,0,0,d)-expected) <= 1.e-11L*scale);
                    long double const cosine = d == 2 ? (2.0L/3)*normal : 0;
                    AMREX_ALWAYS_ASSERT(std::abs(c(i,0,0,d+5)-cosine*returned) <= 1.e-11L*scale);
                    AMREX_ALWAYS_ASSERT(std::abs(l(i,0,0,d+1)-cosine*outgoing) <= 1.e-11L*scale);
                }
            }
        }
        amrex::ParallelDescriptor::ReduceIntSum(faces);
        SumLongDouble(&total_leakage, 1);
        AMREX_ALWAYS_ASSERT(faces == 2 && total_leakage > 0);
    }

    // Independently read native particle attributes; do not use the interval's
    // returned material-work or photon-inventory helper as the energy oracle.
    std::array<long double, 2> MaterialEnergy (WarpXParticleContainer const& ions,
                                              std::string const& path,
                                              bool transparent_kinetic = true)
    {
        std::array<long double, 2> result{};
        auto const& names = ions.GetRealSoANames();
        auto const carry = std::find(names.begin(), names.end(), "radiation_impulse_"+path+"_work");
        AMREX_ALWAYS_ASSERT(carry != names.end());
        int const carry_component = static_cast<int>(std::distance(names.begin(), carry));
        long double const speed = PhysConst::c;
        for (auto const& tile : Snapshot(ions))
        {
            for (std::size_t p = 0; p < tile.count; ++p)
            {
                if (!amrex::ConstParticleIDWrapper(tile.ids[p]).is_valid()) { continue; }
                auto const mass = tile.Value(PIdx::w, p)*ions.getMass();
                long double squared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    auto const u = tile.Value(PIdx::ux+d, p);
                    squared += u*u;
                }
                result[0] += mass*squared/(std::sqrt(1+squared/(speed*speed))+1);
                result[1] += mass*tile.Value(carry_component, p);
                auto const z = tile.Value(PIdx::z, p);
                if (transparent_kinetic && (z < 0.25L || z >= 0.75L))
                {
                    // Transparent K cells receive no real collision impulse.
                    AMREX_ALWAYS_ASSERT(squared == 0 && tile.Value(carry_component, p) == 0);
                }
            }
        }
        SumLongDouble(result.data(), 2);
        return result;
    }

    std::array<long double, 4> CarryChangeAudit (
        WarpXParticleContainer const& ions, std::string const& path,
        std::vector<TileSnapshot> const& before, std::vector<TileSnapshot> const& after)
    {
        auto const& names = ions.GetRealSoANames();
        auto const found = std::find(names.begin(), names.end(), "radiation_impulse_"+path+"_work");
        AMREX_ALWAYS_ASSERT(found != names.end() && before.size() == after.size());
        int const component = static_cast<int>(std::distance(names.begin(), found));
        std::array<long double, 4> result{}; // delta, old absolute work, new absolute work, count
        for (std::size_t t = 0; t < before.size(); ++t) {
            auto const& a = before[t];
            auto const& b = after[t];
            AMREX_ALWAYS_ASSERT(a.key == b.key && a.count == b.count && SameBits(a.ids,b.ids));
            for (std::size_t p = 0; p < a.count; ++p) {
                if (!amrex::ConstParticleIDWrapper(a.ids[p]).is_valid()) { continue; }
                AMREX_ALWAYS_ASSERT(a.Value(PIdx::w,p) == b.Value(PIdx::w,p));
                auto const mass = a.Value(PIdx::w,p)*ions.getMass();
                auto const old = a.Value(component,p), next = b.Value(component,p);
                result[0] += mass*(next-old);
                result[1] += mass*std::abs(old);
                result[2] += mass*std::abs(next);
                result[3] += 1;
            }
        }
        SumLongDouble(result.data(), 4);
        return result;
    }

    long double PhotonEnergy (WarpXParticleContainer const& photons)
    {
        long double result = 0;
        for (auto const& tile : Snapshot(photons))
        {
            for (std::size_t p = 0; p < tile.count; ++p)
            {
                if (!amrex::ConstParticleIDWrapper(tile.ids[p]).is_valid()) { continue; }
                auto const weight = tile.Value(PIdx::w, p);
                long double squared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    auto const u = tile.Value(PIdx::ux+d, p);
                    squared += u*u;
                }
                AMREX_ALWAYS_ASSERT(weight > 0 && squared > 0);
                result += weight*static_cast<long double>(PhysConst::m_e)
                    *static_cast<long double>(PhysConst::c)*std::sqrt(squared);
                auto const z = tile.Value(PIdx::z, p);
                AMREX_ALWAYS_ASSERT(z >= 0 && z < 1 && (z < 0.25L || z >= 0.75L));
            }
        }
        SumLongDouble(&result, 1);
        return result;
    }
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        auto& photons = particles.GetParticleContainerFromName("photons");
        bool kinetic_collisions = false;
        amrex::ParmParse("test").query("kinetic_collisions", kinetic_collisions);
        std::string const path = "packet_diffusion_interval";
        RegisterParticleImpulseState(ions, path);
        simulation.InitData();
        AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == 2);

        auto const& geometry = simulation.Geom(0);
        auto const& boxes = ions.ParticleBoxArray(0);
        auto const& distribution = ions.ParticleDistributionMap(0);
        amrex::MultiFab energy(boxes, distribution, 1, 1);
        amrex::MultiFab opacity(boxes, distribution, 1, 1);
        amrex::MultiFab inertia(boxes, distribution, 3, 0);
        amrex::MultiFab qstore(boxes, distribution, 3, 0);
        qstore.setVal(0);
        bool retain_constitutive = false;
        amrex::ParmParse("test").query("retain_constitutive_momentum", retain_constitutive);
        auto* const conserved_q = retain_constitutive ? &qstore : nullptr;
        amrex::Real const diffusion_opacity = retain_constitutive ? 1024 : 128;
        amrex::MultiFab kinetic_opacity(boxes, distribution, 1, 0);
        amrex::iMultiFab mask(boxes, distribution, 1, 1);
        energy.setVal(0);
        opacity.setVal(0);
        inertia.setVal(0);
        mask.setVal(0);
        kinetic_opacity.setVal(0);
        for (amrex::MFIter iterator(energy); iterator.isValid(); ++iterator)
        {
            auto const e = energy.array(iterator);
            auto const chi = opacity.array(iterator);
            auto const region = mask.array(iterator);
            auto const kinetic_chi = kinetic_opacity.array(iterator);
            amrex::ParallelFor(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                bool const diffusion = i >= 4 && i < 12;
                region(i, j, k) = diffusion ? 1 : 0;
                chi(i, j, k) = diffusion ? diffusion_opacity : 0;
                kinetic_chi(i, j, k) = diffusion ? 0 : 128;
                e(i, j, k) = diffusion ? 1.e6*(1+0.2*(i-4)) : 0;
            });
        }
        energy.FillBoundary(geometry.periodicity());
        opacity.FillBoundary(geometry.periodicity());
        mask.FillBoundary(geometry.periodicity());
        auto const dt = 0.2*geometry.CellSize(0)/PhysConst::c;
        PacketDiffusionNativeIntervalOptions options;
        options.material.max_iterations = 128;
        options.material.impulse_relative_tolerance = 1.e-13;
        options.material.work_relative_tolerance = 1.e-13;
        options.photon_energy = 1.e-16;
        options.energy_relative_tolerance = 1.e-11;
        amrex::ParmParse("test").query("interface_lambda",
                                       options.diffusion.marshak_extrapolation_lambda);
        options.kinetic_scattering_opacity = kinetic_collisions ? &kinetic_opacity : nullptr;
        options.collisions.seed = 42;
        amrex::ParmParse("test").query("proper_rates", options.collisions.moving_cross_sections);
        amrex::ParmParse("test").query("quasi_static_angular_traction",
                                       options.quasi_static_angular_traction);
        options.retain_constitutive_momentum = retain_constitutive;
        AMREX_ALWAYS_ASSERT(!retain_constitutive || options.quasi_static_angular_traction);
        std::uint64_t interval_counter = 0;

        // Malformed callers must reject collectively before constructors or
        // particle access that assume a valid native layout/schema.
        auto const admission_ions = Snapshot(ions);
        auto const admission_photons = Snapshot(photons);
        auto const admission_energy = FieldSnapshot(energy);
        auto const admission_inertia = FieldSnapshot(inertia);
        auto const admission_q = FieldSnapshot(qstore);
        for (auto const& names : {std::vector<std::string>{"ions", "ions"},
                                 std::vector<std::string>{"missing_species"},
                                 std::vector<std::string>{"photons"}})
        {
            auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                names, path, energy, opacity, mask, geometry, dt, &inertia, options, conserved_q);
            AMREX_ALWAYS_ASSERT(!rejected.valid);
            CheckInterfaceInventories(rejected, mask, geometry, options.diffusion.marshak_extrapolation_lambda);
        }
        auto const unregistered = TryAdvancePacketDiffusionNativeInterval(particles, photons,
            {"ions"}, "unregistered_path", energy, opacity, mask, geometry, dt, &inertia, options, conserved_q);
        AMREX_ALWAYS_ASSERT(!unregistered.valid);
        CheckInterfaceInventories(unregistered, mask, geometry, options.diffusion.marshak_extrapolation_lambda);
        for (int invalid_control = 0; invalid_control < 4; ++invalid_control) {
            auto guarded = options;
            guarded.quasi_static_angular_traction = true;
            if (invalid_control < 2) {
                guarded.quasi_static_maximum_beta = invalid_control == 0 ? 0 : 2.e-3;
            } else if (invalid_control == 2) {
                guarded.quasi_static_minimum_optical_depth = diffusion_opacity*geometry.CellSize(0)+1;
            } else {
                // Valid control, but the staged physical kick exceeds this
                // deliberately tiny secant-speed bound: reject during replay.
                guarded.quasi_static_maximum_beta = 1.e-20;
            }
            auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, guarded, conserved_q);
            AMREX_ALWAYS_ASSERT(!rejected.valid);
            CheckInterfaceInventories(rejected, mask, geometry,
                options.diffusion.marshak_extrapolation_lambda);
        }
        if (retain_constitutive)
        {
            amrex::MultiFab shared(energy.boxArray(), energy.DistributionMap(), 4, 1);
            shared.setVal(0);
            amrex::MultiFab energy_alias(shared, amrex::make_alias, 0, 1);
            amrex::MultiFab q_alias(shared, amrex::make_alias, 0, 3);
            amrex::MultiFab::Copy(energy_alias, energy, 0, 0, 1, 0);
            auto const shared_before = FieldSnapshot(shared);
            auto const alias_rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy_alias, opacity, mask, geometry, dt, &inertia,
                options, &q_alias);
            AMREX_ALWAYS_ASSERT(!alias_rejected.valid);
            AMREX_ALWAYS_ASSERT(SameBits(shared_before, FieldSnapshot(shared)));
            for (int invalid_state = 0; invalid_state < 4; ++invalid_state)
            {
                auto guarded = options;
                auto* target_q = conserved_q;
                if (invalid_state == 0) {
                    guarded.minimum_constitutive_relaxation_depth =
                        diffusion_opacity*PhysConst::c*dt+1;
                } else if (invalid_state == 1) {
                    target_q = nullptr;
                } else if (invalid_state == 2) {
                    target_q = &inertia; // Physical q cannot alias legacy diagnostic inertia.
                } else {
                    guarded.quasi_static_angular_traction = false;
                }
                auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                    {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, guarded, target_q);
                AMREX_ALWAYS_ASSERT(!rejected.valid);
                AMREX_ALWAYS_ASSERT(SameBits(admission_q, FieldSnapshot(qstore)));
            }
            // A physical q inventory cannot exceed its cell energy. Reject this
            // malformed caller state without changing even the malformed data.
            for (amrex::MFIter mfi(qstore); mfi.isValid(); ++mfi) {
                auto const q = qstore.array(mfi);
                auto const e = energy.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        q(i,j,k,0) = 2*e(i,j,k);
                    });
            }
            auto const malformed_q = FieldSnapshot(qstore);
            auto const cone_rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, options, conserved_q);
            AMREX_ALWAYS_ASSERT(!cone_rejected.valid);
            AMREX_ALWAYS_ASSERT(SameBits(malformed_q, FieldSnapshot(qstore)));
            qstore.setVal(0);
        }
        else
        {
            auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, options, &qstore);
            AMREX_ALWAYS_ASSERT(!rejected.valid); // Pointer without model opt-in.
        }
        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            auto inconsistent = options;
            if (amrex::ParallelDescriptor::MyProc() == 1) { inconsistent.photon_energy *= 2; }
            auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, inconsistent, conserved_q);
            AMREX_ALWAYS_ASSERT(!rejected.valid);
            CheckInterfaceInventories(rejected, mask, geometry, options.diffusion.marshak_extrapolation_lambda);
            inconsistent = options;
            if (amrex::ParallelDescriptor::MyProc() == 1) {
                inconsistent.quasi_static_angular_traction = !options.quasi_static_angular_traction;
            }
            auto const mode_rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, inconsistent, conserved_q);
            AMREX_ALWAYS_ASSERT(!mode_rejected.valid);
            CheckInterfaceInventories(mode_rejected, mask, geometry,
                options.diffusion.marshak_extrapolation_lambda);
        }
        AMREX_ALWAYS_ASSERT(SameParticles(admission_ions, Snapshot(ions)));
        AMREX_ALWAYS_ASSERT(SameParticles(admission_photons, Snapshot(photons)));
        AMREX_ALWAYS_ASSERT(SameBits(admission_energy, FieldSnapshot(energy)));
        AMREX_ALWAYS_ASSERT(SameBits(admission_inertia, FieldSnapshot(inertia)));
        AMREX_ALWAYS_ASSERT(SameBits(admission_q, FieldSnapshot(qstore)));

        auto reject_unchanged = [&] ()
        {
            auto const old_ions = Snapshot(ions);
            auto const old_photons = Snapshot(photons);
            auto const old_energy = FieldSnapshot(energy);
            auto const old_inertia = FieldSnapshot(inertia);
            auto const old_conserved_q = FieldSnapshot(qstore);
            auto reject_options = options;
            reject_options.material.max_iterations = 1;
            auto const rejected = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, reject_options, conserved_q);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!rejected.valid,
                "One iteration must reject the finite D-work feedback in this fixture");
            CheckInterfaceInventories(rejected, mask, geometry, options.diffusion.marshak_extrapolation_lambda);
            AMREX_ALWAYS_ASSERT(rejected.reason.find("Finite-mass work iteration") !=
                                std::string::npos);
            AMREX_ALWAYS_ASSERT(SameParticles(old_ions, Snapshot(ions)));
            AMREX_ALWAYS_ASSERT(SameParticles(old_photons, Snapshot(photons)));
            AMREX_ALWAYS_ASSERT(SameBits(old_energy, FieldSnapshot(energy)));
            AMREX_ALWAYS_ASSERT(SameBits(old_inertia, FieldSnapshot(inertia)));
            AMREX_ALWAYS_ASSERT(SameBits(old_conserved_q, FieldSnapshot(qstore)));
        };
        reject_unchanged();

        auto const initial_material = MaterialEnergy(ions, path, !kinetic_collisions);
        auto const initial_radiation = PhotonEnergy(photons)+energy.sum(0);
        AMREX_ALWAYS_ASSERT(initial_material[0] == 0 && initial_material[1] == 0);
        long double accumulated_work = 0;
        long double accumulated_carry = 0;
        amrex::Real maximum_energy_error = 0;
        amrex::Real model_residual_norm = 0;
        long double boundary_model_budget = 0;
        long double maximum_unexplained_momentum = 0;
        long double tangential_model_budget = 0;
        amrex::Long collision_events = 0;
        for (int step = 0; step < 8; ++step)
        {
            options.collisions.interval = interval_counter;
            if (step == 3) { reject_unchanged(); }
            auto const before_material = MaterialEnergy(ions, path, !kinetic_collisions);
            auto const before_carry_particles = Snapshot(ions);
            auto const before_radiation = PhotonEnergy(photons)+energy.sum(0);
            auto const before_cp = NativeTotalCp(ions, photons, path);
            amrex::MultiFab before_q(boxes, distribution, 3, 0);
            amrex::MultiFab::Copy(before_q, qstore, 0, 0, 3, 0);
            std::array<long double, 3> before_qsum{};
            for (int d = 0; d < 3; ++d) { before_qsum[d] = qstore.sum(d); }
            std::array<long double,3> old_q{};
            for (int d = 0; d < 3; ++d) { old_q[d] = inertia.sum(d); }
            auto result = TryAdvancePacketDiffusionNativeInterval(particles, photons,
                {"ions"}, path, energy, opacity, mask, geometry, dt, &inertia, options, conserved_q);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(result.valid, result.reason.c_str());
            if (options.quasi_static_angular_traction) {
                AMREX_ALWAYS_ASSERT(result.used_quasi_static_angular_traction);
                AMREX_ALWAYS_ASSERT(result.used_constitutive_momentum == retain_constitutive);
                CheckQuasiStaticNativeStress(result, mask, geometry, dt,
                    retain_constitutive ? &before_q : nullptr, conserved_q);
            }
            CheckInterfaceInventories(result, mask, geometry, options.diffusion.marshak_extrapolation_lambda);
            AMREX_ALWAYS_ASSERT(result.work_iterations >= 1);
            AMREX_ALWAYS_ASSERT(step != 0 || result.work_iterations > 1);
            auto const after_material = MaterialEnergy(ions, path, !kinetic_collisions);
            auto const after_radiation = PhotonEnergy(photons)+energy.sum(0);
            auto const after_cp = NativeTotalCp(ions, photons, path);
            auto const actual_work = after_material[0]-before_material[0];
            auto const carry_change = after_material[1]-before_material[1];
            auto const carry_audit = CarryChangeAudit(ions, path, before_carry_particles,
                Snapshot(ions));
            AMREX_ALWAYS_ASSERT(step != 0 || actual_work > 0);
            AMREX_ALWAYS_ASSERT(std::abs(actual_work-result.actual_material_work) <=
                1.e-11L*(std::abs(actual_work)+std::abs(result.actual_material_work)));
            // Native mass/difference/product, cell and rank sums use Real;
            // the independent paired-particle reference and positive norm use
            // long double. Signed global carry can cancel even in that oracle.
            auto gamma = [] (long double operations, long double epsilon) {
                auto const product = operations*epsilon;
                AMREX_ALWAYS_ASSERT(product >= 0 && product < 0.125L);
                return product/(1-product);
            };
            auto const n = carry_audit[3];
            auto const ranks = amrex::ParallelDescriptor::NProcs();
            auto const eps = std::numeric_limits<long double>::epsilon();
            auto const native_error = gamma(2*n+ranks+8, std::numeric_limits<amrex::Real>::epsilon());
            auto const reference_error = gamma(n+ranks+6, eps);
            auto const norm_error = gamma(n+ranks+4, eps);
            auto const norm = carry_audit[1]+carry_audit[2];
            auto const bound = norm == 0 ? 0 : std::nextafter(
                (native_error+reference_error)*norm/(1-norm_error)*(1+8*eps),
                std::numeric_limits<long double>::infinity());
            auto const physical_term = 1.e-11L*(std::abs(after_material[1])+
                std::abs(before_material[1])+std::abs(result.material_energy_carry_change));
            AMREX_ALWAYS_ASSERT(std::isfinite(bound) && bound >= 0);
            AMREX_ALWAYS_ASSERT(std::abs(carry_audit[0]-result.material_energy_carry_change) <=
                physical_term+bound);
            if (physical_term+bound > 0) {
                // A missing signed carry well beyond the computed arithmetic
                // allowance must still fail this reporting-consistency gate.
                auto const faulty = result.material_energy_carry_change+100*(physical_term+bound);
                AMREX_ALWAYS_ASSERT(std::abs(carry_audit[0]-faulty) > physical_term+bound);
            }
            auto const defect = after_radiation-before_radiation+actual_work+carry_change;
            auto const scale = std::abs(before_radiation)+std::abs(after_radiation)
                +std::abs(actual_work)+std::abs(carry_change);
            AMREX_ALWAYS_ASSERT(std::abs(defect) <= 2.e-11L*scale);
            maximum_energy_error = std::max(maximum_energy_error,
                static_cast<amrex::Real>(std::abs(defect)/scale));
            auto const boundary_term = InterfaceClosureBudget(*result.field.face_current[0],
                mask, geometry, dt, options.diffusion.marshak_extrapolation_lambda);
            boundary_model_budget += std::abs(boundary_term);
            for (int d = 0; d < 3; ++d)
            {
                auto const measured = after_cp[d]-before_cp[d]+
                    result.field.radiation_inertia.sum(d)-old_q[d];
                auto const predicted = result.field.closure_diagnostic.sum(d)+
                    (d == 2 ? boundary_term : 0)+
                    PhysConst::c*result.interface_tangential_model_defect[d];
                tangential_model_budget +=
                    std::abs(PhysConst::c*result.interface_tangential_model_defect[d]);
                // Retained physical q is distinct from diagnostic radiation inertia.
                auto const physical_q_change = retain_constitutive
                    ? static_cast<long double>(qstore.sum(d))-before_qsum[d] : 0;
                auto const unexplained = options.quasi_static_angular_traction
                    ? std::abs(after_cp[d]-before_cp[d]+physical_q_change)/scale
                    : std::abs(measured-predicted)/scale;
                AMREX_ALWAYS_ASSERT(unexplained <= 2.e-11L);
                maximum_unexplained_momentum = std::max(maximum_unexplained_momentum, unexplained);
            }
            for (auto const diagnostic : result.momentum_model_residual)
            {
                AMREX_ALWAYS_ASSERT(std::isfinite(diagnostic));
                model_residual_norm += std::abs(diagnostic);
            }
            AMREX_ALWAYS_ASSERT(energy.min(0) >= 0);
            amrex::MultiFab::Copy(inertia, result.field.radiation_inertia, 0, 0, 3, 0);
            accumulated_work += result.actual_material_work;
            accumulated_carry += result.material_energy_carry_change;
            collision_events += result.collision_events;
            ++interval_counter;
            if (step == 3)
            {
                // Native particle and field persistence, not an in-memory
                // clone: resume the accepted mixed state after disk I/O.
                auto const saved_ions = Snapshot(ions);
                auto const saved_photons = Snapshot(photons);
                auto const saved_energy = FieldSnapshot(energy);
                auto const saved_inertia = FieldSnapshot(inertia);
                auto const saved_q = FieldSnapshot(qstore);
                auto const saved_retain = options.retain_constitutive_momentum;
                auto const saved_relaxation = options.minimum_constitutive_relaxation_depth;
                auto const saved_mode = options.quasi_static_angular_traction;
                auto const saved_lambda = options.diffusion.marshak_extrapolation_lambda;
                auto const saved_beta = options.quasi_static_maximum_beta;
                auto const saved_tau = options.quasi_static_minimum_optical_depth;
                std::string const checkpoint = "native_interval_checkpoint";
                ions.Checkpoint(checkpoint, "ions");
                photons.Checkpoint(checkpoint, "photons");
                amrex::VisMF::Write(energy, checkpoint+"/energy");
                amrex::VisMF::Write(inertia, checkpoint+"/inertia");
                if (retain_constitutive) { amrex::VisMF::Write(qstore, checkpoint+"/conserved_q"); }
                if (amrex::ParallelDescriptor::IOProcessor())
                {
                    std::ofstream state(checkpoint+"/collision_state");
                    // Source-API controls are caller-owned, not WarpX runtime metadata.
                    state << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
                          << "native_collision_interval_v3 " << options.collisions.seed
                          << ' ' << interval_counter << ' ' << saved_mode << ' ' << saved_lambda
                          << ' ' << saved_beta << ' ' << saved_tau << ' ' << saved_retain
                          << ' ' << saved_relaxation << '\n';
                    state.flush();
                    AMREX_ALWAYS_ASSERT(state.good());
                }
                amrex::ParallelDescriptor::Barrier();
                ions.clearParticles();
                photons.clearParticles();
                ions.Restart(checkpoint, "ions");
                photons.Restart(checkpoint, "photons");
                ions.Redistribute();
                photons.Redistribute();
                energy.setVal(-1);
                inertia.setVal(-1);
                amrex::VisMF::Read(energy, checkpoint+"/energy");
                amrex::VisMF::Read(inertia, checkpoint+"/inertia");
                if (retain_constitutive) {
                    qstore.setVal(-1);
                    amrex::VisMF::Read(qstore, checkpoint+"/conserved_q");
                }
                interval_counter = 0;
                options.collisions.seed = 0;
                options.quasi_static_angular_traction = !saved_mode;
                options.diffusion.marshak_extrapolation_lambda = -1;
                options.quasi_static_maximum_beta = -1;
                options.quasi_static_minimum_optical_depth = -1;
                options.retain_constitutive_momentum = !saved_retain;
                options.minimum_constitutive_relaxation_depth = -1;
                std::ifstream state(checkpoint+"/collision_state");
                std::string version;
                AMREX_ALWAYS_ASSERT(static_cast<bool>(state >> version >> options.collisions.seed
                    >> interval_counter >> options.quasi_static_angular_traction
                    >> options.diffusion.marshak_extrapolation_lambda
                    >> options.quasi_static_maximum_beta >> options.quasi_static_minimum_optical_depth
                    >> options.retain_constitutive_momentum
                    >> options.minimum_constitutive_relaxation_depth));
                AMREX_ALWAYS_ASSERT(version == "native_collision_interval_v3" &&
                    options.collisions.seed == 42 && interval_counter == 4);
                AMREX_ALWAYS_ASSERT(options.quasi_static_angular_traction == saved_mode &&
                    options.diffusion.marshak_extrapolation_lambda == saved_lambda &&
                    options.quasi_static_maximum_beta == saved_beta &&
                    options.quasi_static_minimum_optical_depth == saved_tau &&
                    options.retain_constitutive_momentum == saved_retain &&
                    options.minimum_constitutive_relaxation_depth == saved_relaxation);
                AMREX_ALWAYS_ASSERT(SameCheckpointParticles(saved_ions, Snapshot(ions)));
                AMREX_ALWAYS_ASSERT(SameCheckpointParticles(saved_photons, Snapshot(photons)));
                AMREX_ALWAYS_ASSERT(SameBits(saved_energy, FieldSnapshot(energy)));
                AMREX_ALWAYS_ASSERT(SameBits(saved_inertia, FieldSnapshot(inertia)));
                AMREX_ALWAYS_ASSERT(SameBits(saved_q, FieldSnapshot(qstore)));
            }
        }
        AMREX_ALWAYS_ASSERT(!kinetic_collisions || collision_events > 0);
        auto const final_material = MaterialEnergy(ions, path, !kinetic_collisions);
        auto const final_radiation = PhotonEnergy(photons)+energy.sum(0);
        AMREX_ALWAYS_ASSERT(final_material[0] > 0 && photons.TotalNumberOfParticles() > 2);
        AMREX_ALWAYS_ASSERT(std::abs(final_material[0]-initial_material[0]-accumulated_work) <=
            1.e-11L*(std::abs(final_material[0])+std::abs(accumulated_work)));
        AMREX_ALWAYS_ASSERT(std::abs(final_radiation-initial_radiation+final_material[0]
            -initial_material[0]+final_material[1]-initial_material[1]) <=
            2.e-11L*(std::abs(final_radiation)+std::abs(initial_radiation)));
        amrex::Print() << "Static native D/K source: quasi-static angular traction="
                       << options.quasi_static_angular_traction
                       << ", retained constitutive momentum=" << retain_constitutive
                       << ", 8 intervals with native checkpoint, actual ion work="
                       << static_cast<amrex::Real>(accumulated_work)
                       << ", carry=" << static_cast<amrex::Real>(accumulated_carry)
                       << ", energy residual=" << maximum_energy_error
                       << ", kinetic collisions=" << collision_events
                       << ", boundary model budget[J]=" << static_cast<double>(boundary_model_budget)
                       << ", unresolved tangential budget[J]=" << static_cast<double>(tangential_model_budget)
                       << ", unexplained momentum/E=" << static_cast<double>(maximum_unexplained_momentum)
                       << ", omitted-inertia/interface diagnostic=" << model_residual_norm << '\n';
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}

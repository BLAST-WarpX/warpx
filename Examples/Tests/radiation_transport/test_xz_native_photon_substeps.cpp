/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonPathReplay.H"
#include "Radiation/NativePhotonReplayInternals.H"
#include "Radiation/PhotonFourMomentum.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace amrex::literals;

namespace
{
struct PhotonState
{
    std::array<amrex::Real, 7> real{};
    std::uint64_t idcpu = 0;
    int marker = 0;
    int owner = 0;
    int box = 0;
};

PhotonState
ReadOne (WarpXParticleContainer::Base const& photons,
         int real_sentinel, int int_sentinel)
{
    PhotonState result;
    int local_count = 0;
    std::array<int, 7> const components{
        PIdx::x, PIdx::z, PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz,
        real_sentinel};
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        auto const count = tile.numParticles();
        AMREX_ALWAYS_ASSERT(count <= 1);
        if (count == 0) { continue; }
        ++local_count;
        auto const& soa = tile.GetStructOfArrays();
        for (int c = 0; c < 7; ++c)
        {
            amrex::ParticleReal value = 0;
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                soa.GetRealData(components[c]).begin(),
                soa.GetRealData(components[c]).begin()+1, &value);
            result.real[c] = value;
        }
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIdCPUData().begin(), soa.GetIdCPUData().begin()+1,
            &result.idcpu);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIntData(int_sentinel).begin(),
            soa.GetIntData(int_sentinel).begin()+1, &result.marker);
        result.owner = amrex::ParallelDescriptor::MyProc()+1;
        result.box = key.first+1;
    }
    amrex::ParallelDescriptor::ReduceIntSum(local_count);
    AMREX_ALWAYS_ASSERT(local_count == 1);
    amrex::ParallelDescriptor::ReduceRealSum(result.real.data(), 7);
    amrex::ParallelDescriptor::ReduceIntSum(result.marker);
    amrex::ParallelDescriptor::ReduceIntSum(result.owner);
    amrex::ParallelDescriptor::ReduceIntSum(result.box);
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result.idcpu, 1, MPI_UINT64_T, MPI_BOR,
        amrex::ParallelDescriptor::Communicator());
#endif
    --result.owner;
    --result.box;
    return result;
}

void
KeepOneDiagonalPhoton (WarpXParticleContainer& photons, amrex::Real x,
                       amrex::Real z, int real_sentinel, int int_sentinel)
{
    auto const diagonal = static_cast<amrex::ParticleReal>(
        PhysConst::c/std::sqrt(2.0_rt));
    for (auto& [key, tile] : photons.GetParticles(0))
    {
        auto const count = tile.numParticles();
        if (count == 0) { continue; }
        auto& soa = tile.GetStructOfArrays();
        std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
            soa.GetIdCPUData().begin(), soa.GetIdCPUData().begin()+count,
            ids.begin());
        for (long p = 0; p < count; ++p)
        {
            if (key.first != 0 || key.second != 0 || p != 0)
            {
                ids[static_cast<std::size_t>(p)] =
                    amrex::ParticleIdCpus::Invalid;
            }
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
            ids.begin(), ids.end(), soa.GetIdCPUData().begin());
        auto const put = [&] (int component, amrex::ParticleReal value)
        {
            std::vector<amrex::ParticleReal> values(
                static_cast<std::size_t>(count), value);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                values.begin(), values.end(),
                soa.GetRealData(component).begin());
        };
        put(PIdx::x, static_cast<amrex::ParticleReal>(x));
        put(PIdx::z, static_cast<amrex::ParticleReal>(z));
        put(PIdx::w, 1.0_prt);
        put(PIdx::ux, diagonal);
        put(PIdx::uy, 0.0_prt);
        put(PIdx::uz, diagonal);
        put(real_sentinel, 19.5_prt);
        std::vector<int> marker(static_cast<std::size_t>(count), 37);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
            marker.begin(), marker.end(),
            soa.GetIntData(int_sentinel).begin());
    }
    photons.Redistribute();
    AMREX_ALWAYS_ASSERT(photons.TotalNumberOfParticles() == 1);
}

struct LaunchState
{
    amrex::Real x = 0;
    amrex::Real z = 0;
    amrex::Real ux = 0;
    amrex::Real uz = 0;
};

void
SetLaunchState (WarpXParticleContainer::Base& photons,
                LaunchState const& state)
{
    for (auto& [key, tile] : photons.GetParticles(0))
    {
        amrex::ignore_unused(key);
        if (tile.numParticles() == 0) { continue; }
        AMREX_ALWAYS_ASSERT(tile.numParticles() == 1);
        auto& soa = tile.GetStructOfArrays();
        std::array<int, 4> const components{
            PIdx::x, PIdx::z, PIdx::ux, PIdx::uz};
        std::array<amrex::Real, 4> const values{
            state.x, state.z, state.ux, state.uz};
        for (int c = 0; c < 4; ++c)
        {
            auto const value = static_cast<amrex::ParticleReal>(values[c]);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                &value, &value+1, soa.GetRealData(components[c]).begin());
        }
    }
    photons.Redistribute();
}

struct PartitionResult
{
    PhotonState photon;
    int initial_owner = -1;
    int initial_box = -1;
    warpx::radiation::NativePhotonPathCarry carry;
    amrex::Long events = 0;
    amrex::Long faces = 0;
    std::vector<PhotonState> segment_states;
    std::vector<amrex::Long> face_prefix;
    std::vector<amrex::Real> depth_after_segment;
};

struct EventResult
{
    PhotonState photon;
    warpx::radiation::NativePhotonPathCarry carry;
    amrex::Long events = 0;
    amrex::Long faces = 0;
    std::vector<amrex::Long> event_prefix;
    std::vector<PhotonState> segment_states;
    std::unique_ptr<amrex::MultiFab> impulse;
    std::unique_ptr<amrex::MultiFab> debit;
    std::unique_ptr<amrex::MultiFab> scale;
};

amrex::Real
CellValue (amrex::MultiFab const& field, int i, int k, int component)
{
    amrex::IntVect const cell(AMREX_D_DECL(i, k, 0));
    amrex::Real value = 0;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        if (!iterator.validbox().contains(cell)) { continue; }
        auto const& fab = field[iterator];
        auto const box = iterator.fabbox();
        std::vector<amrex::Real> host(
            static_cast<std::size_t>(box.numPts())*field.nComp());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(),
            fab.dataPtr()+host.size(), host.begin());
        value = host[static_cast<std::size_t>(
            box.index(cell)+box.numPts()*component)];
    }
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}

amrex::Real
Bound (amrex::Real a, amrex::Real b, amrex::Real scale)
{
    // Every comparison here has at most four segments, two events and eight
    // face snaps. Each segment contributes one c*dt product and at most two
    // coordinate multiply/add pairs; the event cases add only a few dozen
    // Lorentz, native-conversion and ledger operations per compared scalar.
    // The staged/frozen speeds have |beta| <= sqrt(2)/8, so gamma < 1.017
    // and 1-beta.n > 0.82: no near-singular boost enters this fixture.
    // 512 eps conservatively covers those operation chains at each operand's
    // physical scale. Event, cell and owner topology are asserted exactly.
    return 512*std::numeric_limits<amrex::Real>::epsilon()*
        std::max({std::abs(a), std::abs(b), std::abs(scale)});
}

void
CheckAnisotropicSubsteps (WarpXParticleContainer& photons,
                          amrex::Geometry const& geometry,
                          int real_sentinel, int int_sentinel)
{
    using namespace warpx::radiation;
    auto const dx_x = geometry.CellSize(0);
    auto const dx_z = geometry.CellSize(1);
    AMREX_ALWAYS_ASSERT(dx_x != dx_z);
    auto const narrow_x = dx_x < dx_z;
    auto const narrow_dx = std::min(dx_x, dx_z);
    auto const start_x = geometry.ProbLo(0)+dx_x/4;
    auto const start_z = geometry.ProbLo(1)+dx_z/4;
    KeepOneDiagonalPhoton(photons, start_x, start_z,
                          real_sentinel, int_sentinel);
    SetLaunchState(photons, LaunchState{
        start_x, start_z,
        narrow_x ? PhysConst::c : 0,
        narrow_x ? 0 : PhysConst::c});
    auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
    auto const& boxes = photons.ParticleBoxArray(0);
    auto const& distribution = photons.ParticleDistributionMap(0);
    amrex::MultiFab opacity(boxes, distribution, 1, 0);
    amrex::MultiFab velocity(boxes, distribution, 3, 0);
    opacity.setVal(0);
    velocity.setVal(0);
    NativePhotonCollisionOptions options;
    options.seed = 91;
    options.interval = 37;
    options.max_events_per_photon = 64;
    options.max_optical_depth = 16;
    NativePhotonPathReport prepared;
    NativePhotonPathReplay primitive;
    AMREX_ALWAYS_ASSERT(!primitive.Prepare(photons,
        opacity, velocity, geometry,
        0.6_rt*narrow_dx/PhysConst::c, options, prepared));
    AMREX_ALWAYS_ASSERT(primitive.Prepare(photons,
        opacity, velocity, geometry,
        0.49_rt*narrow_dx/PhysConst::c, options, prepared));

    // 1.36*min(dx) requires three segments. The larger axis alone would
    // allow two, so the collective limit-2 rejection tests min-axis sizing.
    auto const outer_dt = 1.36_rt*narrow_dx/PhysConst::c;
    NativePhotonPathSubsegmentReplay helper;
    AMREX_ALWAYS_ASSERT(!helper.Prepare(photons,
        opacity, velocity, geometry, outer_dt, options, 2, prepared));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helper.Prepare(photons,
        opacity, velocity, geometry, outer_dt, options, 4, prepared),
        prepared.reason.c_str());
    PhotonPacketTransaction trial(photons);
    amrex::MultiFab impulse(boxes, distribution, 3, 0);
    amrex::MultiFab debit(boxes, distribution, 1, 0);
    amrex::MultiFab scale(boxes, distribution, 1, 0);
    NativePhotonPathSubsegmentReport report;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helper.Replay(velocity,
        trial.Particles(), impulse, debit, scale, report),
        report.path.reason.c_str());
    AMREX_ALWAYS_ASSERT(report.subsegments == 3);
    AMREX_ALWAYS_ASSERT(report.path.collision_events == 0);
    AMREX_ALWAYS_ASSERT(report.path.face_crossings == 1);
    for (int c = 0; c < 3; ++c)
    {
        AMREX_ALWAYS_ASSERT(impulse.norm0(c) == 0);
        AMREX_ALWAYS_ASSERT(velocity.norm0(c) == 0);
    }
    AMREX_ALWAYS_ASSERT(debit.norm0() == 0);
    AMREX_ALWAYS_ASSERT(scale.norm0() == 0);
    AMREX_ALWAYS_ASSERT(opacity.norm0() == 0);
    auto const final = ReadOne(trial.Particles(),
                               real_sentinel, int_sentinel);
    auto const expected_x =
        start_x+(narrow_x ? PhysConst::c*outer_dt : 0);
    auto const expected_z =
        start_z+(narrow_x ? 0 : PhysConst::c*outer_dt);
    AMREX_ALWAYS_ASSERT(std::abs(final.real[0]-expected_x) <=
                        Bound(final.real[0], expected_x, dx_x));
    AMREX_ALWAYS_ASSERT(std::abs(final.real[1]-expected_z) <=
                        Bound(final.real[1], expected_z, dx_z));
    AMREX_ALWAYS_ASSERT(final.real[2] == 1.0_rt);
    AMREX_ALWAYS_ASSERT(final.real[6] == 19.5_rt);
    AMREX_ALWAYS_ASSERT(final.marker == 37);
    amrex::IntVect const expected_cell(
        AMREX_D_DECL(narrow_x ? 1 : 0, narrow_x ? 0 : 1, 0));
    int expected_box = -1;
    for (int index = 0; index < boxes.size(); ++index)
    {
        if (boxes[index].contains(expected_cell)) { expected_box = index; }
    }
    AMREX_ALWAYS_ASSERT(expected_box >= 0);
    AMREX_ALWAYS_ASSERT(final.box == expected_box);
    AMREX_ALWAYS_ASSERT(final.owner == distribution[expected_box]);
    int found = 0;
    int next_draw = 0;
    int events = 0;
    for (auto const& carry : report.photon_carries)
    {
        if (carry.idcpu == initial.idcpu)
        {
            ++found;
            next_draw = carry.next_draw;
            events = carry.completed_events;
            AMREX_ALWAYS_ASSERT(carry.residual_optical_depth > 0);
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(found);
    amrex::ParallelDescriptor::ReduceIntSum(next_draw);
    amrex::ParallelDescriptor::ReduceIntSum(events);
    AMREX_ALWAYS_ASSERT(found == 1 && next_draw == 1 && events == 0);
    auto const live = ReadOne(photons, real_sentinel, int_sentinel);
    AMREX_ALWAYS_ASSERT(live.real == initial.real);
    AMREX_ALWAYS_ASSERT(live.owner == initial.owner);
    amrex::Print() << "XZ anisotropic smaller-axis substeps passed: "
        << "dx_x=" << dx_x << " dx_z=" << dx_z << "\n";
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& photons = simulation.GetPartContainer().
            GetParticleContainerFromName("photons");
        photons.AddRealComp("xz_path_real_sentinel", 1);
        photons.AddIntComp("xz_path_int_sentinel", 1);
        simulation.InitData();
        auto const real_sentinel =
            photons.GetRealCompIndex("xz_path_real_sentinel");
        auto const int_sentinel =
            photons.GetIntCompIndex("xz_path_int_sentinel");
        auto const& geometry = simulation.Geom(0);
        amrex::ParmParse xz_parameters("xz_test");
        int anisotropic = 0;
        xz_parameters.query("anisotropic", anisotropic);
        if (anisotropic != 0)
        {
            CheckAnisotropicSubsteps(photons, geometry,
                                      real_sentinel, int_sentinel);
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        auto const dx = geometry.CellSize(0);
        AMREX_ALWAYS_ASSERT(dx == geometry.CellSize(1));
        auto const initial_x = -dx/64;
        auto const initial_z = -dx/64;
        KeepOneDiagonalPhoton(photons, initial_x, initial_z,
                              real_sentinel, int_sentinel);
        auto const initial = ReadOne(photons, real_sentinel, int_sentinel);
        auto const& boxes = photons.ParticleBoxArray(0);
        auto const& distribution = photons.ParticleDistributionMap(0);
        amrex::MultiFab opacity(boxes, distribution, 1, 0);
        amrex::MultiFab velocity(boxes, distribution, 3, 0);
        opacity.setVal(0);
        velocity.setVal(0);
        NativePhotonCollisionOptions options;
        options.seed = 91;
        options.interval = 13;
        options.max_events_per_photon = 64;
        options.max_optical_depth = 16;
        options.angular_law =
            NativePhotonCollisionAngularLaw::FixedFrameDirection;
        options.fixed_frame_direction = {0, 0, -1};
        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            NativePhotonPathReplay invalid;
            NativePhotonPathReport rejected;
            auto const local_dt = amrex::ParallelDescriptor::MyProc() == 0 ?
                std::numeric_limits<amrex::Real>::quiet_NaN() :
                dx/(4*PhysConst::c);
            AMREX_ALWAYS_ASSERT(!invalid.Prepare(photons,
                opacity, velocity, geometry, local_dt, options, rejected));
            int const local_components =
                amrex::ParallelDescriptor::MyProc() == 0 ? 2 : 1;
            amrex::MultiFab invalid_opacity(
                boxes, distribution, local_components, 0);
            invalid_opacity.setVal(0);
            AMREX_ALWAYS_ASSERT(!invalid.Prepare(photons,
                invalid_opacity, velocity, geometry,
                dx/(4*PhysConst::c), options, rejected));
        }

        auto const run = [&] (amrex::Real outer_dt,
                             std::vector<amrex::Real> const& fractions,
                             LaunchState const* launch = nullptr)
        {
            PhotonPacketTransaction trial(photons);
            if (launch != nullptr)
            {
                SetLaunchState(trial.Particles(), *launch);
            }
            PartitionResult result;
            auto const launch_state = ReadOne(trial.Particles(),
                                              real_sentinel, int_sentinel);
            result.initial_owner = launch_state.owner;
            result.initial_box = launch_state.box;
            std::vector<NativePhotonPathCarry> carries;
            for (std::size_t segment = 0; segment < fractions.size(); ++segment)
            {
                NativePhotonPathReplay path;
                NativePhotonPathReport report;
                auto const dt = outer_dt*fractions[segment];
                bool const prepared = segment == 0 ? path.Prepare(
                    trial.Particles(), photons, opacity, velocity,
                    geometry, dt, options, report) : path.Prepare(
                    trial.Particles(), photons, opacity, velocity,
                    geometry, dt, options, carries, report);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    prepared, report.reason.c_str());
                amrex::MultiFab impulse(boxes, distribution, 3, 0);
                amrex::MultiFab debit(boxes, distribution, 1, 0);
                amrex::MultiFab scale(boxes, distribution, 1, 0);
                bool const accepted = path.Replay(
                    velocity, trial.Particles(), impulse, debit, scale,
                    report);
                if (!accepted)
                {
                    auto const state = ReadOne(trial.Particles(),
                                                real_sentinel, int_sentinel);
                    amrex::Print() << "XZ primitive rejection: segment="
                        << segment << " of " << fractions.size()
                        << " dt=" << dt << " launch="
                        << (launch != nullptr ? launch->x : initial_x) << ","
                        << (launch != nullptr ? launch->z : initial_z)
                        << " u="
                        << (launch != nullptr ? launch->ux : initial.real[3])
                        << ","
                        << (launch != nullptr ? launch->uz : initial.real[5])
                        << " state=" << state.real[0] << "," << state.real[1]
                        << " owner=" << state.owner << " box=" << state.box
                        << " reason=" << report.reason << "\n";
                }
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(accepted, report.reason.c_str());
                AMREX_ALWAYS_ASSERT(report.collision_events == 0);
                AMREX_ALWAYS_ASSERT(debit.norm0() == 0);
                AMREX_ALWAYS_ASSERT(scale.norm0() == 0);
                for (int c = 0; c < 3; ++c)
                {
                    AMREX_ALWAYS_ASSERT(impulse.norm0(c) == 0);
                }
                result.events += report.collision_events;
                result.faces += report.face_crossings;
                result.face_prefix.push_back(result.faces);
                amrex::Real segment_depth = 0;
                for (auto const& carry : report.photon_carries)
                {
                    if (carry.idcpu == initial.idcpu)
                    {
                        segment_depth = carry.residual_optical_depth;
                    }
                }
                amrex::ParallelDescriptor::ReduceRealSum(segment_depth);
                result.depth_after_segment.push_back(segment_depth);
                result.segment_states.push_back(ReadOne(
                    trial.Particles(), real_sentinel, int_sentinel));
                carries = std::move(report.photon_carries);
            }
            result.photon = ReadOne(trial.Particles(),
                                    real_sentinel, int_sentinel);
            int found = 0;
            for (auto const& carry : carries)
            {
                if (carry.idcpu == initial.idcpu)
                {
                    result.carry = carry;
                    found = 1;
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(found);
            AMREX_ALWAYS_ASSERT(found == 1);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.next_draw);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.completed_events);
            amrex::ParallelDescriptor::ReduceRealSum(
                result.carry.residual_optical_depth);
            result.carry.idcpu = initial.idcpu;
            return result;
        };
        auto const compare = [&] (PartitionResult const& a,
                                  PartitionResult const& b)
        {
            if (a.events != b.events || a.faces != b.faces)
            {
                amrex::Print() << "XZ partition topology mismatch: events "
                    << a.events << "/" << b.events << " faces "
                    << a.faces << "/" << b.faces << " final "
                    << a.photon.real[0] << "," << a.photon.real[1]
                    << " versus " << b.photon.real[0] << ","
                    << b.photon.real[1] << " boxes " << a.photon.box
                    << "/" << b.photon.box << "\n";
            }
            AMREX_ALWAYS_ASSERT(a.events == b.events && a.faces == b.faces);
            AMREX_ALWAYS_ASSERT(a.photon.idcpu == b.photon.idcpu);
            AMREX_ALWAYS_ASSERT(a.photon.owner == b.photon.owner);
            AMREX_ALWAYS_ASSERT(a.photon.box == b.photon.box);
            AMREX_ALWAYS_ASSERT(a.photon.marker == b.photon.marker);
            AMREX_ALWAYS_ASSERT(a.carry.next_draw == b.carry.next_draw);
            AMREX_ALWAYS_ASSERT(a.carry.completed_events ==
                                b.carry.completed_events);
            AMREX_ALWAYS_ASSERT(a.carry.residual_optical_depth ==
                                b.carry.residual_optical_depth);
            for (int c = 0; c < 7; ++c)
            {
                auto const scale = c < 2 ? dx :
                    c >= 3 && c <= 5 ? PhysConst::c : 1.0_rt;
                AMREX_ALWAYS_ASSERT(std::abs(
                    a.photon.real[c]-b.photon.real[c]) <=
                    Bound(a.photon.real[c], b.photon.real[c], scale));
            }
            AMREX_ALWAYS_ASSERT(a.photon.real[2] == 1.0_rt);
            AMREX_ALWAYS_ASSERT(a.photon.real[6] == 19.5_rt);
        };
        auto const short_dt = dx/(4*PhysConst::c);
        auto const full = run(short_dt, {1});
        auto const halves = run(short_dt, {0.5_rt, 0.5_rt});
        auto const quarters = run(short_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt});
        compare(full, halves);
        compare(full, quarters);
        AMREX_ALWAYS_ASSERT(full.faces == 2);
        auto const expected = initial_x+PhysConst::c*short_dt/std::sqrt(2.0_rt);
        AMREX_ALWAYS_ASSERT(std::abs(full.photon.real[0]-expected) <=
                            Bound(full.photon.real[0], expected, dx));
        AMREX_ALWAYS_ASSERT(std::abs(full.photon.real[1]-expected) <=
                            Bound(full.photon.real[1], expected, dx));
        amrex::IntVect const target_cell(AMREX_D_DECL(2, 2, 0));
        int target_box = -1;
        for (int index = 0; index < boxes.size(); ++index)
        {
            if (boxes[index].contains(target_cell)) { target_box = index; }
        }
        AMREX_ALWAYS_ASSERT(target_box >= 0);
        AMREX_ALWAYS_ASSERT(full.photon.box == target_box);
        AMREX_ALWAYS_ASSERT(full.photon.owner == distribution[target_box]);

        auto const diagonal = PhysConst::c/std::sqrt(2.0_rt);
        for (int sign : {-1, 1})
        {
            auto const start = sign > 0 ? 0.25_rt-dx/64 :
                                          -0.25_rt+dx/64;
            LaunchState const launch{start, start,
                                     sign*diagonal, sign*diagonal};
            auto const periodic_full = run(short_dt, {1}, &launch);
            auto const periodic_halves = run(short_dt,
                {0.5_rt, 0.5_rt}, &launch);
            auto const periodic_quarters = run(short_dt,
                {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, &launch);
            compare(periodic_full, periodic_halves);
            compare(periodic_full, periodic_quarters);
            AMREX_ALWAYS_ASSERT(periodic_full.faces == 2);
            auto const unwrapped = start +
                sign*PhysConst::c*short_dt/std::sqrt(2.0_rt);
            auto const expected_periodic =
                unwrapped >= 0.25_rt ? unwrapped-0.5_rt :
                unwrapped < -0.25_rt ? unwrapped+0.5_rt : unwrapped;
            for (int axis = 0; axis < 2; ++axis)
            {
                AMREX_ALWAYS_ASSERT(std::abs(
                    periodic_full.photon.real[axis]-expected_periodic) <=
                    Bound(periodic_full.photon.real[axis], expected_periodic, dx));
            }
            amrex::IntVect const periodic_cell(
                AMREX_D_DECL(sign > 0 ? 0 : 3, sign > 0 ? 0 : 3, 0));
            int periodic_box = -1;
            for (int index = 0; index < boxes.size(); ++index)
            {
                if (boxes[index].contains(periodic_cell))
                {
                    periodic_box = index;
                }
            }
            AMREX_ALWAYS_ASSERT(periodic_box >= 0);
            AMREX_ALWAYS_ASSERT(periodic_full.photon.box == periodic_box);
            AMREX_ALWAYS_ASSERT(periodic_full.photon.owner ==
                                distribution[periodic_box]);
        }

        // Negative travel that ends exactly on a census face must publish
        // the represented coordinate on the travel side in BOTH axes. The
        // table includes internal X, internal Z, internal corner, each
        // periodic lower face, and the simultaneous periodic lower corner.
        struct CensusCase
        {
            LaunchState launch;
            amrex::Real path_length;
            amrex::Real cut_x, cut_z;
            amrex::Real final_x, final_z;
            int cell_x, cell_z;
            int faces;
        };
        auto const d = dx/64;
        std::array<CensusCase, 6> const census_cases{{
            {{d, -dx/4, -PhysConst::c, 0}, 2*d,
                0, -dx/4, -d, -dx/4, 1, 1, 1},
            {{-dx/4, d, 0, -PhysConst::c}, 2*d,
                -dx/4, 0, -dx/4, -d, 1, 1, 1},
            {{d, d, -diagonal, -diagonal}, 2*std::sqrt(2.0_rt)*d,
                0, 0, -d, -d, 1, 1, 2},
            {{-0.25_rt+d, -dx/4, -PhysConst::c, 0}, 2*d,
                0.25_rt, -dx/4, 0.25_rt-d, -dx/4, 3, 1, 1},
            {{-dx/4, -0.25_rt+d, 0, -PhysConst::c}, 2*d,
                -dx/4, 0.25_rt, -dx/4, 0.25_rt-d, 1, 3, 1},
            {{-0.25_rt+d, -0.25_rt+d, -diagonal, -diagonal},
                2*std::sqrt(2.0_rt)*d,
                0.25_rt, 0.25_rt, 0.25_rt-d, 0.25_rt-d, 3, 3, 2}
        }};
        int observed_rank_transfers = 0;
        for (std::size_t case_index = 0;
             case_index < census_cases.size(); ++case_index)
        {
            auto const& test = census_cases[case_index];
            auto const dt = test.path_length/PhysConst::c;
            auto const one = run(dt, {1}, &test.launch);
            auto const two = run(dt, {0.5_rt, 0.5_rt}, &test.launch);
            auto const four = run(dt,
                {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, &test.launch);
            amrex::Print() << "XZ negative census case " << case_index
                << " faces full/2/4=" << one.faces << "/" << two.faces
                << "/" << four.faces << " first cut="
                << two.segment_states[0].real[0] << ","
                << two.segment_states[0].real[1] << " box="
                << two.segment_states[0].box << " prefix="
                << two.face_prefix[0] << "," << two.face_prefix[1]
                << " depth=" << two.depth_after_segment[0] << "\n";
            compare(one, two);
            compare(one, four);
            AMREX_ALWAYS_ASSERT(one.faces == test.faces);
            amrex::IntVect const physical_cell(
                AMREX_D_DECL(test.cell_x, test.cell_z, 0));
            int expected_box = -1;
            for (int index = 0; index < boxes.size(); ++index)
            {
                if (boxes[index].contains(physical_cell))
                {
                    expected_box = index;
                }
            }
            AMREX_ALWAYS_ASSERT(expected_box >= 0);
            AMREX_ALWAYS_ASSERT(one.photon.box == expected_box);
            AMREX_ALWAYS_ASSERT(one.photon.owner ==
                                distribution[expected_box]);
            observed_rank_transfers +=
                one.initial_owner != one.photon.owner ? 1 : 0;
            AMREX_ALWAYS_ASSERT(two.segment_states[0].box == expected_box);
            AMREX_ALWAYS_ASSERT(two.segment_states[0].owner ==
                                distribution[expected_box]);
            std::array<amrex::Real, 2> const cut{test.cut_x, test.cut_z};
            std::array<amrex::Real, 2> const finish{
                test.final_x, test.final_z};
            for (int axis = 0; axis < 2; ++axis)
            {
                AMREX_ALWAYS_ASSERT(std::abs(
                    two.segment_states[0].real[axis]-cut[axis]) <=
                    Bound(two.segment_states[0].real[axis], cut[axis], dx));
                AMREX_ALWAYS_ASSERT(std::abs(
                    one.photon.real[axis]-finish[axis]) <=
                    Bound(one.photon.real[axis], finish[axis], dx));
            }
        }
        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            AMREX_ALWAYS_ASSERT(observed_rank_transfers >= 1);
        }

        auto const long_dt = 0.875_rt*dx/PhysConst::c;
        auto const long_two = run(long_dt, {0.5_rt, 0.5_rt});
        auto const long_four = run(long_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt});
        compare(long_two, long_four);

        // The full primitive starts in the lower-left box and scatters just
        // beyond the diagonal corner. Its recoil is first written to a
        // diagonal ghost and must land in the upper-right physical cell.
        // The split path reaches that box before its event, so comparison
        // detects a lost or multiply summed ghost contribution.
        auto ghost_options = options;
        auto const event_depth = [&] (int event)
        {
            return -std::log(photon_replay_detail::OpenUnitInterval(
                photon_replay_detail::PhotonDrawKey(
                    ghost_options.seed, initial.idcpu,
                    ghost_options.interval, event, 0)));
        };
        bool found_key = false;
        for (std::uint64_t interval = 0; interval < 4096; ++interval)
        {
            ghost_options.interval = interval;
            auto const first = event_depth(0);
            if (first > 0.5_rt && first < 1.5_rt &&
                event_depth(1) > 4*first)
            {
                found_key = true;
                break;
            }
        }
        AMREX_ALWAYS_ASSERT(found_key);
        auto const chi_value = event_depth(0)/(dx/16);
        amrex::MultiFab ghost_opacity(boxes, distribution, 1, 0);
        for (amrex::MFIter iterator(ghost_opacity); iterator.isValid(); ++iterator)
        {
            auto chi = ghost_opacity.array(iterator);
            amrex::For(iterator.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int k, int j)
            {
                amrex::ignore_unused(j);
                chi(i, k, j) = i == 2 && k == 2 ? chi_value : 0;
            });
        }
        amrex::MultiFab work(boxes, distribution, 3, 0);
        work.setVal(0);
        work.setVal(0.125_rt*PhysConst::c, 0, 1);
        work.setVal(0.125_rt*PhysConst::c, 2, 1);
        ghost_options.fixed_frame_direction = {
            -1/std::sqrt(2.0_rt), 0, -1/std::sqrt(2.0_rt)};
        LaunchState const ghost_launch{
            -dx/8, -dx/8, diagonal, diagonal};
        auto const run_event = [&] (
            amrex::Real outer_dt,
            std::vector<amrex::Real> const& fractions,
            LaunchState const& launch,
            amrex::MultiFab const& scattering_opacity,
            amrex::MultiFab const& rate_velocity,
            amrex::MultiFab const& work_velocity,
            NativePhotonCollisionOptions const& event_options,
            amrex::Real initial_depth = 0)
        {
            PhotonPacketTransaction trial(photons);
            SetLaunchState(trial.Particles(), launch);
            EventResult result;
            result.impulse = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 3, 0);
            result.debit = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 1, 0);
            result.scale = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 1, 0);
            result.impulse->setVal(0);
            result.debit->setVal(0);
            result.scale->setVal(0);
            std::vector<NativePhotonPathCarry> carries;
            if (initial_depth > 0)
            {
                for (auto const& [key, tile] : trial.Particles().GetParticles(0))
                {
                    amrex::ignore_unused(key);
                    if (tile.numParticles() == 1)
                    {
                        carries.push_back(NativePhotonPathCarry{
                            initial.idcpu, 1, 0, initial_depth});
                    }
                }
            }
            for (std::size_t segment = 0; segment < fractions.size(); ++segment)
            {
                NativePhotonPathReplay path;
                NativePhotonPathReport report;
                auto const dt = outer_dt*fractions[segment];
                bool const prepared = segment == 0 && initial_depth == 0 ?
                    path.Prepare(trial.Particles(), photons,
                        scattering_opacity, rate_velocity,
                        geometry, dt, event_options, report) :
                    path.Prepare(trial.Particles(), photons,
                        scattering_opacity, rate_velocity,
                        geometry, dt, event_options, carries, report);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    prepared, report.reason.c_str());
                amrex::MultiFab impulse(boxes, distribution, 3, 0);
                amrex::MultiFab debit(boxes, distribution, 1, 0);
                amrex::MultiFab scale(boxes, distribution, 1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(path.Replay(
                    work_velocity, trial.Particles(), impulse, debit, scale,
                    report), report.reason.c_str());
                amrex::MultiFab::Add(*result.impulse, impulse, 0, 0, 3, 0);
                amrex::MultiFab::Add(*result.debit, debit, 0, 0, 1, 0);
                amrex::MultiFab::Add(*result.scale, scale, 0, 0, 1, 0);
                result.events += report.collision_events;
                result.faces += report.face_crossings;
                result.event_prefix.push_back(result.events);
                result.segment_states.push_back(ReadOne(
                    trial.Particles(), real_sentinel, int_sentinel));
                carries = std::move(report.photon_carries);
            }
            result.photon = ReadOne(trial.Particles(),
                                    real_sentinel, int_sentinel);
            int found = 0;
            for (auto const& carry : carries)
            {
                if (carry.idcpu == initial.idcpu)
                {
                    result.carry = carry;
                    found = 1;
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(found);
            AMREX_ALWAYS_ASSERT(found == 1);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.next_draw);
            amrex::ParallelDescriptor::ReduceIntSum(result.carry.completed_events);
            amrex::ParallelDescriptor::ReduceRealSum(
                result.carry.residual_optical_depth);
            result.carry.idcpu = initial.idcpu;
            return result;
        };
        auto const ghost_dt = 0.46875_rt*dx/PhysConst::c;
        auto const ghost_full = run_event(ghost_dt, {1},
            ghost_launch, ghost_opacity, velocity, work, ghost_options);
        auto const ghost_two = run_event(ghost_dt, {0.5_rt, 0.5_rt},
            ghost_launch, ghost_opacity, velocity, work, ghost_options);
        auto const ghost_four = run_event(ghost_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt},
            ghost_launch, ghost_opacity, velocity, work, ghost_options);
        auto const compare_event = [&] (EventResult const& a,
                                        EventResult const& b)
        {
            AMREX_ALWAYS_ASSERT(a.events == b.events);
            AMREX_ALWAYS_ASSERT(a.faces == b.faces);
            AMREX_ALWAYS_ASSERT(a.photon.idcpu == b.photon.idcpu);
            AMREX_ALWAYS_ASSERT(a.photon.box == b.photon.box);
            AMREX_ALWAYS_ASSERT(a.photon.owner == b.photon.owner);
            AMREX_ALWAYS_ASSERT(a.photon.marker == b.photon.marker);
            AMREX_ALWAYS_ASSERT(a.carry.next_draw == b.carry.next_draw);
            AMREX_ALWAYS_ASSERT(a.carry.completed_events ==
                                b.carry.completed_events);
            AMREX_ALWAYS_ASSERT(std::abs(
                a.carry.residual_optical_depth-
                b.carry.residual_optical_depth) <= Bound(
                    a.carry.residual_optical_depth,
                    b.carry.residual_optical_depth, event_depth(0)));
            for (int c = 0; c < 7; ++c)
            {
                auto const scale = c < 2 ? dx :
                    c >= 3 && c <= 5 ? PhysConst::c : 1.0_rt;
                AMREX_ALWAYS_ASSERT(std::abs(
                    a.photon.real[c]-b.photon.real[c]) <=
                    Bound(a.photon.real[c], b.photon.real[c], scale));
            }
            AMREX_ALWAYS_ASSERT(a.photon.real[2] == b.photon.real[2]);
            AMREX_ALWAYS_ASSERT(a.photon.real[6] == b.photon.real[6]);
            auto const four_a = PacketConversionFourMomentumFromPhoton(
                a.photon.real[2], a.photon.real[3],
                a.photon.real[4], a.photon.real[5]);
            auto const four_b = PacketConversionFourMomentumFromPhoton(
                b.photon.real[2], b.photon.real[3],
                b.photon.real[4], b.photon.real[5]);
            AMREX_ALWAYS_ASSERT(four_a.valid && four_b.valid);
            for (int c = 0; c < 4; ++c)
            {
                AMREX_ALWAYS_ASSERT(std::abs(
                    four_a.value[c]-four_b.value[c]) <=
                    Bound(four_a.value[c], four_b.value[c],
                          four_a.value[0]));
            }
            for (int i = 0; i < 4; ++i)
            {
                for (int k = 0; k < 4; ++k)
                {
                    auto const local_scale = std::max({
                        std::abs(CellValue(*a.scale, i, k, 0)),
                        std::abs(CellValue(*b.scale, i, k, 0)),
                        std::abs(CellValue(*a.debit, i, k, 0)),
                        std::abs(CellValue(*b.debit, i, k, 0))});
                    for (int c = 0; c < 3; ++c)
                    {
                        auto const left = PhysConst::c*
                            CellValue(*a.impulse, i, k, c);
                        auto const right = PhysConst::c*
                            CellValue(*b.impulse, i, k, c);
                        AMREX_ALWAYS_ASSERT(std::abs(left-right) <=
                            Bound(left, right, local_scale));
                    }
                    auto const left = CellValue(*a.debit, i, k, 0);
                    auto const right = CellValue(*b.debit, i, k, 0);
                    AMREX_ALWAYS_ASSERT(std::abs(left-right) <=
                        Bound(left, right, local_scale));
                }
            }
        };
        AMREX_ALWAYS_ASSERT(ghost_full.events == 1);
        AMREX_ALWAYS_ASSERT(ghost_two.event_prefix ==
                            std::vector<amrex::Long>({0, 1}));
        AMREX_ALWAYS_ASSERT(ghost_four.event_prefix ==
                            std::vector<amrex::Long>({0, 0, 1, 1}));
        AMREX_ALWAYS_ASSERT(ghost_two.segment_states[0].box == target_box);
        AMREX_ALWAYS_ASSERT(ghost_two.segment_states[0].owner ==
                            distribution[target_box]);
        AMREX_ALWAYS_ASSERT(ghost_four.segment_states[1].box == target_box);
        AMREX_ALWAYS_ASSERT(CellValue(*ghost_full.debit, 2, 2, 0) != 0);
        AMREX_ALWAYS_ASSERT(CellValue(*ghost_full.impulse, 2, 2, 0) != 0);
        AMREX_ALWAYS_ASSERT(CellValue(*ghost_full.impulse, 2, 2, 2) != 0);
        for (int i = 0; i < 4; ++i)
        {
            for (int k = 0; k < 4; ++k)
            {
                if (i == 2 && k == 2) { continue; }
                AMREX_ALWAYS_ASSERT(CellValue(*ghost_full.debit, i, k, 0) == 0);
                for (int c = 0; c < 3; ++c)
                {
                    AMREX_ALWAYS_ASSERT(CellValue(
                        *ghost_full.impulse, i, k, c) == 0);
                }
            }
        }
        compare_event(ghost_full, ghost_two);
        compare_event(ghost_full, ghost_four);

        // Freeze a nonzero proper-rate frame in the destination cell while
        // the staged material work frame remains different. The keyed first
        // event still belongs to the diagonal ghost's physical owner.
        NativePhotonCollisionOptions moving_options = ghost_options;
        moving_options.moving_cross_sections = true;
        amrex::MultiFab moving_rate(boxes, distribution, 3, 0);
        amrex::MultiFab moving_opacity(boxes, distribution, 1, 0);
        moving_rate.setVal(0);
        auto const beta0 = -0.125_rt;
        auto const beta_norm_squared = 2*beta0*beta0;
        auto const rate_factor =
            (1-std::sqrt(2.0_rt)*beta0)/
            std::sqrt(1-beta_norm_squared);
        auto const moving_chi = event_depth(0)/((dx/16)*rate_factor);
        for (amrex::MFIter iterator(moving_opacity); iterator.isValid();
             ++iterator)
        {
            auto chi = moving_opacity.array(iterator);
            auto rate = moving_rate.array(iterator);
            amrex::For(iterator.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int k, int j)
            {
                auto const target = i == 2 && k == 2;
                chi(i, k, j) = target ? moving_chi : 0;
                rate(i, k, j, 0) = target ? beta0*PhysConst::c : 0;
                rate(i, k, j, 1) = 0;
                rate(i, k, j, 2) = target ? beta0*PhysConst::c : 0;
            });
        }
        auto const moving_full = run_event(ghost_dt, {1},
            ghost_launch, moving_opacity, moving_rate, work, moving_options);
        auto const moving_two = run_event(ghost_dt,
            {0.5_rt, 0.5_rt}, ghost_launch,
            moving_opacity, moving_rate, work, moving_options);
        auto const moving_four = run_event(ghost_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, ghost_launch,
            moving_opacity, moving_rate, work, moving_options);
        AMREX_ALWAYS_ASSERT(moving_full.events == 1);
        AMREX_ALWAYS_ASSERT(CellValue(*moving_full.debit, 2, 2, 0) != 0);
        compare_event(moving_full, moving_two);
        compare_event(moving_full, moving_four);

        // Two keyed isotropic events straddle the diagonal corner and later
        // return crossing. The first deposits in destination cell (2,2),
        // the second in source cell (1,1), with both nonzero staged-work
        // debits. This tests chronology and cell ownership independently of
        // the one-event ghost case.
        NativePhotonCollisionOptions two_options = options;
        two_options.angular_law = NativePhotonCollisionAngularLaw::Isotropic;
        two_options.max_optical_depth = 16;
        auto const keyed_depth = [&] (int event)
        {
            return -std::log(photon_replay_detail::OpenUnitInterval(
                photon_replay_detail::PhotonDrawKey(
                    two_options.seed, initial.idcpu,
                    two_options.interval, event, 0)));
        };
        auto const drawn_direction = [&] (int event)
        {
            auto const cosine = 2*photon_replay_detail::OpenUnitInterval(
                photon_replay_detail::PhotonDrawKey(
                    two_options.seed, initial.idcpu,
                    two_options.interval, event, 1))-1;
            auto const azimuth = 2*3.14159265358979323846_rt*
                photon_replay_detail::OpenUnitInterval(
                    photon_replay_detail::PhotonDrawKey(
                        two_options.seed, initial.idcpu,
                        two_options.interval, event, 2));
            return std::array<amrex::Real, 2>{
                std::sqrt(1-cosine*cosine)*std::cos(azimuth), cosine};
        };
        bool found_two_key = false;
        for (std::uint64_t interval = 0; interval < 65536; ++interval)
        {
            two_options.interval = interval;
            auto const first_direction = drawn_direction(0);
            auto const second_direction = drawn_direction(1);
            auto const second_depth = keyed_depth(1);
            if (first_direction[0] < -0.55_rt &&
                first_direction[1] < -0.55_rt &&
                second_direction[0] > 0.3_rt &&
                second_direction[1] > 0.3_rt &&
                second_depth > 0.5_rt && second_depth < 1.5_rt &&
                keyed_depth(2) > second_depth)
            {
                found_two_key = true;
                break;
            }
        }
        AMREX_ALWAYS_ASSERT(found_two_key);
        auto const first_path = std::sqrt(2.0_rt)*dx/8+dx/16;
        auto const second_path = dx/5;
        auto const two_chi = keyed_depth(1)/second_path;
        amrex::MultiFab two_opacity(boxes, distribution, 1, 0);
        two_opacity.setVal(two_chi);
        auto const two_depth = two_chi*first_path;
        auto const ordered_full = run_event(ghost_dt, {1},
            ghost_launch, two_opacity, velocity, work,
            two_options, two_depth);
        auto const ordered_two = run_event(ghost_dt,
            {0.5_rt, 0.5_rt}, ghost_launch,
            two_opacity, velocity, work, two_options, two_depth);
        auto const ordered_four = run_event(ghost_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, ghost_launch,
            two_opacity, velocity, work, two_options, two_depth);
        AMREX_ALWAYS_ASSERT(ordered_full.events == 2);
        AMREX_ALWAYS_ASSERT(ordered_two.event_prefix ==
                            std::vector<amrex::Long>({0, 2}));
        AMREX_ALWAYS_ASSERT(ordered_four.event_prefix ==
                            std::vector<amrex::Long>({0, 0, 1, 2}));
        for (auto const cell : {
                 std::array<int, 2>{2, 2}, std::array<int, 2>{1, 1}})
        {
            AMREX_ALWAYS_ASSERT(CellValue(
                *ordered_full.debit, cell[0], cell[1], 0) != 0);
            AMREX_ALWAYS_ASSERT(CellValue(
                *ordered_full.impulse, cell[0], cell[1], 0) != 0);
            AMREX_ALWAYS_ASSERT(CellValue(
                *ordered_full.impulse, cell[0], cell[1], 2) != 0);
        }
        compare_event(ordered_full, ordered_two);
        compare_event(ordered_full, ordered_four);

        // An explicitly carried Poisson depth locates one material event
        // before, exactly on, or after an X face, Z face, and simultaneous
        // corner/census cut. The physical recoil cell is an exact invariant.
        amrex::MultiFab uniform_opacity(boxes, distribution, 1, 0);
        uniform_opacity.setVal(128);
        LaunchState const thin_launch{
            initial_x, initial_z, diagonal, diagonal};
        auto const thin_full = run_event(short_dt, {1},
            thin_launch, uniform_opacity, velocity, velocity,
            ghost_options, 8);
        auto const thin_two = run_event(short_dt,
            {0.5_rt, 0.5_rt}, thin_launch,
            uniform_opacity, velocity, velocity, ghost_options, 8);
        auto const thin_four = run_event(short_dt,
            {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, thin_launch,
            uniform_opacity, velocity, velocity, ghost_options, 8);
        AMREX_ALWAYS_ASSERT(thin_full.events == 0);
        AMREX_ALWAYS_ASSERT(thin_full.faces == 2);
        AMREX_ALWAYS_ASSERT(thin_full.carry.next_draw == 1);
        AMREX_ALWAYS_ASSERT(thin_full.carry.completed_events == 0);
        AMREX_ALWAYS_ASSERT(std::abs(
            thin_full.carry.residual_optical_depth-4) <=
            Bound(thin_full.carry.residual_optical_depth, 4, 8));
        AMREX_ALWAYS_ASSERT(thin_full.debit->norm0() == 0);
        for (int c = 0; c < 3; ++c)
        {
            AMREX_ALWAYS_ASSERT(thin_full.impulse->norm0(c) == 0);
        }
        compare_event(thin_full, thin_two);
        compare_event(thin_full, thin_four);
        struct FaceCase
        {
            LaunchState launch;
            amrex::Real face_path;
            bool crosses_x;
            bool crosses_z;
        };
        auto const face_d = dx/64;
        std::array<FaceCase, 3> const face_cases{{
            {{-face_d, -dx/4, PhysConst::c, 0}, face_d, true, false},
            {{-dx/4, -face_d, 0, PhysConst::c}, face_d, false, true},
            {{-face_d, -face_d, diagonal, diagonal},
                std::sqrt(2.0_rt)*face_d, true, true}
        }};
        for (auto const& test : face_cases)
        {
            for (auto const relative_depth : {0.75_rt, 1.0_rt, 1.25_rt})
            {
                auto const depth = 128*test.face_path*relative_depth;
                auto const dt = 2*test.face_path/PhysConst::c;
                auto const one = run_event(dt, {1},
                    test.launch, uniform_opacity, velocity, work,
                    ghost_options, depth);
                auto const two = run_event(dt, {0.5_rt, 0.5_rt},
                    test.launch, uniform_opacity, velocity, work,
                    ghost_options, depth);
                auto const four = run_event(dt,
                    {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt},
                    test.launch, uniform_opacity, velocity, work,
                    ghost_options, depth);
                AMREX_ALWAYS_ASSERT(one.events == 1);
                AMREX_ALWAYS_ASSERT(two.event_prefix[0] ==
                                    (relative_depth > 1 ? 0 : 1));
                AMREX_ALWAYS_ASSERT(two.event_prefix[1] == 1);
                AMREX_ALWAYS_ASSERT(four.event_prefix[1] ==
                                    (relative_depth > 1 ? 0 : 1));
                AMREX_ALWAYS_ASSERT(four.event_prefix[3] == 1);
                int const i = test.crosses_x && relative_depth > 1 ? 2 : 1;
                int const k = test.crosses_z && relative_depth > 1 ? 2 : 1;
                AMREX_ALWAYS_ASSERT(CellValue(*one.debit, i, k, 0) != 0);
                for (int cell_i = 0; cell_i < 4; ++cell_i)
                {
                    for (int cell_k = 0; cell_k < 4; ++cell_k)
                    {
                        if (cell_i == i && cell_k == k) { continue; }
                        AMREX_ALWAYS_ASSERT(CellValue(
                            *one.debit, cell_i, cell_k, 0) == 0);
                        for (int c = 0; c < 3; ++c)
                        {
                            AMREX_ALWAYS_ASSERT(CellValue(
                                *one.impulse, cell_i, cell_k, c) == 0);
                        }
                    }
                }
                compare_event(one, two);
                compare_event(one, four);
            }
        }
        int require_helper = 0;
        int expect_closed = 0;
        xz_parameters.query("require_helper", require_helper);
        xz_parameters.query("expect_closed", expect_closed);
        if (expect_closed != 0)
        {
            NativePhotonPathSubsegmentReplay guard;
            NativePhotonPathReport guarded;
            AMREX_ALWAYS_ASSERT(!guard.Prepare(
                photons, opacity, velocity, geometry, long_dt,
                options, 4, guarded));
            AMREX_ALWAYS_ASSERT(!guarded.valid);
        }
        if (require_helper != 0)
        {
            NativePhotonPathSubsegmentReplay helper;
            NativePhotonPathReport prepared;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helper.Prepare(
                photons, opacity, velocity, geometry, long_dt, options,
                4, prepared), prepared.reason.c_str());
            PhotonPacketTransaction trial(photons);
            amrex::MultiFab impulse(boxes, distribution, 3, 0);
            amrex::MultiFab debit(boxes, distribution, 1, 0);
            amrex::MultiFab scale(boxes, distribution, 1, 0);
            NativePhotonPathSubsegmentReport report;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helper.Replay(
                velocity, trial.Particles(), impulse, debit, scale,
                report), report.path.reason.c_str());
            for (int c = 0; c < 3; ++c)
            {
                AMREX_ALWAYS_ASSERT(impulse.norm0(c) == 0);
            }
            AMREX_ALWAYS_ASSERT(debit.norm0() == 0);
            AMREX_ALWAYS_ASSERT(scale.norm0() == 0);
            PartitionResult actual;
            actual.photon = ReadOne(trial.Particles(),
                                    real_sentinel, int_sentinel);
            actual.events = report.path.collision_events;
            actual.faces = report.path.face_crossings;
            for (auto const& carry : report.photon_carries)
            {
                if (carry.idcpu == initial.idcpu) { actual.carry = carry; }
            }
            amrex::ParallelDescriptor::ReduceIntSum(actual.carry.next_draw);
            amrex::ParallelDescriptor::ReduceIntSum(actual.carry.completed_events);
            amrex::ParallelDescriptor::ReduceRealSum(
                actual.carry.residual_optical_depth);
            actual.carry.idcpu = initial.idcpu;
            compare(actual, long_two);
            if (amrex::ParallelDescriptor::NProcs() > 1)
            {
                NativePhotonPathSubsegmentReplay mismatched_limit;
                NativePhotonPathReport mismatch_report;
                auto const local_limit =
                    amrex::ParallelDescriptor::MyProc() == 0 ? 3 : 4;
                AMREX_ALWAYS_ASSERT(!mismatched_limit.Prepare(
                    photons, opacity, velocity, geometry, long_dt,
                    options, local_limit, mismatch_report));
                amrex::MultiFab invalid_stage(boxes, distribution, 3, 0);
                invalid_stage.setVal(0);
                if (amrex::ParallelDescriptor::MyProc() == 0)
                {
                    invalid_stage.setVal(PhysConst::c);
                }
                PhotonPacketTransaction invalid_trial(photons);
                auto const invalid_before = ReadOne(invalid_trial.Particles(),
                    real_sentinel, int_sentinel);
                amrex::MultiFab invalid_impulse(boxes, distribution, 3, 1);
                amrex::MultiFab invalid_debit(boxes, distribution, 1, 1);
                amrex::MultiFab invalid_scale(boxes, distribution, 1, 1);
                invalid_impulse.setVal(123);
                invalid_debit.setVal(123);
                invalid_scale.setVal(123);
                NativePhotonPathSubsegmentReport invalid_report;
                AMREX_ALWAYS_ASSERT(!helper.Replay(invalid_stage,
                    invalid_trial.Particles(), invalid_impulse,
                    invalid_debit, invalid_scale, invalid_report));
                auto const invalid_after = ReadOne(invalid_trial.Particles(),
                    real_sentinel, int_sentinel);
                AMREX_ALWAYS_ASSERT(invalid_after.real == invalid_before.real);
                AMREX_ALWAYS_ASSERT(invalid_after.owner == invalid_before.owner);
                AMREX_ALWAYS_ASSERT(invalid_after.box == invalid_before.box);
                AMREX_ALWAYS_ASSERT(!invalid_report.path.valid);
                AMREX_ALWAYS_ASSERT(invalid_report.path.collision_events == 0);
                AMREX_ALWAYS_ASSERT(invalid_report.photon_carries.empty());
                invalid_impulse.plus(-123, 0, 3, 1);
                invalid_debit.plus(-123, 0, 1, 1);
                invalid_scale.plus(-123, 0, 1, 1);
                for (int c = 0; c < 3; ++c)
                {
                    AMREX_ALWAYS_ASSERT(invalid_impulse.norm0(c, 1) == 0);
                }
                AMREX_ALWAYS_ASSERT(invalid_debit.norm0(0, 1) == 0);
                AMREX_ALWAYS_ASSERT(invalid_scale.norm0(0, 1) == 0);
            }

            auto retry_options = ghost_options;
            retry_options.max_optical_depth = 64;
            NativePhotonPathSubsegmentReplay retry;
            NativePhotonPathReport retry_prepared;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(retry.Prepare(
                photons, ghost_opacity, velocity, geometry, long_dt,
                retry_options, 4, retry_prepared),
                retry_prepared.reason.c_str());
            amrex::MultiFab opposite_work(boxes, distribution, 3, 0);
            opposite_work.setVal(0);
            opposite_work.setVal(-0.125_rt*PhysConst::c, 0, 1);
            opposite_work.setVal(-0.125_rt*PhysConst::c, 2, 1);
            PhotonPacketTransaction forward(photons);
            PhotonPacketTransaction reverse(photons);
            PhotonPacketTransaction repeated(photons);
            amrex::MultiFab impulse_a(boxes, distribution, 3, 0);
            amrex::MultiFab impulse_b(boxes, distribution, 3, 0);
            amrex::MultiFab impulse_c(boxes, distribution, 3, 0);
            amrex::MultiFab debit_a(boxes, distribution, 1, 0);
            amrex::MultiFab debit_b(boxes, distribution, 1, 0);
            amrex::MultiFab debit_c(boxes, distribution, 1, 0);
            amrex::MultiFab scale_a(boxes, distribution, 1, 0);
            amrex::MultiFab scale_b(boxes, distribution, 1, 0);
            amrex::MultiFab scale_c(boxes, distribution, 1, 0);
            NativePhotonPathSubsegmentReport report_a, report_b, report_c;
            AMREX_ALWAYS_ASSERT(retry.Replay(work,
                forward.Particles(), impulse_a, debit_a, scale_a, report_a));
            AMREX_ALWAYS_ASSERT(retry.Replay(opposite_work,
                reverse.Particles(), impulse_b, debit_b, scale_b, report_b));
            AMREX_ALWAYS_ASSERT(retry.Replay(work,
                repeated.Particles(), impulse_c, debit_c, scale_c, report_c));
            AMREX_ALWAYS_ASSERT(report_a.path.collision_events > 0);
            AMREX_ALWAYS_ASSERT(report_a.path.collision_events ==
                                report_c.path.collision_events);
            auto const forward_state = ReadOne(forward.Particles(),
                real_sentinel, int_sentinel);
            auto const reverse_state = ReadOne(reverse.Particles(),
                real_sentinel, int_sentinel);
            auto const repeated_state = ReadOne(repeated.Particles(),
                real_sentinel, int_sentinel);
            AMREX_ALWAYS_ASSERT(forward_state.real == repeated_state.real);
            AMREX_ALWAYS_ASSERT(forward_state.owner == repeated_state.owner);
            AMREX_ALWAYS_ASSERT(forward_state.box == repeated_state.box);
            AMREX_ALWAYS_ASSERT(forward_state.real != reverse_state.real);
            for (int component = 0; component < 3; ++component)
            {
                amrex::MultiFab difference(boxes, distribution, 1, 0);
                amrex::MultiFab::Copy(
                    difference, impulse_a, component, 0, 1, 0);
                amrex::MultiFab::Subtract(
                    difference, impulse_c, component, 0, 1, 0);
                AMREX_ALWAYS_ASSERT(difference.norm0() == 0);
            }
            amrex::MultiFab::Subtract(debit_a, debit_c, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(scale_a, scale_c, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(debit_a.norm0() == 0);
            AMREX_ALWAYS_ASSERT(scale_a.norm0() == 0);
            AMREX_ALWAYS_ASSERT(report_a.photon_carries.size() ==
                                report_c.photon_carries.size());
            for (std::size_t n = 0; n < report_a.photon_carries.size(); ++n)
            {
                auto const& a = report_a.photon_carries[n];
                auto const& c = report_c.photon_carries[n];
                AMREX_ALWAYS_ASSERT(a.idcpu == c.idcpu);
                AMREX_ALWAYS_ASSERT(a.next_draw == c.next_draw);
                AMREX_ALWAYS_ASSERT(a.completed_events == c.completed_events);
                AMREX_ALWAYS_ASSERT(a.residual_optical_depth ==
                                    c.residual_optical_depth);
            }

            // The first segment accepts one event. A second event on one
            // owner rank exceeds the OUTER cap in the later segment, so the
            // helper must publish neither its private photon nor any valid
            // cell, ghost, or counter output.
            auto limited = options;
            limited.fixed_frame_direction = {0, 0, 1};
            limited.max_events_per_photon = 1;
            auto const depth_at = [&] (int event)
            {
                return -std::log(photon_replay_detail::OpenUnitInterval(
                    photon_replay_detail::PhotonDrawKey(
                        limited.seed, initial.idcpu,
                        limited.interval, event, 0)));
            };
            auto const first = depth_at(0);
            auto const second = first+depth_at(1);
            auto const total_depth =
                (std::max(2*first, second)+2*second)/2;
            limited.max_optical_depth = 2*total_depth;
            amrex::MultiFab capped_opacity(boxes, distribution, 1, 0);
            capped_opacity.setVal(total_depth/(PhysConst::c*long_dt));
            PhotonPacketTransaction preview(photons);
            NativePhotonPathReplay first_segment;
            NativePhotonPathReport first_report;
            AMREX_ALWAYS_ASSERT(first_segment.Prepare(
                preview.Particles(), photons, capped_opacity, velocity,
                geometry, long_dt/2, limited, first_report));
            amrex::MultiFab preview_impulse(boxes, distribution, 3, 0);
            amrex::MultiFab preview_debit(boxes, distribution, 1, 0);
            amrex::MultiFab preview_scale(boxes, distribution, 1, 0);
            AMREX_ALWAYS_ASSERT(first_segment.Replay(velocity,
                preview.Particles(), preview_impulse, preview_debit,
                preview_scale, first_report));
            AMREX_ALWAYS_ASSERT(first_report.collision_events == 1);
            NativePhotonPathSubsegmentReplay capped;
            NativePhotonPathReport capped_prepared;
            AMREX_ALWAYS_ASSERT(capped.Prepare(photons,
                capped_opacity, velocity, geometry, long_dt,
                limited, 4, capped_prepared));
            PhotonPacketTransaction rejected_trial(photons);
            auto const rejected_before = ReadOne(rejected_trial.Particles(),
                real_sentinel, int_sentinel);
            amrex::MultiFab rejected_impulse(boxes, distribution, 3, 1);
            amrex::MultiFab rejected_debit(boxes, distribution, 1, 1);
            amrex::MultiFab rejected_scale(boxes, distribution, 1, 1);
            rejected_impulse.setVal(123);
            rejected_debit.setVal(123);
            rejected_scale.setVal(123);
            auto const read_next_id = []()
            {
                auto const value =
                    WarpXParticleContainer::ParticleType::NextID();
                WarpXParticleContainer::ParticleType::NextID(value);
                return value;
            };
            auto const id_before = read_next_id();
            NativePhotonPathSubsegmentReport rejection;
            AMREX_ALWAYS_ASSERT(!capped.Replay(velocity,
                rejected_trial.Particles(), rejected_impulse,
                rejected_debit, rejected_scale, rejection));
            auto const rejected_after = ReadOne(rejected_trial.Particles(),
                real_sentinel, int_sentinel);
            AMREX_ALWAYS_ASSERT(rejected_after.real == rejected_before.real);
            AMREX_ALWAYS_ASSERT(rejected_after.idcpu == rejected_before.idcpu);
            AMREX_ALWAYS_ASSERT(rejected_after.owner == rejected_before.owner);
            AMREX_ALWAYS_ASSERT(rejected_after.box == rejected_before.box);
            AMREX_ALWAYS_ASSERT(read_next_id() == id_before);
            AMREX_ALWAYS_ASSERT(!rejection.path.valid);
            AMREX_ALWAYS_ASSERT(rejection.path.collision_events == 0);
            AMREX_ALWAYS_ASSERT(rejection.path.face_crossings == 0);
            AMREX_ALWAYS_ASSERT(rejection.path.colliding_photons == 0);
            AMREX_ALWAYS_ASSERT(rejection.photon_carries.empty());
            rejected_impulse.plus(-123, 0, 3, 1);
            rejected_debit.plus(-123, 0, 1, 1);
            rejected_scale.plus(-123, 0, 1, 1);
            for (int component = 0; component < 3; ++component)
            {
                AMREX_ALWAYS_ASSERT(rejected_impulse.norm0(component, 1) == 0);
            }
            AMREX_ALWAYS_ASSERT(rejected_debit.norm0(0, 1) == 0);
            AMREX_ALWAYS_ASSERT(rejected_scale.norm0(0, 1) == 0);

            // The helper's real two-segment event path must agree with two
            // independent legal primitive partitions, including the
            // physical destination cell's recoil after diagonal SumBoundary.
            auto const ghost_long_two = run_event(long_dt,
                {0.5_rt, 0.5_rt}, ghost_launch,
                ghost_opacity, velocity, work, retry_options);
            auto const ghost_long_four = run_event(long_dt,
                {0.25_rt, 0.25_rt, 0.25_rt, 0.25_rt}, ghost_launch,
                ghost_opacity, velocity, work, retry_options);
            compare_event(ghost_long_two, ghost_long_four);
            SetLaunchState(photons, ghost_launch);
            NativePhotonPathSubsegmentReplay ghost_helper;
            NativePhotonPathReport ghost_prepared;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ghost_helper.Prepare(
                photons, ghost_opacity, velocity, geometry, long_dt,
                retry_options, 4, ghost_prepared),
                ghost_prepared.reason.c_str());
            PhotonPacketTransaction helper_trial(photons);
            amrex::MultiFab helper_impulse(boxes, distribution, 3, 0);
            amrex::MultiFab helper_debit(boxes, distribution, 1, 0);
            amrex::MultiFab helper_scale(boxes, distribution, 1, 0);
            NativePhotonPathSubsegmentReport helper_report;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ghost_helper.Replay(work,
                helper_trial.Particles(), helper_impulse, helper_debit,
                helper_scale, helper_report),
                helper_report.path.reason.c_str());
            AMREX_ALWAYS_ASSERT(helper_report.subsegments == 2);
            EventResult helper_result;
            helper_result.photon = ReadOne(helper_trial.Particles(),
                real_sentinel, int_sentinel);
            helper_result.events = helper_report.path.collision_events;
            helper_result.faces = helper_report.path.face_crossings;
            helper_result.impulse = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 3, 0);
            helper_result.debit = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 1, 0);
            helper_result.scale = std::make_unique<amrex::MultiFab>(
                boxes, distribution, 1, 0);
            amrex::MultiFab::Copy(*helper_result.impulse,
                helper_impulse, 0, 0, 3, 0);
            amrex::MultiFab::Copy(*helper_result.debit,
                helper_debit, 0, 0, 1, 0);
            amrex::MultiFab::Copy(*helper_result.scale,
                helper_scale, 0, 0, 1, 0);
            int found_carry = 0;
            for (auto const& carry : helper_report.photon_carries)
            {
                if (carry.idcpu == initial.idcpu)
                {
                    helper_result.carry = carry;
                    found_carry = 1;
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(found_carry);
            AMREX_ALWAYS_ASSERT(found_carry == 1);
            amrex::ParallelDescriptor::ReduceIntSum(
                helper_result.carry.next_draw);
            amrex::ParallelDescriptor::ReduceIntSum(
                helper_result.carry.completed_events);
            amrex::ParallelDescriptor::ReduceRealSum(
                helper_result.carry.residual_optical_depth);
            helper_result.carry.idcpu = initial.idcpu;
            AMREX_ALWAYS_ASSERT(helper_result.events >= 1);
            AMREX_ALWAYS_ASSERT(CellValue(
                *helper_result.debit, 2, 2, 0) != 0);
            compare_event(helper_result, ghost_long_two);
            SetLaunchState(photons, LaunchState{
                initial.real[0], initial.real[1],
                initial.real[3], initial.real[5]});
        }
        auto const live = ReadOne(photons, real_sentinel, int_sentinel);
        AMREX_ALWAYS_ASSERT(live.idcpu == initial.idcpu);
        AMREX_ALWAYS_ASSERT(live.real == initial.real);
        amrex::Print() << "XZ transparent diagonal partition gate passed; "
            << "helper requested=" << require_helper << "\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}

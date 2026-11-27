/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/RZPressureInduction.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/RZPressureInductionStage.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

using namespace amrex::literals;

namespace {
struct Fields {
    amrex::MultiFab B;
    amrex::MultiFab U;
    amrex::MultiFab T;
    amrex::MultiFab pressure;
    amrex::MultiFab charge;
    amrex::MultiFab volume;
};

struct InitialThermalState {
    amrex::Real temperature = amrex::Real(0.0);
    amrex::Real pressure = amrex::Real(0.0);
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE InitialThermalState
thermalStateFromChargeEnergy (amrex::Real const charge,
                              amrex::Real const energy) {
    ElectronThermodynamicsExecutor eos{};
    amrex::Real const temperature =
        eos.temperatureFromChargeDensityThermalEnergyDensity(charge, energy);
    return {
        temperature,
        eos.stateFromChargeDensityTemperature(charge, temperature).pressure};
}

void
initialize (Fields& state, amrex::BoxArray const& boxes,
            amrex::DistributionMapping const& mapping,
            amrex::Geometry const& geometry, int cells, bool constant_charge) {
    for (auto* field : {&state.B, &state.U, &state.T, &state.pressure,
                        &state.charge, &state.volume}) {
        if (!field->isDefined()) {
            field->define(boxes, mapping, 1, 1);
        }
        field->setVal(-123);
    }

    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    for (amrex::MFIter it(state.B); it.isValid(); ++it) {
        auto const B = state.B.array(it), U = state.U.array(it);
        auto const T = state.T.array(it), P = state.pressure.array(it);
        auto const rho = state.charge.array(it), V = state.volume.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            amrex::Real const r = amrex::Real(i) / amrex::Real(cells);
            amrex::Real const phase =
                2 * MathConst::pi * amrex::Real(j) / amrex::Real(cells);
            amrex::Real const shape =
                amrex::max(amrex::Real(0.0), 1 - 4 * r * r);
            U(i, j, k) =
                1.e9_rt * (2 + shape * shape * shape * shape *
                                   (1 + amrex::Real(0.2) * std::cos(phase)));
            rho(i, j, k) =
                1.e7_rt *
                (constant_charge ? 1 : 1 + amrex::Real(0.2) * std::sin(phase));
            InitialThermalState const thermal =
                thermalStateFromChargeEnergy(rho(i, j, k), U(i, j, k));
            T(i, j, k) = thermal.temperature;
            P(i, j, k) = thermal.pressure;
            B(i, j, k) = constant_charge ? 0 : 5 * r * (1 - r * r);
            V(i, j, k) = i == 0 ? MathConst::pi * dr * dr * dz / 3
                                : 2 * MathConst::pi * i * dr * dr * dz *
                                      (i == cells ? amrex::Real(0.5) : 1.0);
        });
    }
}

void
initializeUniformTemperatureProfile (Fields& state, int cells) {
    for (amrex::MFIter it(state.B); it.isValid(); ++it) {
        auto const U = state.U.array(it), T = state.T.array(it);
        auto const P = state.pressure.array(it), rho = state.charge.array(it);
        amrex::ParallelFor(
            it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real const phase =
                    2 * MathConst::pi * amrex::Real(j) / amrex::Real(cells);
                amrex::Real const ratio =
                    amrex::Real(1.0) + amrex::Real(0.2) * std::sin(phase);
                rho(i, j, k) = 1.e7_rt * ratio;
                U(i, j, k) = 2.e9_rt * ratio;
                InitialThermalState const thermal =
                    thermalStateFromChargeEnergy(rho(i, j, k), U(i, j, k));
                T(i, j, k) = thermal.temperature;
                P(i, j, k) = thermal.pressure;
            });
    }
}

void
setLargeMagneticField (Fields& state) {
    for (amrex::MFIter it(state.B); it.isValid(); ++it) {
        auto const B = state.B.array(it);
        amrex::ParallelFor(
            it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                B(i, j, k) = i == 0 ? amrex::Real(0.0)
                                    : std::numeric_limits<amrex::Real>::max();
            });
    }
}

void
snapshotFields (Fields const& source, Fields& snapshot) {
    auto const& boxes = source.B.boxArray();
    auto const& mapping = source.B.DistributionMap();
    for (auto* field :
         {&snapshot.B, &snapshot.U, &snapshot.T, &snapshot.pressure,
          &snapshot.charge, &snapshot.volume}) {
        if (!field->isDefined()) {
            field->define(boxes, mapping, 1, 1);
        }
    }
    amrex::MultiFab::Copy(snapshot.B, source.B, 0, 0, 1, snapshot.B.nGrow());
    amrex::MultiFab::Copy(snapshot.U, source.U, 0, 0, 1, snapshot.U.nGrow());
    amrex::MultiFab::Copy(snapshot.T, source.T, 0, 0, 1, snapshot.T.nGrow());
    amrex::MultiFab::Copy(snapshot.pressure, source.pressure, 0, 0, 1,
                          snapshot.pressure.nGrow());
    amrex::MultiFab::Copy(snapshot.charge, source.charge, 0, 0, 1,
                          snapshot.charge.nGrow());
    amrex::MultiFab::Copy(snapshot.volume, source.volume, 0, 0, 1,
                          snapshot.volume.nGrow());
}

amrex::FArrayBox
copyToHost (amrex::MultiFab const& field, amrex::MFIter const& it) {
    amrex::FArrayBox host(field[it].box(), 1, amrex::The_Pinned_Arena());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                     field[it].dataPtr() + field[it].box().numPts(),
                     host.dataPtr());
    return host;
}

amrex::IArrayBox
copyToHost (amrex::iMultiFab const& field, amrex::MFIter const& it) {
    amrex::IArrayBox mask(field[it].box(), 1, amrex::The_Pinned_Arena());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                     field[it].dataPtr() + field[it].box().numPts(),
                     mask.dataPtr());
    return mask;
}

std::array<long double, 4>
inventory (amrex::MultiFab const& B, amrex::MultiFab const& U,
           amrex::MultiFab const& volume, amrex::Geometry const& geom,
           amrex::Real first_stage_dt = 0) {
    std::array<long double, 4> total{};
    auto const owner = B.OwnerMask(geom.periodicity());
    int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
    auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
    for (amrex::MFIter it(B); it.isValid(); ++it) {
        auto const Bh = copyToHost(B, it), Uh = copyToHost(U, it);
        auto const Vh = copyToHost(volume, it);
        auto const mask = copyToHost(*owner, it);
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1) && j < nz; ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                amrex::IntVect const node(i, j);
                if (!mask(node)) {
                    continue;
                }
                long double const magnetic_volume =
                    i == 0 ? MathConst::pi * dr * dr * dz / 4
                           : 2 * MathConst::pi * i * dr * dr * dz *
                                 (i == nr ? 0.5L : 1.0L);
                long double const field = Bh(node);
                total[0] +=
                    magnetic_volume * field * field / (2 * PhysConst::mu0);
                total[1] += static_cast<long double>(Vh(node)) * Uh(node);
                if (first_stage_dt > 0) {
                    long double const r = static_cast<long double>(i) / nr;
                    long double const phase = 2 * MathConst::pi * j / nz;
                    long double const shape = std::max(0.0L, 1 - 4 * r * r);
                    long double const pressure_dr =
                        -(64.0L / 3) * 1.e9L * r / geom.ProbHi(0) * shape *
                        shape * shape * (1 + 0.2L * std::cos(phase));
                    long double const reciprocal_dz =
                        -0.2L * 2 * MathConst::pi / geom.ProbLength(1) *
                        std::cos(phase) /
                        (1.e7L * std::pow(1 + 0.2L * std::sin(phase), 2));
                    long double const expected =
                        first_stage_dt * reciprocal_dz * pressure_dr;
                    long double const error =
                        field - 5 * r * (1 - r * r) - expected;
                    total[2] += magnetic_volume * error * error;
                    total[3] += magnetic_volume * expected * expected;
                }
            }
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, total.data(), 4, MPI_LONG_DOUBLE, MPI_SUM,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return total;
}

amrex::Real
minimumValue (amrex::MultiFab const& field) {
    amrex::Real minimum = std::numeric_limits<amrex::Real>::max();
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const host = copyToHost(field, it);
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                minimum = std::min(minimum, host(amrex::IntVect(i, j)));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(minimum);
    return minimum;
}

amrex::Real
minimumTemperatureFromEnergy (Fields const& state,
                              ElectronThermodynamicsExecutor eos) {
    amrex::Real minimum = std::numeric_limits<amrex::Real>::max();
    for (amrex::MFIter it(state.B); it.isValid(); ++it) {
        auto const energy = copyToHost(state.U, it);
        auto const charge = copyToHost(state.charge, it);
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                amrex::IntVect const node(i, j);
                minimum = std::min(
                    minimum,
                    eos.temperatureFromChargeDensityThermalEnergyDensity(
                        charge(node), energy(node)));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(minimum);
    return minimum;
}

long double
caloricError (Fields const& state, amrex::iMultiFab const& owner,
              ElectronThermodynamicsExecutor eos) {
    long double error = 0;
    for (amrex::MFIter it(state.B); it.isValid(); ++it) {
        auto const copy = [&it] (amrex::MultiFab const& field) {
            amrex::FArrayBox host(field[it].box(), 1,
                                  amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                             field[it].dataPtr() + field[it].box().numPts(),
                             host.dataPtr());
            return host;
        };
        auto const Uh = copy(state.U), Th = copy(state.T);
        auto const Rhoh = copy(state.charge);
        amrex::IArrayBox mask(owner[it].box(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, owner[it].dataPtr(),
                         owner[it].dataPtr() + owner[it].box().numPts(),
                         mask.dataPtr());
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                amrex::IntVect const node(i, j);
                if (!mask(node)) {
                    continue;
                }
                amrex::Real const charge = Rhoh(node);
                amrex::Real const energy = Uh(node);
                amrex::Real const expected_temperature =
                    eos.temperatureFromChargeDensityThermalEnergyDensity(
                        charge, energy);
                amrex::Real const expected_energy =
                    eos.stateFromChargeDensityTemperature(charge, Th(node))
                        .internal_energy_density;
                long double const temperature_error = static_cast<long double>(
                    std::abs(Th(node) - expected_temperature) /
                    expected_temperature);
                long double const energy_error = static_cast<long double>(
                    std::abs(Uh(node) - expected_energy) / energy);
                error = std::max(error, temperature_error);
                error = std::max(error, energy_error);
            }
        }
    }
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_LONG_DOUBLE, MPI_MAX,
                  amrex::ParallelDescriptor::Communicator());
#endif
    return error;
}

bool
bitwiseEqual (amrex::MultiFab const& left, amrex::MultiFab const& right) {
    AMREX_ALWAYS_ASSERT(left.nComp() == 1 && left.nGrow() == 1 &&
                        right.nGrow() == 1);
    int different = 0;
    for (amrex::MFIter it(left); it.isValid(); ++it) {
        auto const a = copyToHost(left, it), b = copyToHost(right, it);
        auto const bytes = static_cast<std::size_t>(left[it].box().numPts()) *
                           sizeof(amrex::Real);
        different |= std::memcmp(a.dataPtr(), b.dataPtr(), bytes) != 0;
    }
    amrex::ParallelDescriptor::ReduceIntMax(different);
    return different == 0;
}

void
assertRollback (Fields const& before, Fields const& after) {
    AMREX_ALWAYS_ASSERT(bitwiseEqual(before.B, after.B) &&
                        bitwiseEqual(before.U, after.U) &&
                        bitwiseEqual(before.T, after.T) &&
                        bitwiseEqual(before.pressure, after.pressure) &&
                        bitwiseEqual(before.charge, after.charge) &&
                        bitwiseEqual(before.volume, after.volume));
}
} // namespace

int
main (int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        int cells = 32, max_grid = 32, steps = 128;
        bool constant_charge = false;
        amrex::ParmParse pp("test");
        pp.query("cells", cells);
        pp.query("max_grid", max_grid);
        pp.query("steps", steps);
        pp.query("constant_charge", constant_charge);

        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
        amrex::RealBox const physical({0, 0}, {1.e-3_rt, 1.e-3_rt});
        int periodic[2]{0, 1};
        amrex::Geometry const geom(domain, &physical, 1, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(max_grid);
        boxes.convert(amrex::IntVect::TheNodeVector());
        amrex::DistributionMapping const mapping(boxes);

        ElectronThermodynamicsExecutor const eos{};
        Fields state, reference;
        initialize(state, boxes, mapping, geom, cells, constant_charge);
        initialize(reference, boxes, mapping, geom, cells, constant_charge);
        Fields original_state;
        snapshotFields(state, original_state);

        auto const owner = state.B.OwnerMask(geom.periodicity());
        auto const initial = inventory(state.B, state.U, state.volume, geom);
        auto const initial_reference =
            inventory(reference.B, reference.U, reference.volume, geom);
        AMREX_ALWAYS_ASSERT(initial[0] == initial_reference[0] &&
                            initial[1] == initial_reference[1]);

        AMREX_ALWAYS_ASSERT(warpx::hybrid::tryAdvanceRZPressureInductionStage(
            state.B, state.U, state.T, state.charge, state.volume, geom,
            1.e-10_rt, eos, amrex::Real(0.0)));
        AMREX_ALWAYS_ASSERT(warpx::hybrid::tryAdvanceRZPressureInduction(
            reference.B, reference.U, reference.pressure, reference.charge,
            reference.volume, geom, 1.e-10_rt));
        AMREX_ALWAYS_ASSERT(bitwiseEqual(state.B, reference.B) &&
                            bitwiseEqual(state.U, reference.U));

        AMREX_ALWAYS_ASSERT(caloricError(state, *owner, eos) < 1.e-13L);

        if (!constant_charge) {
            auto const analytic =
                inventory(state.B, state.U, state.volume, geom, 1.e-10_rt);
            long double const error = std::sqrt(analytic[2] / analytic[3]);
            amrex::Print() << "RZ stage first-stage relative L2 error=" << error
                           << '\n';
            AMREX_ALWAYS_ASSERT(error < 0.05L);
        }

        for (int stage = 1; stage < steps; ++stage) {
            AMREX_ALWAYS_ASSERT(
                warpx::hybrid::tryAdvanceRZPressureInductionStage(
                    state.B, state.U, state.T, state.charge, state.volume, geom,
                    1.e-10_rt, eos, amrex::Real(0.0)));
        }
        AMREX_ALWAYS_ASSERT(caloricError(state, *owner, eos) < 1.e-13L);

        auto const final = inventory(state.B, state.U, state.volume, geom);
        long double const defect =
            (final[0] - initial[0]) + (final[1] - initial[1]);
        long double const transfer = std::abs(final[0] - initial[0]);
        amrex::Print() << "RZ pressure stage magnetic transfer="
                       << final[0] - initial[0]
                       << " caloric transfer=" << final[1] - initial[1]
                       << " energy defect=" << defect << '\n';
        AMREX_ALWAYS_ASSERT(std::abs(defect) <
                            1.e-11L * (initial[0] + initial[1]));

        if (constant_charge) {
            Fields huge_dt, huge_dt_before;
            initialize(huge_dt, boxes, mapping, geom, cells, true);
            snapshotFields(huge_dt, huge_dt_before);
            AMREX_ALWAYS_ASSERT(
                warpx::hybrid::tryAdvanceRZPressureInductionStage(
                    huge_dt.B, huge_dt.U, huge_dt.T, huge_dt.charge,
                    huge_dt.volume, geom, 1.e6_rt * 1.e-10_rt, eos,
                    amrex::Real(0.0)));
            // This is an accepted null-source control, not a rollback. At this
            // artificial timestep, centered-curl roundoff can create tiny B.
            // Require the exact primitive result rather than unchanged B bits.
            AMREX_ALWAYS_ASSERT(warpx::hybrid::tryAdvanceRZPressureInduction(
                huge_dt_before.B, huge_dt_before.U, huge_dt_before.pressure,
                huge_dt_before.charge, huge_dt_before.volume, geom,
                1.e6_rt * 1.e-10_rt));
            AMREX_ALWAYS_ASSERT(bitwiseEqual(huge_dt_before.B, huge_dt.B) &&
                                bitwiseEqual(huge_dt_before.U, huge_dt.U));
            AMREX_ALWAYS_ASSERT(caloricError(huge_dt, *owner, eos) < 1.e-13L);
        }

        Fields rejected, before;
        // A nonparallel-gradient control is needed to force operator failure;
        // a constant-charge null source need not do work even for extreme B.
        initialize(rejected, boxes, mapping, geom, cells, false);
        setLargeMagneticField(rejected);
        snapshotFields(rejected, before);
        AMREX_ALWAYS_ASSERT(!warpx::hybrid::tryAdvanceRZPressureInductionStage(
            rejected.B, rejected.U, rejected.T, rejected.charge,
            rejected.volume, geom, 1.e-10_rt, eos, amrex::Real(0.0)));
        assertRollback(before, rejected);

        // The ideal executor reports floor 0 and ceiling max Real.  This is
        // explicitly an invalid-initial-state boundary: tiny positive energy at
        // maximum charge density inverts to zero temperature.
        initialize(rejected, boxes, mapping, geom, cells, constant_charge);
        rejected.charge.setVal(std::numeric_limits<amrex::Real>::max(), 0, 1);
        rejected.U.setVal(std::numeric_limits<amrex::Real>::min(), 0, 1);
        rejected.T.setVal(1.0_rt, 0, 1);
        snapshotFields(rejected, before);
        AMREX_ALWAYS_ASSERT(!warpx::hybrid::tryAdvanceRZPressureInductionStage(
            rejected.B, rejected.U, rejected.T, rejected.charge,
            rejected.volume, geom, 1.e-10_rt, eos, amrex::Real(0.0)));
        assertRollback(before, rejected);

        Fields thermal_reference, thermal_initial;
        initialize(thermal_reference, boxes, mapping, geom, cells, false);
        initialize(thermal_initial, boxes, mapping, geom, cells, false);
        initializeUniformTemperatureProfile(thermal_reference, cells);
        initializeUniformTemperatureProfile(thermal_initial, cells);
        amrex::Real const thermal_dt = 1.e-11_rt;
        AMREX_ALWAYS_ASSERT(warpx::hybrid::tryAdvanceRZPressureInduction(
            thermal_reference.B, thermal_reference.U,
            thermal_reference.pressure, thermal_reference.charge,
            thermal_reference.volume, geom, thermal_dt));
        amrex::Real const initial_temperature = minimumValue(thermal_initial.T);
        amrex::Real const primitive_temperature =
            minimumTemperatureFromEnergy(thermal_reference, eos);
        AMREX_ALWAYS_ASSERT(amrex::Math::isfinite(primitive_temperature) &&
                            primitive_temperature > amrex::Real(0.0) &&
                            primitive_temperature < initial_temperature);
        amrex::Real const stage_floor =
            amrex::Real(0.5) * (primitive_temperature + initial_temperature);
        Fields thermal_rejected, thermal_before;
        initialize(thermal_rejected, boxes, mapping, geom, cells, false);
        initializeUniformTemperatureProfile(thermal_rejected, cells);
        snapshotFields(thermal_rejected, thermal_before);
        AMREX_ALWAYS_ASSERT(!warpx::hybrid::tryAdvanceRZPressureInductionStage(
            thermal_rejected.B, thermal_rejected.U, thermal_rejected.T,
            thermal_rejected.charge, thermal_rejected.volume, geom, thermal_dt,
            eos, stage_floor));
        assertRollback(thermal_before, thermal_rejected);
        amrex::Print() << "RZ stage candidate temperature floor=" << stage_floor
                       << '\n';

        if (constant_charge) {
            AMREX_ALWAYS_ASSERT(state.B.norm0(0) < 1.e-10_rt);
            AMREX_ALWAYS_ASSERT(bitwiseEqual(state.U, original_state.U));
        } else {
            AMREX_ALWAYS_ASSERT(transfer > 1.e-6L * initial[1]);
            AMREX_ALWAYS_ASSERT(std::abs(defect) < 1.e-8L * transfer);
        }
    }
    amrex::Finalize();
}

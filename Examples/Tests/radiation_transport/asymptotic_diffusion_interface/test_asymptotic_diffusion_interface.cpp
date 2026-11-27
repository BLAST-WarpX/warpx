/* Copyright 2026 The WarpX Community
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/PacketDiffusionInterface.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
    using warpx::radiation::AsymptoticDiffusionBoundaryFace;
    using warpx::radiation::AsymptoticDiffusionCell;
    using warpx::radiation::AsymptoticDiffusionControls;
    using warpx::radiation::AsymptoticDiffusionInput;
    using warpx::radiation::AsymptoticDiffusionInteriorFace;
    using warpx::radiation::CrossedPacketAtDiffusionInterface;
    using warpx::radiation::PacketDiffusionInterfaceInput;
    using Vector = std::array<double, 3>;

    int failures = 0;

    void
    check (bool condition, std::string const& name)
    {
        if (!condition)
        {
            ++failures;
            std::cerr << "FAILED: " << name << '\n';
        }
    }

    bool
    close (double left, double right, double relative = 2.0e-11,
           double absolute = 1.0e-13)
    {
        double const scale = std::max({1.0, std::abs(left), std::abs(right)});
        return std::abs(left - right) <=
               std::max(absolute, relative * scale);
    }

    AsymptoticDiffusionCell
    makeCell (double energy, std::array<double, 3> widths,
              std::array<double, 3> opacity)
    {
        long double volume = 1.0L;
        for (double width : widths)
        {
            if (width > 0.0)
            {
                volume *= static_cast<long double>(width);
            }
        }
        return AsymptoticDiffusionCell{energy, static_cast<double>(volume), widths,
                                       opacity};
    }

    AsymptoticDiffusionInteriorFace
    makeInterior (std::size_t left, std::size_t right, int axis, double area,
                  AsymptoticDiffusionCell const& left_cell,
                  AsymptoticDiffusionCell const& right_cell)
    {
        return AsymptoticDiffusionInteriorFace{
            left, right, axis, area, left_cell.widths[axis], right_cell.widths[axis],
            left_cell.opacity[axis], right_cell.opacity[axis]};
    }

    AsymptoticDiffusionBoundaryFace
    makeBoundary (std::size_t cell, int axis, int sign,
                  AsymptoticDiffusionCell const& cell_record)
    {
        long double area = 1.0L;
        for (int d = 0; d < 3; ++d)
        {
            if (d != axis && cell_record.widths[d] > 0.0)
            {
                area *= static_cast<long double>(cell_record.widths[d]);
            }
        }
        return AsymptoticDiffusionBoundaryFace{
            cell, axis, sign, static_cast<double>(area),
            cell_record.widths[axis], cell_record.opacity[axis]};
    }

    AsymptoticDiffusionControls
    testControls (double dt, double speed_of_light = 2.0)
    {
        AsymptoticDiffusionControls controls;
        controls.dt = dt;
        controls.speed_of_light = speed_of_light;
        return controls;
    }

    AsymptoticDiffusionInput
    makeUnequalPair ()
    {
        std::array<double, 3> const left_opacity{2.3, 4.1, 5.9};
        std::array<double, 3> const right_opacity{5.7, 3.3, 7.1};
        AsymptoticDiffusionInput input;
        input.dimension = 3;
        input.cells = {makeCell(17.0, {0.7, 2.7, 3.1}, left_opacity),
                       makeCell(9.0, {1.9, 2.7, 3.1}, right_opacity)};
        input.interior_faces.push_back(makeInterior(
            0, 1, 0, 2.7 * 3.1, input.cells[0], input.cells[1]));
        return input;
    }

    void
    testUnequalHalfCellLawsAndPressureIdentity ()
    {
        AsymptoticDiffusionInput input = makeUnequalPair();
        AsymptoticDiffusionControls const controls = testControls(1.0e-9);
        auto const result =
            warpx::radiation::AdvanceAsymptoticDiffusion(input, controls);
        check(result.accepted, "unequal pair solve accepted");
        if (!result.accepted || !result.candidate.has_value())
        {
            return;
        }

        auto const& candidate = *result.candidate;
        double const e_left = candidate.energy_densities[0];
        double const e_right = candidate.energy_densities[1];
        double const tau_left = 2.3 * 0.7;
        double const tau_right = 5.7 * 1.9;
        double const expected_current =
            -2.0 * controls.speed_of_light * (e_right - e_left) /
            (3.0 * (tau_left + tau_right));
        double const expected_face_density =
            (tau_right * e_left + tau_left * e_right) / (tau_left + tau_right);
        check(close(candidate.interior_faces[0].current_left_to_right,
                    expected_current),
              "unequal D/D exact current");
        check(close(candidate.interior_faces[0].face_energy_density,
                    expected_face_density),
              "unequal weighted side-opacity face density");
        check(close(candidate.interior_faces[0].pressure,
                    expected_face_density / 3.0),
              "unequal face pressure is e_face/3");

        auto pressureOnFace = [&candidate] (std::size_t cell, int sign) {
            for (auto const& face : candidate.cells[cell].faces)
            {
                if (face.axis == 0 && face.outward_sign == sign)
                {
                    return face.pressure;
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
        double const pi_left_plus = pressureOnFace(0, +1);
        double const pi_left_minus = pressureOnFace(0, -1);
        double const pi_right_plus = pressureOnFace(1, +1);
        double const pi_right_minus = pressureOnFace(1, -1);
        double const left_identity = -controls.dt * input.cells[0].volume *
                                     (pi_left_plus - pi_left_minus) /
                                     input.cells[0].widths[0];
        double const right_identity = -controls.dt * input.cells[1].volume *
                                      (pi_right_plus - pi_right_minus) /
                                      input.cells[1].widths[0];
        check(close(candidate.cells[0].estimator_impulse[0], left_identity),
              "non-unit left pressure-estimator identity");
        check(close(candidate.cells[1].estimator_impulse[0], right_identity),
              "non-unit right pressure-estimator identity");

        double const left_qa = input.cells[0].volume *
                               candidate.interior_faces[0].current_left_to_right /
                               (2.0 * controls.speed_of_light);
        check(close(candidate.cells[0].radiation_inertia[0], left_qa),
              "non-unit Q_A diagnostic");
        check(std::all_of(candidate.cells[0].closure_diagnostic.begin(),
                          candidate.cells[0].closure_diagnostic.end(),
                          [](double value) { return std::isfinite(value); }),
              "closure diagnostic finite");
    }

    void
    testIndependentTwoCellReferenceAndSubdivision ()
    {
        AsymptoticDiffusionInput input;
        input.dimension = 1;
        input.cells = {makeCell(13.0, {0.83, 0.0, 0.0}, {2.9, 0.0, 0.0}),
                       makeCell(5.0, {1.37, 0.0, 0.0}, {6.1, 0.0, 0.0})};
        input.interior_faces.push_back(
            makeInterior(0, 1, 0, 1.0, input.cells[0], input.cells[1]));

        double const total_energy = 18.0;
        double const volume_left = input.cells[0].volume;
        double const volume_right = input.cells[1].volume;
        double const conductance =
            2.0 * 2.0 / (3.0 * (2.9 * 0.83 + 6.1 * 1.37));
        double const decay_rate = conductance * (1.0 / volume_left + 1.0 / volume_right);
        double const initial_difference =
            input.cells[0].energy / volume_left -
            input.cells[1].energy / volume_right;

        double const dt = 0.03;
        AsymptoticDiffusionControls const controls = testControls(dt);
        auto one_step = warpx::radiation::AdvanceAsymptoticDiffusion(input, controls);
        check(one_step.accepted, "two-cell backward Euler accepted");
        if (!one_step.accepted)
        {
            std::cerr << "two-cell rejection: " << one_step.rejection_reason << '\n';
        }
        if (one_step.accepted && one_step.candidate.has_value())
        {
            double const backward_factor =
                1.0 / (1.0 + conductance * dt * (1.0 / volume_left + 1.0 / volume_right));
            double const expected_difference = initial_difference * backward_factor;
            double const expected_left =
                (total_energy + volume_right * expected_difference) /
                (volume_left + volume_right);
            double const expected_right =
                (total_energy - volume_left * expected_difference) /
                (volume_left + volume_right);
            check(close(one_step.candidate->energy_densities[0], expected_left),
                  "independent one-step left density");
            check(close(one_step.candidate->energy_densities[1], expected_right),
                  "independent one-step right density");
        }

        auto subdivide = [&] (int steps) {
            AsymptoticDiffusionInput local = input;
            AsymptoticDiffusionControls local_controls = testControls(dt / steps);
            for (int step = 0; step < steps; ++step)
            {
                auto result = warpx::radiation::AdvanceAsymptoticDiffusion(
                    local, local_controls);
                if (!result.accepted || !result.candidate.has_value())
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                local.cells[0].energy = result.candidate->energies[0];
                local.cells[1].energy = result.candidate->energies[1];
            }
            return local.cells[0].energy / local.cells[0].volume;
        };

        double const exact_left =
            (total_energy +
             volume_right * initial_difference * std::exp(-decay_rate * dt)) /
            (volume_left + volume_right);
        double const coarse = subdivide(4);
        double const fine = subdivide(32);
        double const coarse_error = std::abs(coarse - exact_left);
        double const fine_error = std::abs(fine - exact_left);
        check(std::isfinite(coarse) && std::isfinite(fine),
              "subdivided solves accepted");
        check(fine_error < coarse_error, "subdivision reduces error");
        check(coarse_error / fine_error > 5.0 && coarse_error / fine_error < 11.0,
              "first-order timestep subdivision convergence");
    }

    void
    testColdThickAdmissionAnalytic ()
    {
        PacketDiffusionInterfaceInput input;
        input.diffusion.dimension = 1;
        input.diffusion.cells.push_back(
            makeCell(0.0, {0.3, 0.0, 0.0}, {1000.0, 0.0, 0.0}));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, +1, input.diffusion.cells[0]));
        input.packets.push_back(CrossedPacketAtDiffusionInterface{0, 2.0e-8, 0.4});

        AsymptoticDiffusionControls const controls = testControls(1.0e-9);
        auto const result = warpx::radiation::AdvancePacketDiffusionInterface(
            input, controls);
        check(result.accepted, "cold thick packet admission accepted");
        if (!result.accepted || !result.candidate.has_value())
        {
            return;
        }

        double const optical_depth = 300.0;
        double const probability =
            warpx::radiation::PacketDiffusionInterfaceProbability(
                optical_depth, 0.4, controls.marshak_extrapolation_lambda);
        double const admitted = probability * 2.0e-8;
        double const leakage_coefficient =
            2.0 * controls.speed_of_light /
            (3.0 * optical_depth + 6.0 * controls.marshak_extrapolation_lambda);
        double const expected_density =
            admitted / (input.diffusion.cells[0].volume +
                        controls.dt * input.diffusion.boundary_faces[0].area *
                            leakage_coefficient);
        auto const& candidate = result.candidate->diffusion;
        check(close(candidate.energy_densities[0], expected_density),
              "cold thick analytic backward-Euler density");
        check(close(result.candidate->packets[0].admitted_energy, admitted),
              "deterministic admitted packet energy");
        check(close(result.candidate->packets[0].reflected_energy,
                    (1.0 - probability) * 2.0e-8),
              "deterministic reflected packet energy");
        check(close(result.candidate->packets[0].admitted_energy +
                        result.candidate->packets[0].reflected_energy,
                    2.0e-8),
              "packet split conserves already-crossed energy");
        check(close(candidate.boundary_faces[0].leaked_energy,
                    controls.dt * leakage_coefficient * expected_density),
              "cold thick leakage energy");
        check(close(candidate.energies[0] - input.diffusion.cells[0].energy +
                        candidate.boundary_faces[0].net_outward_energy,
                    0.0),
              "cold admission/leakage cell energy account");
    }

    void
    testMultiStepColdThickThinRegion ()
    {
        PacketDiffusionInterfaceInput input;
        input.diffusion.dimension = 1;
        for (int cell = 0; cell < 3; ++cell)
        {
            input.diffusion.cells.push_back(
                makeCell(0.0, {0.2, 0.0, 0.0}, {500.0, 0.0, 0.0}));
        }
        input.diffusion.interior_faces.push_back(
            makeInterior(0, 1, 0, 1.0, input.diffusion.cells[0],
                         input.diffusion.cells[1]));
        input.diffusion.interior_faces.push_back(
            makeInterior(1, 2, 0, 1.0, input.diffusion.cells[1],
                         input.diffusion.cells[2]));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, -1, input.diffusion.cells[0]));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(2, 0, +1, input.diffusion.cells[2]));
        input.packets.push_back(CrossedPacketAtDiffusionInterface{0, 4.0e-7, 0.8});

        AsymptoticDiffusionControls const controls = testControls(2.0e-10);
        for (int step = 0; step < 5; ++step)
        {
            auto const result = warpx::radiation::AdvancePacketDiffusionInterface(
                input, controls);
            check(result.accepted, "multi-step thick/thin step accepted");
            if (!result.accepted || !result.candidate.has_value())
            {
                return;
            }
            auto const& candidate = result.candidate->diffusion;
            for (std::size_t cell = 0; cell < input.diffusion.cells.size(); ++cell)
            {
                double account = candidate.energies[cell] -
                                 input.diffusion.cells[cell].energy;
                for (auto const& face : candidate.cells[cell].faces)
                {
                    if (!face.reflecting)
                    {
                        account += face.outward_energy;
                    }
                }
                check(close(account, 0.0, 2.0e-10, 2.0e-19),
                      "multi-step cell energy account");
                input.diffusion.cells[cell].energy = candidate.energies[cell];
            }
            check(candidate.interior_faces[0].current_left_to_right > 0.0,
                  "thick interior current follows admitted-side gradient");
            check(candidate.interior_faces[1].current_left_to_right > 0.0,
                  "thick current advances toward thin leakage boundary");
            check(candidate.boundary_faces[1].leaked_energy > 0.0,
                  "thin D/K boundary leaks radiation");
        }
        check(input.diffusion.cells[0].energy > input.diffusion.cells[1].energy &&
                  input.diffusion.cells[1].energy > input.diffusion.cells[2].energy,
              "multi-step cold thick profile advances");
    }

    void
    testSteadyFluxSigns ()
    {
        AsymptoticDiffusionInput input;
        input.dimension = 1;
        input.cells = {makeCell(12.0, {1.1, 0.0, 0.0}, {9.0, 0.0, 0.0}),
                       makeCell(7.0, {0.8, 0.0, 0.0}, {4.0, 0.0, 0.0}),
                       makeCell(3.0, {1.4, 0.0, 0.0}, {11.0, 0.0, 0.0})};
        input.interior_faces.push_back(
            makeInterior(0, 1, 0, 1.0, input.cells[0], input.cells[1]));
        input.interior_faces.push_back(
            makeInterior(1, 2, 0, 1.0, input.cells[1], input.cells[2]));
        auto const result = warpx::radiation::AdvanceAsymptoticDiffusion(
            input, testControls(1.0e-9));
        check(result.accepted, "steady-sign solve accepted");
        if (!result.accepted || !result.candidate.has_value())
        {
            return;
        }
        auto const& candidate = *result.candidate;
        for (auto const& face : candidate.interior_faces)
        {
            check(face.current_left_to_right > 0.0,
                  "rightward steady diffusion current sign");
            check(face.left_outward_energy > 0.0 && face.right_outward_energy < 0.0,
                  "owned outward D/D energies have opposite signs");
        }
        for (auto const& cell : candidate.cells)
        {
            check(cell.estimator_impulse[0] > 0.0,
                  "homogeneous rightward flux gives positive D estimator");
            check(cell.radiation_inertia[0] > 0.0,
                  "homogeneous rightward Q_A has transport sign");
        }
    }

    void
    testProbabilityLimitsAndThinRejection ()
    {
        double const thick = warpx::radiation::PacketDiffusionInterfaceProbability(
            200.0, 0.6);
        double const thick_limit =
            4.0 * (1.0 + 1.5 * 0.6) / (3.0 * 200.0);
        check(close(thick / thick_limit, 1.0, 1.0e-2),
              "thick optical-depth probability limit");
        check(std::isnan(warpx::radiation::PacketDiffusionInterfaceProbability(
                  0.1, 1.0)),
              "thin invalid probability is NaN rather than clipped");

        PacketDiffusionInterfaceInput input;
        input.diffusion.dimension = 1;
        input.diffusion.cells.push_back(
            makeCell(1.0, {0.1, 0.0, 0.0}, {1.0, 0.0, 0.0}));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, +1, input.diffusion.cells[0]));
        input.packets.push_back(CrossedPacketAtDiffusionInterface{0, 1.0, 1.0});
        PacketDiffusionInterfaceInput const unchanged = input;
        auto const result = warpx::radiation::AdvancePacketDiffusionInterface(
            input, testControls(1.0e-9));
        check(!result.accepted && !result.candidate.has_value(),
              "thin packet probability rejects candidate");
        check(input.diffusion.cells[0].energy == unchanged.diffusion.cells[0].energy &&
                  input.packets[0].energy == unchanged.packets[0].energy &&
                  input.packets[0].inward_cosine ==
                      unchanged.packets[0].inward_cosine,
              "rejected packet input remains unchanged");
    }

    void
    testObliqueThreeDimensionalForces ()
    {
        PacketDiffusionInterfaceInput input;
        input.diffusion.dimension = 3;
        input.diffusion.cells.push_back(
            makeCell(3.0, {0.4, 0.6, 0.8}, {30.0, 30.0, 30.0}));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, +1, input.diffusion.cells[0]));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 1, +1, input.diffusion.cells[0]));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 2, +1, input.diffusion.cells[0]));
        input.packets.push_back(CrossedPacketAtDiffusionInterface{0, 2.0e-6, 0.2});
        input.packets.push_back(CrossedPacketAtDiffusionInterface{1, 3.0e-6, 0.55});
        input.packets.push_back(CrossedPacketAtDiffusionInterface{2, 1.5e-6, 0.77});

        AsymptoticDiffusionControls const controls = testControls(1.0e-10);
        auto const result = warpx::radiation::AdvancePacketDiffusionInterface(
            input, controls);
        check(result.accepted, "oblique 3D interface accepted");
        if (!result.accepted || !result.candidate.has_value())
        {
            return;
        }
        auto const& candidate = result.candidate->diffusion;
        Vector expected_estimator{0.0, 0.0, 0.0};
        for (std::size_t face = 0; face < 3; ++face)
        {
            double const contribution =
                input.diffusion.cells[0].volume *
                input.diffusion.cells[0].opacity[face] *
                candidate.boundary_faces[face].net_outward_energy /
                (2.0 * input.diffusion.boundary_faces[face].area *
                 controls.speed_of_light);
            expected_estimator[face] = contribution;
            check(candidate.boundary_faces[face].outward_sign == +1,
                  "3D outward boundary sign");
        }
        for (int d = 0; d < 3; ++d)
        {
            check(close(candidate.cells[0].estimator_impulse[d],
                        expected_estimator[d]),
                  "oblique 3D independent estimator component");
            check(candidate.cells[0].estimator_impulse[d] != 0.0,
                  "all Cartesian force components represented");
        }
        for (auto const& packet : result.candidate->packets)
        {
            check(close(packet.admitted_energy + packet.reflected_energy,
                        packet.crossed_energy),
                  "oblique deterministic packet split conserves energy");
        }
    }

    void
    testZeroColdStart ()
    {
        AsymptoticDiffusionInput input;
        input.dimension = 2;
        input.cells = {makeCell(0.0, {0.9, 1.3, 0.0}, {7.0, 5.0, 0.0}),
                       makeCell(0.0, {1.1, 1.3, 0.0}, {3.0, 8.0, 0.0})};
        input.interior_faces.push_back(
            makeInterior(0, 1, 0, 1.3, input.cells[0], input.cells[1]));
        auto const result = warpx::radiation::AdvanceAsymptoticDiffusion(
            input, testControls(1.0e-9));
        check(result.accepted, "zero cold start accepted");
        if (result.accepted && result.candidate.has_value())
        {
            check(std::all_of(result.candidate->energies.begin(),
                              result.candidate->energies.end(),
                              [](double energy) { return energy == 0.0; }),
                  "zero cold start remains exactly zero");
            check(std::all_of(result.candidate->cells[0].closure_diagnostic.begin(),
                              result.candidate->cells[0].closure_diagnostic.end(),
                              [](double value) { return value == 0.0; }),
                  "zero cold closure diagnostic defaults to zero");
            check(std::any_of(result.candidate->cells[0].faces.begin(),
                              result.candidate->cells[0].faces.end(),
                              [](auto const& face) { return face.reflecting; }),
                  "absent Cartesian faces are reflecting diagnostics");
        }
    }

    void
    testInvalidInputsAndImmutableRejection ()
    {
        auto testRejected = [] (PacketDiffusionInterfaceInput const& input,
                                std::string const& name) {
            auto const result = warpx::radiation::AdvancePacketDiffusionInterface(
                input, testControls(1.0e-9));
            check(!result.accepted && !result.candidate.has_value() &&
                      !result.rejection_reason.empty(),
                  name);
        };

        PacketDiffusionInterfaceInput input;
        input.diffusion.dimension = 1;
        input.diffusion.cells = {makeCell(2.0, {0.5, 0.0, 0.0}, {10.0, 0.0, 0.0}),
                                 makeCell(1.0, {0.7, 0.0, 0.0}, {20.0, 0.0, 0.0})};
        input.diffusion.interior_faces.push_back(
            makeInterior(0, 1, 0, 1.0, input.diffusion.cells[0],
                         input.diffusion.cells[1]));
        input.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, -1, input.diffusion.cells[0]));

        PacketDiffusionInterfaceInput bad_index = input;
        bad_index.packets.push_back(CrossedPacketAtDiffusionInterface{1, 1.0, 0.5});
        testRejected(bad_index, "invalid boundary index rejected");

        PacketDiffusionInterfaceInput bad_nan = input;
        bad_nan.diffusion.cells[1].energy = std::numeric_limits<double>::quiet_NaN();
        testRejected(bad_nan, "NaN cell energy rejected");

        PacketDiffusionInterfaceInput duplicate_interior = input;
        duplicate_interior.diffusion.interior_faces.push_back(
            makeInterior(1, 0, 0, 1.0, input.diffusion.cells[1],
                         input.diffusion.cells[0]));
        testRejected(duplicate_interior, "reversed duplicate D/D ownership rejected");

        PacketDiffusionInterfaceInput duplicate_boundary = input;
        duplicate_boundary.diffusion.boundary_faces.push_back(
            makeBoundary(0, 0, -1, input.diffusion.cells[0]));
        testRejected(duplicate_boundary, "duplicate K/D ownership rejected");

        PacketDiffusionInterfaceInput bad_volume = input;
        bad_volume.diffusion.cells[0].volume *= 1.01;
        testRejected(bad_volume, "inconsistent rectangular volume rejected");

        PacketDiffusionInterfaceInput bad_area = input;
        bad_area.diffusion.interior_faces[0].area *= 1.02;
        testRejected(bad_area, "inconsistent D/D area rejected");

        PacketDiffusionInterfaceInput bad_dimension = input;
        bad_dimension.diffusion.dimension = 4;
        testRejected(bad_dimension, "unsupported Cartesian dimension rejected");

        PacketDiffusionInterfaceInput bad_old_diagnostic = input;
        bad_old_diagnostic.diffusion.old_radiation_inertia.push_back(Vector{1.0, 0.0, 0.0});
        testRejected(bad_old_diagnostic, "wrong-length old Q_A rejected");

        PacketDiffusionInterfaceInput too_many_cells = input;
        too_many_cells.diffusion.cells.resize(65);
        too_many_cells.diffusion.interior_faces.clear();
        too_many_cells.diffusion.boundary_faces.clear();
        too_many_cells.packets.clear();
        testRejected(too_many_cells, "explicit dense host cell limit enforced");

        PacketDiffusionInterfaceInput const before = input;
        auto const rejected = warpx::radiation::AdvancePacketDiffusionInterface(
            duplicate_interior, testControls(0.0));
        check(!rejected.accepted && !rejected.candidate.has_value(),
              "invalid controls rejected");
        check(duplicate_interior.diffusion.cells.size() == before.diffusion.cells.size() &&
                  duplicate_interior.diffusion.interior_faces.size() ==
                      before.diffusion.interior_faces.size() + 1 &&
                  duplicate_interior.diffusion.cells[0].energy ==
                      before.diffusion.cells[0].energy,
              "geometric rejection input remains unchanged");
    }
} // namespace

int
main ()
{
    testUnequalHalfCellLawsAndPressureIdentity();
    testIndependentTwoCellReferenceAndSubdivision();
    testColdThickAdmissionAnalytic();
    testMultiStepColdThickThinRegion();
    testSteadyFluxSigns();
    testProbabilityLimitsAndThinRejection();
    testObliqueThreeDimensionalForces();
    testZeroColdStart();
    testInvalidInputsAndImmutableRejection();

    if (failures != 0)
    {
        std::cerr << failures << " asymptotic diffusion interface check(s) failed\n";
        return 1;
    }
    std::cout << "all asymptotic diffusion interface checks passed\n";
    return 0;
}

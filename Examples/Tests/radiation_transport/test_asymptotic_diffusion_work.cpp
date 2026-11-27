/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "Radiation/AsymptoticDiffusion.H"
#include "Radiation/PacketDiffusionInterface.H"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void Require (bool const condition, std::string const& message)
    {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void Near (double const actual, double const expected, double const scale,
               std::string const& message)
    {
        Require(std::isfinite(actual) && std::abs(actual - expected) <= 2.e-13 * scale,
                message);
    }

    warpx::radiation::AsymptoticDiffusionInput ClosedCell (double const energy)
    {
        warpx::radiation::AsymptoticDiffusionInput input;
        input.dimension = 3;
        input.cells.push_back({energy, 30.0, {2.0, 3.0, 5.0}, {9.0, 9.0, 9.0}});
        return input;
    }
}

int main ()
{
    using warpx::radiation::AdvanceAsymptoticDiffusion;
    using warpx::radiation::AsymptoticDiffusionControls;
    try {
        AsymptoticDiffusionControls controls;
        controls.dt = 1.0;
        controls.speed_of_light = 3.0;

        // A prescribed scalar debit is authoritative even when this trial's
        // force is zero. The native outer work iteration owns consistency.
        for (double const scale : {1.0, 1.e-100, 1.e100}) {
            for (double const signed_fraction : {0.4, -0.7}) {
                auto input = ClosedCell(scale);
                double const work = signed_fraction * scale;
                input.requested_work = {work};
                auto const result = AdvanceAsymptoticDiffusion(input, controls);
                Require(result.accepted && result.candidate.has_value(),
                        "closed prescribed-work step rejected: " + result.rejection_reason);
                auto const& candidate = *result.candidate;
                Near(candidate.energies[0], scale - work, scale, "wrong signed work debit");
                Require(candidate.requested_work.size() == 1, "missing accepted work vector");
                Near(candidate.requested_work[0], work, scale, "work vector not propagated");
                Near(candidate.total_requested_work, work, scale, "wrong total work");
                Near(candidate.cells[0].requested_work, work, scale, "wrong cell work");
                Near(candidate.cells[0].energy_balance_residual, 0.0, scale,
                     "cell work ledger does not close");
                Near(candidate.energy_balance_residual, 0.0, scale,
                     "global work ledger does not close");
                Near(input.cells[0].energy, scale, scale, "input energy was mutated");
            }
        }

        // A negative source-row RHS is allowed when implicit neighbor inflow
        // pays the work. Check against an independent two-by-two elimination.
        auto input = ClosedCell(0.0);
        input.cells.push_back({60.0, 60.0, {4.0, 3.0, 5.0}, {4.0, 4.0, 4.0}});
        input.interior_faces.push_back({0, 1, 0, 15.0, 2.0, 4.0, 9.0, 4.0});
        input.requested_work = {0.1, 0.0};
        auto const result = AdvanceAsymptoticDiffusion(input, controls);
        Require(result.accepted && result.candidate.has_value(),
                "neighbor-supported work step rejected: " + result.rejection_reason);
        double const conductance = 15.0 / 17.0;
        double const left_diag = 30.0 + conductance;
        double const right_diag = 60.0 + conductance;
        double const determinant = left_diag * right_diag - conductance * conductance;
        double const density_left = (-0.1 * right_diag + conductance * 60.0) / determinant;
        double const density_right = (-0.1 * conductance + left_diag * 60.0) / determinant;
        Near(result.candidate->energies[0], 30.0 * density_left, 60.0,
             "neighbor-supported work differs from independent elimination");
        Near(result.candidate->energies[1], 60.0 * density_right, 60.0,
             "neighbor-supported donor energy is wrong");
        Near(result.candidate->energies[0] + result.candidate->energies[1] + 0.1,
             60.0, 60.0, "closed two-cell energy plus work is not conserved");
        for (auto const& cell : result.candidate->cells) {
            Near(cell.energy_balance_residual, 0.0, 60.0, "two-cell local ledger failed");
            for (int axis = 0; axis < 3; ++axis) {
                double pressure_force = 0.0;
                double pressure_scale = 0.0;
                for (auto const& face : cell.faces) {
                    if (face.axis == axis) {
                        pressure_force -= controls.dt * face.area * face.pressure *
                                          face.outward_sign;
                        pressure_scale += controls.dt * face.area * std::abs(face.pressure);
                    }
                }
                Near(cell.estimator_impulse[axis], pressure_force, pressure_scale,
                     "pressure and impulse must use the same face state");
                Near(cell.closure_diagnostic[axis], cell.radiation_inertia[axis],
                     std::abs(cell.radiation_inertia[axis]) +
                         controls.speed_of_light * pressure_scale,
                     "omitted inertia was hidden or double-counted");
            }
        }

        // Sub-unit metrics still require relative geometric consistency.
        auto microscopic = ClosedCell(1.e-100);
        microscopic.cells[0].widths = {1.e-6, 2.e-6, 3.e-6};
        microscopic.cells[0].volume = 6.e-18;
        microscopic.boundary_faces.push_back({0, 0, 1, 6.e-12, 1.e-6, 9.0});
        auto bad_geometry = microscopic;
        bad_geometry.boundary_faces[0].area = 7.e-12;
        auto const bad_metric = AdvanceAsymptoticDiffusion(bad_geometry, controls);
        Require(!bad_metric.accepted && !bad_metric.candidate,
                "absolute metric floor admitted a malformed microscopic face");
        warpx::radiation::AsymptoticDiffusionInput overflow_geometry;
        overflow_geometry.dimension = 2;
        overflow_geometry.cells.push_back({1.0, 1.e300, {1.e200, 1.e200, 0.0},
                                           {1.0, 1.0, 0.0}});
        auto const overflow_metric = AdvanceAsymptoticDiffusion(overflow_geometry, controls);
        Require(!overflow_metric.accepted && !overflow_metric.candidate,
                "infinity comparison admitted unrepresentable geometric volume");

        auto old_inertia = ClosedCell(1.0);
        old_inertia.old_radiation_inertia = {{{0.3, -0.2, 0.1}}};
        auto const inertia_step = AdvanceAsymptoticDiffusion(old_inertia, controls);
        Require(inertia_step.accepted && inertia_step.candidate.has_value(),
                "old diagnostic state caused scalar rejection");
        for (int axis = 0; axis < 3; ++axis) {
            Near(inertia_step.candidate->cells[0].closure_diagnostic[axis],
                 -old_inertia.old_radiation_inertia[0][axis], 1.0,
                 "old omitted-inertia diagnostic was silently reset");
        }
        double const tangent_probability =
            warpx::radiation::PacketDiffusionInterfaceProbability(200.0, 0.0);
        Near(tangent_probability, 4.0 / (600.0 + 6.0 * 0.7104), tangent_probability,
             "valid tangent cosine endpoint rejected");
        auto empty_thin_interface = ClosedCell(1.0);
        empty_thin_interface.cells[0].opacity[0] = 0.1;
        empty_thin_interface.boundary_faces.push_back({0, 0, 1, 15.0, 2.0, 0.1});
        auto const thin_face = AdvanceAsymptoticDiffusion(empty_thin_interface, controls);
        Require(!thin_face.accepted && !thin_face.candidate,
                "thin D/K classification accepted merely because no packet arrived");

        for (auto const& work : {std::vector<double>{2.0},
                                std::vector<double>{0.0, 0.0},
                                std::vector<double>{std::numeric_limits<double>::quiet_NaN()},
                                std::vector<double>{std::numeric_limits<double>::infinity()}}) {
            auto rejected_input = ClosedCell(1.0);
            rejected_input.requested_work = work;
            auto const rejected = AdvanceAsymptoticDiffusion(rejected_input, controls);
            Require(!rejected.accepted && !rejected.candidate.has_value(),
                    "invalid or unavailable scalar work accepted");
            Require(rejected_input.cells[0].energy == 1.0,
                    "rejection changed caller-owned energy");
        }
        std::cout << "asymptotic diffusion signed work and immutable rejection: PASS\n";
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

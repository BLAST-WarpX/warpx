/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "AsymptoticDiffusion.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace warpx::radiation
{
    namespace
    {
        using Vector = AsymptoticDiffusionVector;

        constexpr std::size_t hard_max_cells = 64;

        AsymptoticDiffusionResult
        reject (std::string reason)
        {
            AsymptoticDiffusionResult result;
            result.accepted = false;
            result.rejection_reason = std::move(reason);
            return result;
        }

        bool
        finite_positive (double value)
        {
            return std::isfinite(value) && value > 0.0;
        }

        bool
        finite_nonnegative (double value)
        {
            return std::isfinite(value) && value >= 0.0;
        }

        bool
        close_metric (double left, double right, double tolerance)
        {
            if (!std::isfinite(left) || !std::isfinite(right))
            {
                return false;
            }
            double const scale = std::max(std::abs(left), std::abs(right));
            return std::abs(left - right) <= tolerance * scale;
        }

        double
        cross_section (AsymptoticDiffusionCell const& cell, int axis, int dimension)
        {
            long double area = 1.0L;
            for (int d = 0; d < dimension; ++d)
            {
                if (d != axis)
                {
                    area *= static_cast<long double>(cell.widths[d]);
                }
            }
            return static_cast<double>(area);
        }

        struct ValidatedGeometry
        {
            std::vector<double> admissions;
            std::vector<double> requested_work;
            std::vector<Vector> old_radiation_inertia;
            std::vector<std::array<int, 6>> occupied_slot;
        };

        std::optional<ValidatedGeometry>
        validateGeometry (AsymptoticDiffusionInput const& input,
                          AsymptoticDiffusionControls const& controls)
        {
            if (input.dimension < 1 || input.dimension > 3)
            {
                return std::nullopt;
            }
            if (input.cells.empty() || input.cells.size() > controls.max_cells ||
                input.cells.size() > hard_max_cells)
            {
                return std::nullopt;
            }

            std::size_t const number_cells = input.cells.size();
            for (AsymptoticDiffusionCell const& cell : input.cells)
            {
                if (!finite_nonnegative(cell.energy) || !finite_positive(cell.volume))
                {
                    return std::nullopt;
                }
                long double product = 1.0L;
                for (int d = 0; d < 3; ++d)
                {
                    bool const represented = d < input.dimension;
                    if (represented)
                    {
                        if (!finite_positive(cell.widths[d]) ||
                            !finite_positive(cell.opacity[d]))
                        {
                            return std::nullopt;
                        }
                        product *= static_cast<long double>(cell.widths[d]);
                    }
                    else if (cell.widths[d] != 0.0 || cell.opacity[d] != 0.0)
                    {
                        return std::nullopt;
                    }
                }
                if (!std::isfinite(product) ||
                    !close_metric(cell.volume, static_cast<double>(product),
                                  controls.geometry_tolerance))
                {
                    return std::nullopt;
                }
            }

            if (!input.boundary_admissions.empty() &&
                input.boundary_admissions.size() != input.boundary_faces.size())
            {
                return std::nullopt;
            }
            for (double admission : input.boundary_admissions)
            {
                if (!finite_nonnegative(admission))
                {
                    return std::nullopt;
                }
            }

            if (!input.requested_work.empty() && input.requested_work.size() != number_cells)
            {
                return std::nullopt;
            }
            for (double const work : input.requested_work)
            {
                if (!std::isfinite(work))
                {
                    return std::nullopt;
                }
            }

            if (!input.old_radiation_inertia.empty() &&
                input.old_radiation_inertia.size() != number_cells)
            {
                return std::nullopt;
            }
            for (Vector const& vector : input.old_radiation_inertia)
            {
                for (double component : vector)
                {
                    if (!std::isfinite(component))
                    {
                        return std::nullopt;
                    }
                }
            }

            ValidatedGeometry validated;
            validated.admissions.resize(input.boundary_faces.size(), 0.0);
            if (!input.boundary_admissions.empty())
            {
                validated.admissions = input.boundary_admissions;
            }
            validated.requested_work.assign(number_cells, 0.0);
            if (!input.requested_work.empty())
            {
                validated.requested_work = input.requested_work;
            }
            validated.old_radiation_inertia.resize(number_cells, Vector{0.0, 0.0, 0.0});
            if (!input.old_radiation_inertia.empty())
            {
                validated.old_radiation_inertia = input.old_radiation_inertia;
            }
            validated.occupied_slot.assign(number_cells, std::array<int, 6>{});

            auto occupy = [&validated](std::size_t cell, int axis, int sign) {
                int const slot = 2 * axis + (sign > 0 ? 1 : 0);
                bool const already_owned = validated.occupied_slot[cell][slot] != 0;
                validated.occupied_slot[cell][slot] = 1;
                return !already_owned;
            };

            std::set<std::array<std::size_t, 3>> interior_connections;
            for (AsymptoticDiffusionInteriorFace const& face : input.interior_faces)
            {
                if (face.left >= number_cells || face.right >= number_cells ||
                    face.left == face.right || face.axis < 0 || face.axis >= input.dimension ||
                    !finite_positive(face.area) || !finite_positive(face.left_width) ||
                    !finite_positive(face.right_width) || !finite_positive(face.left_opacity) ||
                    !finite_positive(face.right_opacity))
                {
                    return std::nullopt;
                }

                AsymptoticDiffusionCell const& left = input.cells[face.left];
                AsymptoticDiffusionCell const& right = input.cells[face.right];
                auto const connection = std::array<std::size_t, 3>{
                    std::min(face.left, face.right), std::max(face.left, face.right),
                    static_cast<std::size_t>(face.axis)};
                if (!interior_connections.insert(connection).second)
                {
                    return std::nullopt;
                }
                for (int axis = 0; axis < input.dimension; ++axis)
                {
                    if (axis != face.axis &&
                        !close_metric(left.widths[axis], right.widths[axis],
                                      controls.geometry_tolerance))
                    {
                        return std::nullopt;
                    }
                }
                if (!close_metric(face.left_width, left.widths[face.axis],
                                  controls.geometry_tolerance) ||
                    !close_metric(face.right_width, right.widths[face.axis],
                                  controls.geometry_tolerance) ||
                    !close_metric(face.left_opacity, left.opacity[face.axis],
                                  controls.geometry_tolerance) ||
                    !close_metric(face.right_opacity, right.opacity[face.axis],
                                  controls.geometry_tolerance))
                {
                    return std::nullopt;
                }

                double const left_area = cross_section(left, face.axis, input.dimension);
                double const right_area = cross_section(right, face.axis, input.dimension);
                if (!close_metric(face.area, left_area, controls.geometry_tolerance) ||
                    !close_metric(face.area, right_area, controls.geometry_tolerance))
                {
                    return std::nullopt;
                }
                if (!occupy(face.left, face.axis, +1) || !occupy(face.right, face.axis, -1))
                {
                    return std::nullopt;
                }
            }

            for (AsymptoticDiffusionBoundaryFace const& face : input.boundary_faces)
            {
                if (face.cell >= number_cells || face.axis < 0 ||
                    face.axis >= input.dimension ||
                    (face.outward_sign != -1 && face.outward_sign != 1) ||
                    !finite_positive(face.area) || !finite_positive(face.width) ||
                    !finite_positive(face.opacity))
                {
                    return std::nullopt;
                }
                AsymptoticDiffusionCell const& cell = input.cells[face.cell];
                if (!close_metric(face.width, cell.widths[face.axis],
                                  controls.geometry_tolerance) ||
                    !close_metric(face.opacity, cell.opacity[face.axis],
                                  controls.geometry_tolerance))
                {
                    return std::nullopt;
                }
                double const expected_area =
                    cross_section(cell, face.axis, input.dimension);
                if (!close_metric(face.area, expected_area, controls.geometry_tolerance))
                {
                    return std::nullopt;
                }
                if (!occupy(face.cell, face.axis, face.outward_sign))
                {
                    return std::nullopt;
                }
            }

            return validated;
        }

        bool
        solveDenseWithPartialPivoting (std::vector<std::vector<long double>>& matrix,
                                       std::vector<long double>& rhs,
                                       std::vector<long double>& solution)
        {
            std::size_t const number_cells = rhs.size();
            solution.assign(number_cells, 0.0L);

            for (std::size_t row = 0; row < number_cells; ++row)
            {
                long double scale = 0.0L;
                for (long double value : matrix[row])
                {
                    scale = std::max(scale, std::abs(value));
                }
                if (!std::isfinite(scale) || scale <= 0.0L)
                {
                    return false;
                }
                for (long double& value : matrix[row])
                {
                    value /= scale;
                }
                rhs[row] /= scale;
            }

            long double const minimum_pivot =
                256.0L * std::numeric_limits<long double>::epsilon();
            for (std::size_t column = 0; column < number_cells; ++column)
            {
                std::size_t pivot_row = column;
                long double pivot_value = std::abs(matrix[column][column]);
                for (std::size_t row = column + 1; row < number_cells; ++row)
                {
                    long double const candidate = std::abs(matrix[row][column]);
                    if (candidate > pivot_value)
                    {
                        pivot_value = candidate;
                        pivot_row = row;
                    }
                }
                if (!std::isfinite(pivot_value) || pivot_value <= minimum_pivot)
                {
                    return false;
                }
                std::swap(matrix[column], matrix[pivot_row]);
                std::swap(rhs[column], rhs[pivot_row]);

                for (std::size_t row = column + 1; row < number_cells; ++row)
                {
                    long double const factor = matrix[row][column] / matrix[column][column];
                    if (!std::isfinite(factor))
                    {
                        return false;
                    }
                    matrix[row][column] = 0.0L;
                    for (std::size_t next_column = column + 1; next_column < number_cells;
                         ++next_column)
                    {
                        matrix[row][next_column] -= factor * matrix[column][next_column];
                    }
                    rhs[row] -= factor * rhs[column];
                }
            }

            for (std::size_t reverse = number_cells; reverse > 0; --reverse)
            {
                std::size_t const row = reverse - 1;
                long double value = rhs[row];
                for (std::size_t column = row + 1; column < number_cells; ++column)
                {
                    value -= matrix[row][column] * solution[column];
                }
                solution[row] = value / matrix[row][row];
                if (!std::isfinite(solution[row]))
                {
                    return false;
                }
            }
            return true;
        }

        Vector
        scaledNormalVector (int axis, int sign, double scale)
        {
            Vector vector{0.0, 0.0, 0.0};
            vector[axis] = static_cast<double>(sign) * scale;
            return vector;
        }

        void
        addScaled (Vector& destination, Vector const& source, double scale)
        {
            for (std::size_t d = 0; d < destination.size(); ++d)
            {
                destination[d] += scale * source[d];
            }
        }

        bool
        finiteVector (Vector const& vector)
        {
            return std::all_of(vector.begin(), vector.end(),
                               [](double value) { return std::isfinite(value); });
        }
    } // namespace

    AsymptoticDiffusionResult
    AdvanceAsymptoticDiffusion (AsymptoticDiffusionInput const& input,
                                AsymptoticDiffusionControls const& controls)
    {
        if (!finite_positive(controls.dt) || !finite_positive(controls.speed_of_light) ||
            !finite_positive(controls.marshak_extrapolation_lambda) ||
            controls.max_cells == 0 || controls.max_cells > hard_max_cells ||
            !finite_positive(controls.solver_tolerance) ||
            !finite_positive(controls.geometry_tolerance))
        {
            return reject("Invalid asymptotic diffusion controls.");
        }

        std::optional<ValidatedGeometry> geometry = validateGeometry(input, controls);
        if (!geometry.has_value())
        {
            return reject("Invalid asymptotic diffusion geometry, ownership, or input values.");
        }

        std::size_t const number_cells = input.cells.size();
        std::vector<std::vector<long double>> matrix(
            number_cells, std::vector<long double>(number_cells, 0.0L));
        std::vector<long double> rhs(number_cells, 0.0L);
        std::vector<std::vector<long double>> original_matrix(
            number_cells, std::vector<long double>(number_cells, 0.0L));
        std::vector<long double> original_rhs(number_cells, 0.0L);

        for (std::size_t cell = 0; cell < number_cells; ++cell)
        {
            long double const volume = static_cast<long double>(input.cells[cell].volume);
            long double const old_energy =
                static_cast<long double>(input.cells[cell].energy);
            matrix[cell][cell] = volume;
            rhs[cell] = old_energy - static_cast<long double>(geometry->requested_work[cell]);
        }

        for (std::size_t face_index = 0; face_index < input.interior_faces.size(); ++face_index)
        {
            AsymptoticDiffusionInteriorFace const& face = input.interior_faces[face_index];
            long double const tau_left = static_cast<long double>(face.left_opacity) *
                                         static_cast<long double>(face.left_width);
            long double const tau_right = static_cast<long double>(face.right_opacity) *
                                          static_cast<long double>(face.right_width);
            long double const denominator = tau_left + tau_right;
            long double const conductance =
                2.0L * static_cast<long double>(controls.speed_of_light) *
                static_cast<long double>(face.area) *
                static_cast<long double>(controls.dt) / (3.0L * denominator);
            if (!std::isfinite(tau_left) || !std::isfinite(tau_right) ||
                !std::isfinite(denominator) || !std::isfinite(conductance) ||
                denominator <= 0.0L)
            {
                return reject("Non-finite interior diffusion conductance.");
            }
            matrix[face.left][face.left] += conductance;
            matrix[face.left][face.right] -= conductance;
            matrix[face.right][face.right] += conductance;
            matrix[face.right][face.left] -= conductance;
        }

        for (AsymptoticDiffusionBoundaryFace const& face : input.boundary_faces)
        {
            long double const tau = static_cast<long double>(face.opacity) *
                                    static_cast<long double>(face.width);
            long double const interface_denominator = 3.0L * tau +
                6.0L * static_cast<long double>(controls.marshak_extrapolation_lambda);
            // Classification must admit every incident angle, including a
            // later normal ray; absence of packets cannot make a thin D face valid.
            if (!std::isfinite(interface_denominator) || interface_denominator < 10.0L)
            {
                return reject("D/K opacity does not define a probability for every angle.");
            }
            long double const leakage_coefficient =
                2.0L * static_cast<long double>(controls.speed_of_light) /
                interface_denominator;
            long double const leakage = leakage_coefficient *
                                         static_cast<long double>(face.area) *
                                         static_cast<long double>(controls.dt);
            if (!std::isfinite(tau) || tau <= 0.0L ||
                !std::isfinite(leakage_coefficient) || !std::isfinite(leakage))
            {
                return reject("Non-finite D/K leakage coefficient.");
            }
            matrix[face.cell][face.cell] += leakage;
        }

        for (std::size_t boundary = 0; boundary < input.boundary_faces.size(); ++boundary)
        {
            std::size_t const cell = input.boundary_faces[boundary].cell;
            rhs[cell] += static_cast<long double>(geometry->admissions[boundary]);
        }

        original_matrix = matrix;
        original_rhs = rhs;
        std::vector<long double> long_solution;
        if (!solveDenseWithPartialPivoting(matrix, rhs, long_solution))
        {
            return reject("Dense asymptotic diffusion solve failed pivot or finiteness checks.");
        }

        std::vector<double> densities(number_cells, 0.0);
        std::vector<double> energies(number_cells, 0.0);
        for (std::size_t cell = 0; cell < number_cells; ++cell)
        {
            densities[cell] = static_cast<double>(long_solution[cell]);
            energies[cell] = static_cast<double>(static_cast<long double>(input.cells[cell].volume) *
                                                 long_solution[cell]);
            if (!finite_nonnegative(densities[cell]) || !finite_nonnegative(energies[cell]))
            {
                return reject("Backward-Euler solution produced a non-finite or negative energy.");
            }
        }

        for (std::size_t row = 0; row < number_cells; ++row)
        {
            long double lhs = original_matrix[row][row] *
                              static_cast<long double>(densities[row]);
            long double scale = std::abs(original_rhs[row]);
            scale = std::max(scale, std::abs(lhs));
            for (std::size_t column = 0; column < number_cells; ++column)
            {
                if (column != row)
                {
                    lhs += original_matrix[row][column] *
                           static_cast<long double>(densities[column]);
                    scale = std::max(
                        scale, std::abs(original_matrix[row][column] *
                                        static_cast<long double>(densities[column])));
                }
            }
            long double const residual = lhs - original_rhs[row];
            if (!std::isfinite(residual) || !std::isfinite(scale) ||
                std::abs(residual) >
                    static_cast<long double>(controls.solver_tolerance) * scale)
            {
                return reject("Backward-Euler row residual exceeded the acceptance tolerance.");
            }
        }

        AsymptoticDiffusionCandidate candidate;
        candidate.energies = energies;
        candidate.energy_densities = densities;
        candidate.interior_faces.resize(input.interior_faces.size());
        candidate.boundary_faces.resize(input.boundary_faces.size());
        candidate.cells.resize(number_cells);
        candidate.requested_work = geometry->requested_work;

        std::vector<std::vector<AsymptoticDiffusionCellFaceDiagnostics>> cell_faces(
            number_cells);
        std::vector<std::array<bool, 6>> present_faces(number_cells);

        auto addCellFace = [&] (std::size_t cell, int axis, int sign, double area,
                                double outward_current, double outward_energy,
                                double face_energy_density, bool reflecting,
                                Vector estimator_contribution) {
            if (!std::isfinite(outward_current) || !std::isfinite(outward_energy) ||
                !std::isfinite(face_energy_density) || !finiteVector(estimator_contribution))
            {
                return false;
            }
            AsymptoticDiffusionCellFaceDiagnostics diagnostics;
            diagnostics.axis = axis;
            diagnostics.outward_sign = sign;
            diagnostics.area = area;
            diagnostics.outward_current = outward_current;
            diagnostics.outward_energy = outward_energy;
            diagnostics.face_energy_density = face_energy_density;
            diagnostics.pressure = face_energy_density / 3.0;
            diagnostics.reflecting = reflecting;
            diagnostics.estimator_contribution = estimator_contribution;
            cell_faces[cell].push_back(diagnostics);
            present_faces[cell][2 * axis + (sign > 0 ? 1 : 0)] = true;
            return true;
        };

        for (std::size_t face_index = 0; face_index < input.interior_faces.size(); ++face_index)
        {
            AsymptoticDiffusionInteriorFace const& face = input.interior_faces[face_index];
            AsymptoticDiffusionCell const& left = input.cells[face.left];
            AsymptoticDiffusionCell const& right = input.cells[face.right];
            long double const tau_left = static_cast<long double>(face.left_opacity) *
                                         static_cast<long double>(face.left_width);
            long double const tau_right = static_cast<long double>(face.right_opacity) *
                                          static_cast<long double>(face.right_width);
            long double const denominator = tau_left + tau_right;
            long double const current =
                -2.0L * static_cast<long double>(controls.speed_of_light) *
                (static_cast<long double>(densities[face.right]) -
                 static_cast<long double>(densities[face.left])) /
                (3.0L * denominator);
            long double const face_density =
                (tau_right * static_cast<long double>(densities[face.left]) +
                 tau_left * static_cast<long double>(densities[face.right])) /
                denominator;
            long double const left_energy = static_cast<long double>(controls.dt) *
                                            static_cast<long double>(face.area) * current;
            long double const right_energy = -left_energy;

            AsymptoticDiffusionInteriorFaceDiagnostics& diagnostics =
                candidate.interior_faces[face_index];
            diagnostics.face = face_index;
            diagnostics.left = face.left;
            diagnostics.right = face.right;
            diagnostics.axis = face.axis;
            diagnostics.area = face.area;
            diagnostics.current_left_to_right = static_cast<double>(current);
            diagnostics.left_outward_energy = static_cast<double>(left_energy);
            diagnostics.right_outward_energy = static_cast<double>(right_energy);
            diagnostics.face_energy_density = static_cast<double>(face_density);
            diagnostics.pressure = static_cast<double>(face_density / 3.0L);
            if (!std::isfinite(current) || !std::isfinite(face_density) ||
                !std::isfinite(left_energy) || !finite_nonnegative(diagnostics.pressure))
            {
                return reject("Non-finite interior face reconstruction.");
            }

            long double const left_estimator =
                static_cast<long double>(left.volume) *
                static_cast<long double>(face.left_opacity) * left_energy /
                (2.0L * static_cast<long double>(face.area) *
                 static_cast<long double>(controls.speed_of_light));
            long double const right_estimator =
                static_cast<long double>(right.volume) *
                static_cast<long double>(face.right_opacity) * right_energy /
                (2.0L * static_cast<long double>(face.area) *
                 static_cast<long double>(controls.speed_of_light));
            if (!addCellFace(face.left, face.axis, +1, face.area,
                             static_cast<double>(current),
                             static_cast<double>(left_energy),
                             static_cast<double>(face_density), false,
                             scaledNormalVector(face.axis, +1,
                                                static_cast<double>(left_estimator))) ||
                !addCellFace(face.right, face.axis, -1, face.area,
                             static_cast<double>(-current),
                             static_cast<double>(right_energy),
                             static_cast<double>(face_density), false,
                             scaledNormalVector(face.axis, -1,
                                                static_cast<double>(right_estimator))))
            {
                return reject("Non-finite interior cell face diagnostic.");
            }
        }

        for (std::size_t face_index = 0; face_index < input.boundary_faces.size(); ++face_index)
        {
            AsymptoticDiffusionBoundaryFace const& face = input.boundary_faces[face_index];
            AsymptoticDiffusionCell const& cell = input.cells[face.cell];
            long double const tau = static_cast<long double>(face.opacity) *
                                    static_cast<long double>(face.width);
            long double const leakage_coefficient =
                2.0L * static_cast<long double>(controls.speed_of_light) /
                (3.0L * tau + 6.0L * static_cast<long double>(
                                          controls.marshak_extrapolation_lambda));
            long double const leaked_energy =
                static_cast<long double>(controls.dt) *
                static_cast<long double>(face.area) * leakage_coefficient *
                static_cast<long double>(densities[face.cell]);
            double const admitted_energy = geometry->admissions[face_index];
            long double const net_outward_energy = leaked_energy -
                                                   static_cast<long double>(admitted_energy);
            long double const outward_current =
                net_outward_energy / (static_cast<long double>(controls.dt) *
                                      static_cast<long double>(face.area));
            long double const face_density =
                static_cast<long double>(densities[face.cell]) -
                3.0L * tau * outward_current /
                    (2.0L * static_cast<long double>(controls.speed_of_light));

            AsymptoticDiffusionBoundaryFaceDiagnostics& diagnostics =
                candidate.boundary_faces[face_index];
            diagnostics.face = face_index;
            diagnostics.cell = face.cell;
            diagnostics.axis = face.axis;
            diagnostics.outward_sign = face.outward_sign;
            diagnostics.area = face.area;
            diagnostics.admitted_energy = admitted_energy;
            diagnostics.leaked_energy = static_cast<double>(leaked_energy);
            diagnostics.net_outward_energy = static_cast<double>(net_outward_energy);
            diagnostics.outward_current = static_cast<double>(outward_current);
            diagnostics.face_energy_density = static_cast<double>(face_density);
            diagnostics.pressure = static_cast<double>(face_density / 3.0L);
            if (!std::isfinite(leaked_energy) || !std::isfinite(net_outward_energy) ||
                !std::isfinite(outward_current) || !std::isfinite(face_density) ||
                !finite_nonnegative(diagnostics.pressure))
            {
                return reject("Non-finite boundary face reconstruction.");
            }

            long double const estimator =
                static_cast<long double>(cell.volume) *
                static_cast<long double>(face.opacity) * net_outward_energy /
                (2.0L * static_cast<long double>(face.area) *
                 static_cast<long double>(controls.speed_of_light));
            if (!addCellFace(face.cell, face.axis, face.outward_sign, face.area,
                             static_cast<double>(outward_current),
                             static_cast<double>(net_outward_energy),
                             static_cast<double>(face_density), false,
                             scaledNormalVector(face.axis, face.outward_sign,
                                                static_cast<double>(estimator))))
            {
                return reject("Non-finite boundary cell face diagnostic.");
            }
            candidate.total_leaked_energy += diagnostics.leaked_energy;
        }
        candidate.total_admitted_energy = geometry->admissions.empty()
                                               ? 0.0
                                               : [&geometry] () {
                                                     long double total = 0.0L;
                                                     for (double value : geometry->admissions)
                                                     {
                                                         total += static_cast<long double>(value);
                                                     }
                                                     return static_cast<double>(total);
                                                 }();

        for (std::size_t cell = 0; cell < number_cells; ++cell)
        {
            for (int axis = 0; axis < input.dimension; ++axis)
            {
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    int const slot = 2 * axis + (sign > 0 ? 1 : 0);
                    if (!present_faces[cell][slot])
                    {
                        double const reflecting_density = densities[cell];
                        Vector const zero_contribution{0.0, 0.0, 0.0};
                        if (!addCellFace(cell, axis, sign,
                                         cross_section(input.cells[cell], axis, input.dimension),
                                         0.0, 0.0, reflecting_density, true,
                                         zero_contribution))
                        {
                            return reject("Non-finite reflecting face reconstruction.");
                        }
                    }
                }
            }

            AsymptoticDiffusionCellDiagnostics& diagnostics = candidate.cells[cell];
            diagnostics.cell = cell;
            diagnostics.old_energy = input.cells[cell].energy;
            diagnostics.new_energy = energies[cell];
            diagnostics.volume = input.cells[cell].volume;
            diagnostics.new_energy_density = densities[cell];
            diagnostics.requested_work = geometry->requested_work[cell];
            diagnostics.faces = cell_faces[cell];

            long double energy_balance = static_cast<long double>(energies[cell]) -
                                         static_cast<long double>(input.cells[cell].energy) +
                                         static_cast<long double>(diagnostics.requested_work);
            long double energy_scale = std::abs(static_cast<long double>(energies[cell])) +
                                       std::abs(static_cast<long double>(input.cells[cell].energy)) +
                                       std::abs(static_cast<long double>(diagnostics.requested_work));

            Vector pressure_term{0.0, 0.0, 0.0};
            for (AsymptoticDiffusionCellFaceDiagnostics const& face : diagnostics.faces)
            {
                energy_balance += static_cast<long double>(face.outward_energy);
                energy_scale += std::abs(static_cast<long double>(face.outward_energy));
                Vector const normal =
                    scaledNormalVector(face.axis, face.outward_sign, 1.0);
                addScaled(diagnostics.radiation_inertia, normal,
                          input.cells[cell].volume * face.outward_current /
                              (2.0 * controls.speed_of_light));
                addScaled(diagnostics.estimator_impulse, face.estimator_contribution, 1.0);
                addScaled(pressure_term, normal,
                          controls.speed_of_light * controls.dt * face.area * face.pressure);
            }
            diagnostics.energy_balance_residual = static_cast<double>(energy_balance);
            if (!std::isfinite(diagnostics.energy_balance_residual) ||
                std::abs(energy_balance) > controls.solver_tolerance * energy_scale)
            {
                return reject("Cell scalar energy-plus-work ledger exceeded acceptance tolerance.");
            }
            for (int d = 0; d < 3; ++d)
            {
                diagnostics.closure_diagnostic[d] =
                    diagnostics.radiation_inertia[d] -
                    geometry->old_radiation_inertia[cell][d] + pressure_term[d] +
                    controls.speed_of_light * diagnostics.estimator_impulse[d];
            }
            if (!finiteVector(diagnostics.radiation_inertia) ||
                !finiteVector(diagnostics.estimator_impulse) ||
                !finiteVector(diagnostics.closure_diagnostic))
            {
                return reject("Non-finite cell momentum diagnostic.");
            }
        }

        if (!std::isfinite(candidate.total_admitted_energy) ||
            !std::isfinite(candidate.total_leaked_energy))
        {
            return reject("Non-finite admission or leakage ledger.");
        }

        long double total_work = 0.0L;
        long double global_balance = 0.0L;
        for (auto const& cell : candidate.cells)
        {
            total_work += static_cast<long double>(cell.requested_work);
            global_balance += static_cast<long double>(cell.energy_balance_residual);
        }
        candidate.total_requested_work = static_cast<double>(total_work);
        candidate.energy_balance_residual = static_cast<double>(global_balance);
        if (!std::isfinite(candidate.total_requested_work) ||
            !std::isfinite(candidate.energy_balance_residual))
        {
            return reject("Non-finite scalar work ledger.");
        }

        AsymptoticDiffusionResult result;
        result.accepted = true;
        result.candidate = std::move(candidate);
        return result;
    }
} // namespace warpx::radiation

/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_H_
#define WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_H_

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace warpx::radiation
{

    /** Cartesian vector used by the bounded, host-only diagnostic interface. */
    using AsymptoticDiffusionVector = std::array<double, 3>;

    /** One rectangular diffusion cell.
     *
     * `energy` is the integrated radiation energy [J] and `volume` is the cell
     * volume [m^dim].  For `dimension == 1`, `widths = {dx, 0, 0}`; for
     * `dimension == 2`, `widths = {dx, dy, 0}`; and for `dimension == 3` all
     * three entries are positive.  The same rule applies to the frozen, gray
     * side opacity [1/m].  The volume must equal the product of represented
     * widths.  Unrepresented entries must be identically zero.
     */
    struct AsymptoticDiffusionCell
    {
        double energy = 0.0;
        double volume = 0.0;
        std::array<double, 3> widths = {0.0, 0.0, 0.0};
        std::array<double, 3> opacity = {0.0, 0.0, 0.0};
    };

    /** One uniquely oriented interior D/D face.
     *
     * The left cell is the cell on the negative side of the face and the right
     * cell is on the positive side.  A physical face may be represented only
     * once (in either orientation); reversed duplicates are rejected.  Side
     * widths and side opacities are repeated here so that face ownership is
     * unambiguous.  They must agree with both owned cell records.  `area` is
     * the rectangular cross-section area [m^(dim-1)].
     */
    struct AsymptoticDiffusionInteriorFace
    {
        std::size_t left = 0;
        std::size_t right = 0;
        int axis = 0;
        double area = 0.0;
        double left_width = 0.0;
        double right_width = 0.0;
        double left_opacity = 0.0;
        double right_opacity = 0.0;
    };

    /** One D-side K/D boundary face.
     *
     * `outward_sign` is +1 or -1 along `axis` and points out of the owned D
     * cell. Width is the full D-side cell width used in its half-cell law;
     * width and transport opacity must agree with the owning cell record.
     * Every K/D face must have P(mu)<=1 for all mu in [0,1], including when no
     * packet reaches it this step. Missing faces are reflecting; they
     * are not represented by this record.
     */
    struct AsymptoticDiffusionBoundaryFace
    {
        std::size_t cell = 0;
        int axis = 0;
        int outward_sign = 1;
        double area = 0.0;
        double width = 0.0;
        double opacity = 0.0;
    };

    /** Complete private scalar region to advance on one rank.
     *
     * `boundary_admissions` contains packet-to-D energy [J] already admitted
     * through the corresponding boundary face.  It is either empty (all zero)
     * or has exactly one entry per boundary face.  The caller supplies the
     * complete private region; this helper performs no MPI communication or
     * native-owner integration.
     */
    struct AsymptoticDiffusionInput
    {
        int dimension = 3;
        std::vector<AsymptoticDiffusionCell> cells;
        std::vector<AsymptoticDiffusionInteriorFace> interior_faces;
        std::vector<AsymptoticDiffusionBoundaryFace> boundary_faces;
        std::vector<double> boundary_admissions;

        /** Prescribed D-region material-work debit [J/cell] for this private
         * iteration. Positive work removes scalar radiation energy; negative
         * work returns energy to it. Empty means zero. A nonempty vector must
         * contain one finite entry per cell. The native outer secant iteration
         * supplies authoritative staged RequestedWork; this helper does not
         * approximate it with velocity dot impulse or debit collar work. */
        std::vector<double> requested_work;

        /** Optional previous radiation-inertia diagnostic Q_A [J].  An empty
         * vector means zero, which is the cold-start value. */
        std::vector<AsymptoticDiffusionVector> old_radiation_inertia;
    };

    struct AsymptoticDiffusionControls
    {
        double dt = 0.0;
        double speed_of_light = 299792458.0;
        double marshak_extrapolation_lambda = 0.7104;

        /** Dense host solve bound.  This helper is not a production parallel
         * linear solver and intentionally rejects larger private regions. */
        std::size_t max_cells = 64;

        /** Relative row-residual bound for accepting the dense solve. */
        double solver_tolerance = 1.0e-12;

        /** Relative consistency bound for rectangular metric checks. */
        double geometry_tolerance = 1.0e-10;
    };

    struct AsymptoticDiffusionInteriorFaceDiagnostics
    {
        std::size_t face = 0;
        std::size_t left = 0;
        std::size_t right = 0;
        int axis = 0;
        double area = 0.0;
        double current_left_to_right = 0.0; ///< [W m^-2]
        double left_outward_energy = 0.0;   ///< [J]
        double right_outward_energy = 0.0;  ///< [J]
        double face_energy_density = 0.0;   ///< [J m^-3]
        double pressure = 0.0;              ///< isotropic scalar [Pa]
    };

    struct AsymptoticDiffusionBoundaryFaceDiagnostics
    {
        std::size_t face = 0;
        std::size_t cell = 0;
        int axis = 0;
        int outward_sign = 1;
        double area = 0.0;
        double admitted_energy = 0.0;   ///< [J]
        double leaked_energy = 0.0;     ///< [J]
        double net_outward_energy = 0.0; ///< leakage - admission [J]
        double outward_current = 0.0;    ///< [W m^-2]
        double face_energy_density = 0.0;
        double pressure = 0.0;
    };

    struct AsymptoticDiffusionCellFaceDiagnostics
    {
        int axis = 0;
        int outward_sign = 1;
        double area = 0.0;
        double outward_current = 0.0;
        double outward_energy = 0.0;
        double face_energy_density = 0.0;
        double pressure = 0.0;
        bool reflecting = false;
        AsymptoticDiffusionVector estimator_contribution = {0.0, 0.0, 0.0};
    };

    struct AsymptoticDiffusionCellDiagnostics
    {
        std::size_t cell = 0;
        double old_energy = 0.0;
        double new_energy = 0.0;
        double volume = 0.0;
        double new_energy_density = 0.0;
        double requested_work = 0.0; ///< supplied scalar reservoir debit [J]
        double energy_balance_residual = 0.0; ///< Enew-Eold+sum Qout+work [J]
        double energy_equation_residual = 0.0;

        /// Q_A = V/(2c) sum_f F_out n [J]; diagnostic state, not causal P1/M1.
        AsymptoticDiffusionVector radiation_inertia = {0.0, 0.0, 0.0};

        /// All-D Densmore estimator, including D/D and D/K owned faces.
        AsymptoticDiffusionVector estimator_impulse = {0.0, 0.0, 0.0};

        /// R = Q_A^new-Q_A^old+c dt sum A Pi n+c I_C [J].  This is an omitted
        /// radiation inertia/scalar closure diagnostic, not a momentum
        /// residual, momentum sink, or accepted material impulse.
        AsymptoticDiffusionVector closure_diagnostic = {0.0, 0.0, 0.0};

        std::vector<AsymptoticDiffusionCellFaceDiagnostics> faces;
    };

    struct AsymptoticDiffusionCandidate
    {
        std::vector<double> energies;
        std::vector<double> energy_densities;
        std::vector<AsymptoticDiffusionInteriorFaceDiagnostics> interior_faces;
        std::vector<AsymptoticDiffusionBoundaryFaceDiagnostics> boundary_faces;
        std::vector<AsymptoticDiffusionCellDiagnostics> cells;
        std::vector<double> requested_work; ///< accepted supplied debit [J/cell]
        double total_admitted_energy = 0.0;
        double total_leaked_energy = 0.0;
        double total_requested_work = 0.0;
        double energy_balance_residual = 0.0; ///< global scalar ledger [J]
    };

    struct AsymptoticDiffusionResult
    {
        bool accepted = false;
        std::string rejection_reason;
        std::optional<AsymptoticDiffusionCandidate> candidate;

        explicit operator bool () const noexcept { return accepted; }
    };

    /** Construct and solve one private backward-Euler scalar diffusion step.
     *
     * The unknown is the nonnegative radiation energy density.  Interior and
     * leakage coefficients are evaluated at the new time. The prescribed
     * requested_work is a signed scalar energy debit for the caller's private
     * material-work iteration. The assembled matrix
     * is a conservative M-matrix.  Energies and flux cones are never clipped.
     * All inputs are const.  A rejected return contains no accepted candidate,
     * leakage, impulse, or packet split and leaves caller-owned state unchanged.
     */
    AsymptoticDiffusionResult
    AdvanceAsymptoticDiffusion (AsymptoticDiffusionInput const& input,
                                AsymptoticDiffusionControls const& controls);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_ASYMPTOTIC_DIFFUSION_H_

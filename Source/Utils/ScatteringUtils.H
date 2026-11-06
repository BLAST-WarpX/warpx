/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SCATTERING_UTILS_H_
#define WARPX_SCATTERING_UTILS_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace ScatteringUtils {

    /**
     * \brief View of an energy-dependent table of Legendre coefficients.
     *
     * Each entry in m_energies corresponds to one center-of-mass energy.
     * Within each energy group in m_coefficients, index \f$l\f$ stores \f$L_l\f$, the
     * coefficient of the Legendre polynomial \f$P_l\f$.
     * The view does not own either array, so their storage must remain valid wherever this
     * object is used, including on a GPU.
     */
    struct AnisotropicCoefficientTable
    {
        amrex::ParticleReal const* m_energies = nullptr;  ///< Increasing energy grid [eV].
        amrex::ParticleReal const* m_coefficients = nullptr;  ///< Coefficients grouped by energy.
        int m_num_energies = 0;  ///< Number of energy rows.
        int m_num_coefficients = 0;  ///< Number of Legendre orders stored in each row.

        /**
         * \brief Return a linearly interpolated coefficient for the given energy.
         *
         * For an energy between two rows, this interpolates between the values selected by
         * \p a_coefficient_index.
         * An energy outside the table uses the nearest endpoint row; the caller is
         * responsible for warning that this clamping occurred.
         *
         * @param[in] a_energy centre-of-mass energy in eV
         * @param[in] a_coefficient_index column index into the coefficient table
         * @return interpolated (or clamped) coefficient value
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        amrex::ParticleReal getCoefficient (
            amrex::ParticleReal const a_energy,
            int const a_coefficient_index) const
        {
            using namespace amrex::literals;

            // m_coefficients stores one consecutive block for each entry in m_energies.
            // The block for m_energies[0] starts at index zero.
            // The block for m_energies[m_num_energies - 1] starts at
            // (m_num_energies - 1) * m_num_coefficients.
            // a_coefficient_index selects a coefficient within either block.
            // Do not extrapolate the fitted coefficients beyond the tabulated energies.
            if (a_energy <= m_energies[0]) {
                return m_coefficients[a_coefficient_index];
            }
            if (a_energy >= m_energies[m_num_energies - 1]) {
                return m_coefficients[(m_num_energies - 1) * m_num_coefficients + a_coefficient_index];
            }

            // Find lo and hi such that m_energies[lo] < a_energy <= m_energies[hi].
            // TODO: replace this linear scan with a binary search for large tables.
            int hi = 1;
            while ((hi < m_num_energies - 1) && (a_energy > m_energies[hi])) {
                ++hi;
            }
            int const lo = hi - 1;

            amrex::ParticleReal const energy_lo = m_energies[lo];
            amrex::ParticleReal const energy_hi = m_energies[hi];
            // lo and hi select the two coefficient blocks in m_coefficients.
            amrex::ParticleReal const coeff_lo = m_coefficients[lo * m_num_coefficients + a_coefficient_index];
            amrex::ParticleReal const coeff_hi = m_coefficients[hi * m_num_coefficients + a_coefficient_index];

            // interp is zero at the lower row and one at the upper row.
            amrex::ParticleReal const interp = (a_energy - energy_lo) / (energy_hi - energy_lo);

            return coeff_lo + (coeff_hi - coeff_lo) * interp;
        }
    };

    /**
     * \brief Sample the cosine of the fusion scattering angle from a coefficient
     * table, linearly interpolating the coefficients in energy.
     *
     * The probability density is expressed in terms of Legendre coefficients \f$L_l\f$:
     * \f$f(\mu) = 1/2 + \sum_{l=1}^{l_{\mathrm{max}}} (2l+1)L_lP_l(\mu)/2\f$,
     * where \f$\mu = \cos(\theta)\f$.
     * The cumulative distribution \f$F(\mu)\f$ is the integral of \f$f\f$ from -1 to
     * \f$\mu\f$, or equivalently the probability of sampling a cosine no greater than
     * \f$\mu\f$.
     * This function uses inverse-transform sampling: it finds the value of \f$\mu\f$ for
     * which the cumulative distribution \f$F(\mu)\f$ equals \p a_random_number.
     *
     * @param[in] a_coefficient_table table of anisotropic Legendre coefficients
     * @param[in] a_energy center-of-mass kinetic energy in eV
     * @param[in] a_random_number uniform random number in [0,1]
     * @param[in,out] a_energy_range_status optional per-event status.
     *     The caller initializes it to zero; this function changes it to -1 below the table
     *     or 1 above the table.
     * @return sampled value of \f$\cos(\theta)\f$ in [-1, 1]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal sampleCosTheta (
        AnisotropicCoefficientTable const& a_coefficient_table,
        amrex::ParticleReal const a_energy,
        amrex::ParticleReal const a_random_number,
        int* const a_energy_range_status)
    {
        using namespace amrex::literals;

        // TODO: In single precision the tolerance might need to be adjusted.
        constexpr amrex::ParticleReal tol = 1.0e-8_prt;
        constexpr int iter_max = 100;
        // Index zero stores L_0, so the number of coefficients minus one is the highest order.
        int const l_max = a_coefficient_table.m_num_coefficients - 1;

        // Record whether getCoefficient will clamp this event's energy.
        // Do this once here, rather than once for every coefficient during every iteration below.
        if (a_energy_range_status != nullptr) {
            if (a_energy < a_coefficient_table.m_energies[0]) {
                *a_energy_range_status = -1;
            } else if (a_energy > a_coefficient_table.m_energies[a_coefficient_table.m_num_energies - 1])
            {
                *a_energy_range_status = 1;
            }
        }

        // Start Newton's method at the midpoint of the physically allowed interval.
        amrex::ParticleReal mu = 0.0_prt;

        for (int iter = 0; iter < iter_max; ++iter) {
            // These are the isotropic (all L_l above order zero are zero) parts of the
            // probability density f(mu) and cumulative distribution F(mu).
            amrex::ParticleReal fmu = 0.5_prt;
            amrex::ParticleReal Fmu = 0.5_prt * (mu + 1.0_prt);

            // Build consecutive Legendre polynomials from P_0 = 1 and P_1 = mu.
            // Keeping only three orders avoids a temporary array in this GPU device routine.
            amrex::ParticleReal P_l_minus_one = 1.0_prt;
            amrex::ParticleReal P_l = mu;
            for (int l = 1; l <= l_max; ++l) {
                auto const l_prt = static_cast<amrex::ParticleReal>(l);
                // Use Bonnet's recurrence for the next Legendre polynomial:
                // (l + 1) P_{l+1}(mu) = (2l + 1) mu P_l(mu) - l P_{l-1}(mu).
                amrex::ParticleReal const P_l_plus_one = ((2.0_prt * l_prt + 1.0_prt) * mu * P_l - l_prt * P_l_minus_one) / (l_prt + 1.0_prt);
                amrex::ParticleReal const coefficient = a_coefficient_table.getCoefficient(a_energy, l);

                // Accumulate the PDF and its integral.
                // The identity integral(P_l dmu) = (P_{l+1} - P_{l-1}) / (2l + 1) gives the CDF term.
                fmu += (l_prt + 0.5_prt) * coefficient * P_l;
                Fmu += 0.5_prt * coefficient * (P_l_plus_one - P_l_minus_one);

                // Advance the recurrence by one Legendre order.
                P_l_minus_one = P_l;
                P_l = P_l_plus_one;
            }

            // Inverse-transform sampling requires solving F(mu) - a_random_number = 0.
            amrex::ParticleReal const residual = Fmu - a_random_number;
            if (std::abs(residual) < tol) {
                return mu;
            }

            // Since f(mu) is the derivative of F(mu) with respect to mu, this is one Newton step.
            // Clamp the result to the valid range for cos(theta) if the Newton step moves it outside [-1, 1].
            amrex::ParticleReal const dmu = residual / fmu;
            mu -= dmu;
            mu = amrex::max(-1.0_prt, amrex::min(1.0_prt, mu));

            // A negligible change in mu also indicates convergence when roundoff
            // prevents the residual itself from dropping below the tolerance.
            if (std::abs(dmu) < tol) {
                return mu;
            }
        }

        // Return the best bounded estimate if Newton's method reaches its iteration limit.
        return mu;
    }
}

#endif // WARPX_SCATTERING_UTILS_H_

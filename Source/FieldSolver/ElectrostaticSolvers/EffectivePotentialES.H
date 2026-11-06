/* Copyright 2024-2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EFFECTIVEPOTENTIALES_H_
#define WARPX_EFFECTIVEPOTENTIALES_H_

#include "ElectrostaticSolver.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/EffectivePotentialPoissonSolver.H>

class EffectivePotentialES final : public ElectrostaticSolver
{
public:

    explicit EffectivePotentialES (int nlevs_max) : ElectrostaticSolver (nlevs_max) {
        ReadParameters();
    }

    void InitData () override;

    void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level,
        bool verbose_step) override;

    /**
     * Deposit the total charge density and compute the dressing `sigma` of
     * the Poisson equation on all mesh refinement levels.
     * \param[out] rho_fp The total charge density (fine patch) on each level
     * \param[in] rho_cp The coarse patch of the charge density on each level
     *                    (used as work space for the synchronization across levels)
     */
    void ComputeSigma (
        ablastr::fields::MultiLevelScalarField const& rho_fp,
        ablastr::fields::MultiLevelScalarField const& rho_cp
    );

    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        ablastr::fields::MultiLevelVectorField const& efield,
        int verbosity
    );

    /**
     * Compute the potential `phi` by solving the semi-implicit Poisson equation using the Effective Potential method
     * with `rho` as the source.
     * More specifically, this solves the equation
     *  \f[
     *      \boldsymbol{\nabla}\cdot(\sigma\boldsymbol{\nabla}) \phi = -\frac{\rho}{\epsilon_0}
     *  \f]
     * \param[in] rho The total charge density
     * \param[out] phi The potential to be computed by this function
     * \param[out] efield The electric field corresponding to the calculated phi (only used with embedded boundaries)
     * \param[in] mlmg_options Settings for the MLMG solve, @see ablastr::fields::MLMGOptions
     */
    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        ablastr::fields::MultiLevelVectorField const& efield,
        ablastr::fields::MLMGOptions const & mlmg_options
    ) const;

    bool m_overwrite_sigma;
};

#endif // WARPX_EFFECTIVEPOTENTIALES_H_

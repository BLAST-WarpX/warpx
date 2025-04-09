/* Copyright 2024 The WarpX Community
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

    EffectivePotentialES (int nlevs_max) : ElectrostaticSolver (nlevs_max) {
        ReadParameters();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (nlevs_max == 1),
            "Effective potential electrostatic solver only supports one level at present"
        );
    }

    void InitData () override;

    void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level) override;

    void ComputeSigma ( amrex::MultiFab& sigma ) const;

    /**
     * Compute the potential `phi` by solving the semi-implicit Poisson equation using the Effective Potential method
     * with `rho` as the source.
     * More specifically, this solves the equation
     *  \f[
     *      \vec{\nabla}\cdot(\sigma\vec{\nabla}) \phi = -\frac{\rho}{\epsilon_0}
     *  \f]
     * \param[out] phi The potential to be computed by this function
     * \param[in] rho The total charge density
     * \param[in] sigma Represents the modified dielectric
     * \param[in] required_precision The relative convergence threshold for the MLMG solver
     * \param[in] absolute_tolerance The absolute convergence threshold for the MLMG solver
     * \param[in] max_iters The maximum number of iterations allowed for the MLMG solver
     * \param[in] verbosity The verbosity setting for the MLMG solver
     */
    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi
    ) const;
    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        amrex::MultiFab const& sigma,
        amrex::Real required_precision,
        amrex::Real absolute_tolerance,
        int max_iters,
        int verbosity
    ) const;

};

#endif // WARPX_EFFECTIVEPOTENTIALES_H_

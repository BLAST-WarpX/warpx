/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTROSTATICSOLVER_H_
#define WARPX_ELECTROSTATICSOLVER_H_

#include "PoissonBoundaryHandler.H"
#include "Fluids/MultiFluidContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_Array.H>


/**
 * \brief Base class for Electrostatic Solver
 *
 */
class ElectrostaticSolver
{
public:
    ElectrostaticSolver() = default;
    ElectrostaticSolver( int nlevs_max );

    virtual ~ElectrostaticSolver();

    // Prohibit Move and Copy operations
    ElectrostaticSolver(const ElectrostaticSolver&) = delete;
    ElectrostaticSolver& operator=(const ElectrostaticSolver&) = delete;
    ElectrostaticSolver(ElectrostaticSolver&&) = delete;
    ElectrostaticSolver& operator=(ElectrostaticSolver&&) = delete;

    void ReadParameters ();

    virtual void InitData () {}

    /**
     * \brief Computes charge density, rho, and solves Poisson's equation
     *        to obtain the associated electrostatic potential, phi.
     *        Using the electrostatic potential, the electric field is computed
     *        in lab frame, and if relativistic, then the electric and magnetic
     *        fields are computed using potential, phi, and
     *        velocity of source for potential, beta.
     *        This function must be defined in the derived classes.
     */
    virtual void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level) = 0;

    /**
     * \brief Set Dirichlet boundary conditions for the electrostatic solver.
     * The given potential's values are fixed on the boundaries of the given
     * dimension according to the desired values from the simulation input file,
     * boundary.potential_lo and boundary.potential_hi.
     * \param[inout] phi The electrostatic potential
     * \param[in] idim The dimension for which the Dirichlet boundary condition is set
     */
    void setPhiBC (
        ablastr::fields::MultiLevelScalarField const& phi,
        amrex::Real t
    ) const;

    /**
     * Compute the potential `phi` by solving the Poisson equation with `rho` as
     * a source, assuming that the source moves at a constant speed \f$\vec{\beta}\f$.
     * This uses the amrex solver.
     * More specifically, this solves the equation
     *  \f[
     *      \vec{\nabla}^2 r \phi - (\vec{\beta}\cdot\vec{\nabla})^2 r \phi = -\frac{r \rho}{\epsilon_0}
     *  \f]
     * \param[out] phi The potential to be computed by this function
     * \param[in] rho The charge density for a given species (relativistic solver)
     *                               or total charge density (labframe solver)
     * \param[in] beta Represents the velocity of the source of `phi`
     * \param[in] required_precision The relative convergence threshold for the MLMG solver
     * \param[in] absolute_tolerance The absolute convergence threshold for the MLMG solver
     * \param[in] max_iters The maximum number of iterations allowed for the MLMG solver
     * \param[in] verbosity The verbosity setting for the MLMG solver
     */
    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        std::array<amrex::Real, 3> beta,
        amrex::Real required_precision,
        amrex::Real absolute_tolerance,
        int max_iters,
        int verbosity,
        bool is_igf_2d_slices
    ) const;

    /**
     * \brief Compute the electric field that corresponds to `phi`, and
     *        add it to the set of MultiFab `E`.
     * The electric field is calculated by assuming that the source that
     * produces the `phi` potential is moving with a constant speed \f$\vec{\beta}\f$:
     * \f[
     *     \vec{E} = -\vec{\nabla}\phi + \vec{\beta}(\vec{\beta} \cdot \vec{\nabla}\phi)
     * \f]
     * (where the second term represent the term \f$\partial_t \vec{A}\f$, in
     *     the case of a moving source)
     *
     * \param[inout] E Electric field on the grid
     * \param[in] phi The potential from which to compute the electric field
     * \param[in] beta Represents the velocity of the source of `phi`
     */
    void computeE (
        ablastr::fields::MultiLevelVectorField const& E,
        ablastr::fields::MultiLevelScalarField const& phi,
        std::array<amrex::Real, 3> beta
    ) const;

    /**
     * \brief Compute the magnetic field that corresponds to `phi`, and
     *        add it to the set of MultiFab `B`.
     *The magnetic field is calculated by assuming that the source that
     *produces the `phi` potential is moving with a constant speed \f$\vec{\beta}\f$:
     *\f[
     *    \vec{B} = -\frac{1}{c}\vec{\beta}\times\vec{\nabla}\phi
     *\f]
     *(this represents the term \f$\vec{\nabla} \times \vec{A}\f$, in the case of a moving source)
     *
     *\param[inout] B Electric field on the grid
     *\param[in] phi The potential from which to compute the electric field
     *\param[in] beta Represents the velocity of the source of `phi`
     */
    void computeB (
        ablastr::fields::MultiLevelVectorField const& B,
        ablastr::fields::MultiLevelScalarField const& phi,
        std::array<amrex::Real, 3> beta
    ) const;

    /** Maximum levels for the electrostatic solver grid */
    int num_levels;

    /** Boundary handler object to set potential for EB and on the domain boundary */
    std::unique_ptr<PoissonBoundaryHandler> m_poisson_boundary_handler;

    /** Parameters for MLMG Poisson solve */
    amrex::Real self_fields_required_precision = 1e-11;
    amrex::Real self_fields_absolute_tolerance = 0.0;
    /** Limit on number of MLMG iterations */
    int self_fields_max_iters = 200;
    /** Verbosity for the MLMG solver.
     *  0 : no verbosity
     *  1 : timing and convergence at the end of MLMG
     *  2 : convergence progress at every MLMG iteration
     */
    int self_fields_verbosity = 2;

    /** Parameters for FFT Poisson solver aka IGF */
    // 0: full 3D, 1: many 2D z-slices (quasi-3D)
    bool is_igf_2d_slices = false;
};

#endif // WARPX_ELECTROSTATICSOLVER_H_

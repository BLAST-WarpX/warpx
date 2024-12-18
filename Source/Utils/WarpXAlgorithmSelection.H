/* Copyright 2019 David Grote, Luca Fedeli, Remi Lehe
 * Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_WARPXALGORITHMSELECTION_H_
#define WARPX_UTILS_WARPXALGORITHMSELECTION_H_

#include <AMReX_BaseFwd.H>
#include <AMReX_Enum.H>
#include <ablastr/utils/Enums.H>

#include <string>

using namespace ablastr::utils::enums;  // NOLINT(google-global-names-in-headers)


/**
  * \brief struct to determine the computational medium, i.e., vacuum or material/macroscopic
           default is vacuum.
  */
AMREX_ENUM(MediumForEM,
           Vacuum,
           Macroscopic,
           Default = Vacuum);

/**
  * \brief struct to select the overall evolve scheme
  */
AMREX_ENUM(EvolveScheme,
           Explicit,
           ThetaImplicitEM,
           SemiImplicitEM,
           StrangImplicitSpectralEM,
           Default = Explicit);

/**
  * \brief struct to select algorithm for macroscopic Maxwell solver
           LaxWendroff (semi-implicit) represents sigma*E = sigma*0.5*(E^(n) + E^(n+1))
           Backward Euler (fully-implicit) represents sigma*E = sigma*E^(n+1)
           default is Backward Euler as it is more robust.
  */
AMREX_ENUM(MacroscopicSolverAlgo,
           BackwardEuler,
           LaxWendroff,
           Default = BackwardEuler);

AMREX_ENUM(ElectromagneticSolverAlgo,
           None,
           Yee,
           CKC,
           PSATD,
           ECT,
           HybridPIC,
           hybrid = HybridPIC,
           Default = Yee);

AMREX_ENUM(ElectrostaticSolverAlgo,
           None,
           Relativistic,
           LabFrameElectroMagnetostatic,
           LabFrame,
           LabFrameEffectivePotential,
           Default = None);

AMREX_ENUM(PoissonSolverAlgo,
           Multigrid,
           IntegratedGreenFunction,
           fft = IntegratedGreenFunction,
           Default = Multigrid);

AMREX_ENUM(ParticlePusherAlgo,
           Boris,
           Vay,
           HigueraCary,
           higuera = HigueraCary,
           Default = Boris);

AMREX_ENUM(CurrentDepositionAlgo,
           Esirkepov,
           Direct,
           Vay,
           Villasenor,
           Default = Esirkepov);

AMREX_ENUM(ChargeDepositionAlgo,
           Standard,
           Default = Standard);

AMREX_ENUM(GatheringAlgo,
           EnergyConserving,
           MomentumConserving,
           Default = EnergyConserving);

AMREX_ENUM(PSATDSolutionType,
           FirstOrder,
           SecondOrder,
           Default = SecondOrder);

AMREX_ENUM(JInTime,
           Constant,
           Linear,
           Default = Constant);

AMREX_ENUM(RhoInTime,
           Constant,
           Linear,
           Default = Linear);

/** Strategy to compute weights for use in load balance.
 */
AMREX_ENUM(LoadBalanceCostsUpdateAlgo,
           Timers,     //!< load balance according to in-code timer-based weights (i.e., with  `costs`)
           Heuristic,  /**< load balance according to weights computed from number of cells
                          and number of particles per box (i.e., with `costs_heuristic`) */
           Default = Timers);

/** Field boundary conditions at the domain boundary
 */
AMREX_ENUM(FieldBoundaryType,
           PML,
           Periodic,
           PEC,      //!< perfect electric conductor (PEC) with E_tangential=0
           PMC,      //!< perfect magnetic conductor (PMC) with B_tangential=0
           Damped,   // Fields in the guard cells are damped for PSATD
                     //in the moving window direction
           Absorbing_SilverMueller, // Silver-Mueller boundary condition
           absorbingsilvermueller = Absorbing_SilverMueller,
           Neumann, // For electrostatic, the normal E is set to zero
           None,    // The fields values at the boundary are not updated. This is
                    // useful for RZ simulations, at r=0.
           Open,    // Used in the Integrated Green Function Poisson solver
                    // Note that the solver implicitely assumes open BCs:
                    // no need to enforce them separately
           PECInsulator, // Mixed boundary with PEC and insulator
           Default = PML);

/** Particle boundary conditions at the domain boundary
 */
AMREX_ENUM(ParticleBoundaryType,
           Absorbing,     //!< particles crossing domain boundary are removed
           Open,          //!< particles cross domain boundary leave with damped j
           Reflecting,    //!< particles are reflected
           Periodic,      //!< particles are introduced from the periodic boundary
           Thermal,
           None,          //!< For r=0 boundary with RZ simulations
           Default = Absorbing);

/** MPI reductions
 */
AMREX_ENUM(ReductionType,
           Maximum,
           Minimum,
           Sum,
           Integral = Sum);

#endif // WARPX_UTILS_WARPXALGORITHMSELECTION_H_

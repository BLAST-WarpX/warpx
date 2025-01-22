/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_SOLVER_H_
#define WARPX_FINITE_DIFFERENCE_SOLVER_H_

#include "EmbeddedBoundary/WarpXFaceInfoBox_fwd.H"
#include "FiniteDifferenceSolver_fwd.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "BoundaryConditions/PML_fwd.H"
#include "Evolve/WarpXDtType.H"
#include "HybridPICModel/HybridPICModel_fwd.H"
#include "MacroscopicProperties/MacroscopicProperties_fwd.H"

#include <ablastr/utils/Enums.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

/**
 * \brief Top-level class for the electromagnetic finite-difference solver
 *
 * Stores the coefficients of the finite-difference stencils,
 * and has member functions to update fields over one time step.
 */
class FiniteDifferenceSolver
{
    public:

        // Constructor
        /** \brief Initialize the finite-difference Maxwell solver (for a given refinement level)
         *
         * This function initializes the stencil coefficients for the chosen finite-difference algorithm
         *
         * \param fdtd_algo Identifies the chosen algorithm, as defined in WarpXAlgorithmSelection.H
         * \param cell_size Cell size along each dimension, for the chosen refinement level
         * \param grid_type Whether the solver is applied to a collocated or staggered grid
         */
        FiniteDifferenceSolver (
            ElectromagneticSolverAlgo fdtd_algo,
            std::array<amrex::Real,3> cell_size,
            ablastr::utils::enums::GridType grid_type );

        void EvolveB ( ablastr::fields::MultiFabRegister& fields,
                       int lev,
                       PatchType patch_type,
                       std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& flag_info_cell,
                       std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 >& borrowing,
                       amrex::Real dt );

        void EvolveE ( ablastr::fields::MultiFabRegister & fields,
                       int lev,
                       PatchType patch_type,
                       ablastr::fields::VectorField const& Efield,
                       std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
                       amrex::Real dt );

        void EvolveF ( amrex::MultiFab* Ffield,
                       ablastr::fields::VectorField const& Efield,
                       amrex::MultiFab* rhofield,
                       int rhocomp,
                       amrex::Real dt );

        void EvolveG (amrex::MultiFab* Gfield,
                      ablastr::fields::VectorField const& Bfield,
                      amrex::Real dt);

        void EvolveECTRho ( ablastr::fields::VectorField const& Efield,
                            ablastr::fields::VectorField const& edge_lengths,
                            ablastr::fields::VectorField const& face_areas,
                            ablastr::fields::VectorField const& ECTRhofield,
                            int lev );

        void ApplySilverMuellerBoundary (
            ablastr::fields::VectorField & Efield,
            ablastr::fields::VectorField & Bfield,
            amrex::Box domain_box,
            amrex::Real dt,
            amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> field_boundary_lo,
            amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> field_boundary_hi);

        void ComputeDivE (
            ablastr::fields::VectorField const & Efield,
            amrex::MultiFab& divE
        );

        /**
          * \brief Macroscopic E-update for non-vacuum medium using the user-selected
          * finite-difference algorithm and macroscopic sigma-method defined in
          * WarpXAlgorithmSelection.H
          *
          * \param[out] Efield  vector of electric field MultiFabs updated at a given level
          * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
          * \param[in] Jfield   vector of current density MultiFabs at a given level
          * \param[in] eb_update_E indicate in which cell E should be updated (related to embedded boundaries)
          * \param[in] dt       timestep of the simulation
          * \param[in] macroscopic_properties contains user-defined properties of the medium.
          */
        void MacroscopicEvolveE (
                      ablastr::fields::VectorField const& Efield,
                      ablastr::fields::VectorField const& Bfield,
                      ablastr::fields::VectorField const& Jfield,
                      std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
                      amrex::Real dt,
                      std::unique_ptr<MacroscopicProperties> const& macroscopic_properties);

        void EvolveBPML (
            ablastr::fields::MultiFabRegister& fields,
            PatchType patch_type,
            int level,
            amrex::Real dt,
            bool dive_cleaning
        );

       void EvolveEPML (
            ablastr::fields::MultiFabRegister& fields,
            PatchType patch_type,
            int level,
            MultiSigmaBox const& sigba,
            amrex::Real dt,
            bool pml_has_particles
        );

       void EvolveFPML ( amrex::MultiFab* Ffield,
                         ablastr::fields::VectorField Efield,
                         amrex::Real dt );

        /**
          * \brief E-update in the hybrid PIC algorithm as described in
          * Winske et al. (2003) Eq. 10.
          * https://link.springer.com/chapter/10.1007/3-540-36530-3_8
          *
          * \param[out] Efield  vector of electric field MultiFabs updated at a given level
          * \param[in] Jfield   vector of total plasma current MultiFabs at a given level
          * \param[in] Jifield  vector of ion current density MultiFabs at a given level
          * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
          * \param[in] rhofield scalar ion charge density Multifab at a given level
          * \param[in] Pefield  scalar electron pressure MultiFab at a given level
          * \param[in] eb_update_E indicate in which cell E should be updated (related to embedded boundaries)
          * \param[in] lev  level number for the calculation
          * \param[in] hybrid_model instance of the hybrid-PIC model
          * \param[in] solve_for_Faraday boolean flag for whether the E-field is solved to be used in Faraday's equation
          */
        void HybridPICSolveE ( ablastr::fields::VectorField const& Efield,
                               ablastr::fields::VectorField & Jfield,
                               ablastr::fields::VectorField const& Jifield,
                               ablastr::fields::VectorField const& Bfield,
                               amrex::MultiFab const& rhofield,
                               amrex::MultiFab const& Pefield,
                               std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
                               int lev, HybridPICModel const* hybrid_model,
                               bool solve_for_Faraday );

        /**
          * \brief Calculation of total current using Ampere's law (without
          * displacement current): J = (curl x B) / mu0.
          *
          * \param[out] Jfield  vector of current MultiFabs at a given level
          * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
          * \param[in] eb_update_E indicate in which cell E should be updated (related to embedded boundaries)
          * \param[in] lev  level number for the calculation
          */
        void CalculateCurrentAmpere (
                      ablastr::fields::VectorField& Jfield,
                      ablastr::fields::VectorField const& Bfield,
                      std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
                      int lev );

    private:

        ElectromagneticSolverAlgo m_fdtd_algo;
        ablastr::utils::enums::GridType m_grid_type;

#ifdef WARPX_DIM_RZ
        amrex::Real m_dr, m_rmin;
        int m_nmodes;
        // host-only
        amrex::Vector<amrex::Real> m_h_stencil_coefs_r, m_h_stencil_coefs_z;
        // device copy after init
        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_r;
        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;
#else
        // host-only
        amrex::Vector<amrex::Real> m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z;
        // device copy after init
        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_x;
        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_y;
        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;
#endif

    public:
        // The member functions below contain extended __device__ lambda.
        // In order to compile with nvcc, they need to be public.

#ifdef WARPX_DIM_RZ
        template< typename T_Algo >
        void EvolveBCylindrical (
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& Efield,
            int lev,
            amrex::Real dt );

        template< typename T_Algo >
        void EvolveECylindrical (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& Jfield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
            amrex::MultiFab const* Ffield,
            int lev,
            amrex::Real dt );

        template< typename T_Algo >
        void EvolveFCylindrical (
            amrex::MultiFab* Ffield,
            ablastr::fields::VectorField const & Efield,
            amrex::MultiFab* rhofield,
            int rhocomp,
            amrex::Real dt );

        template< typename T_Algo >
        void ComputeDivECylindrical (
            ablastr::fields::VectorField const & Efield,
            amrex::MultiFab& divE
        );

        template<typename T_Algo>
        void HybridPICSolveECylindrical (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& Jfield,
            ablastr::fields::VectorField const& Jifield,
            ablastr::fields::VectorField const& Bfield,
            amrex::MultiFab const& rhofield,
            amrex::MultiFab const& Pefield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
            int lev, HybridPICModel const* hybrid_model,
            bool solve_for_Faraday );

        template<typename T_Algo>
        void CalculateCurrentAmpereCylindrical (
            ablastr::fields::VectorField& Jfield,
            ablastr::fields::VectorField const& Bfield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
            int lev
        );

#else
        template< typename T_Algo >
        void EvolveBCartesian (
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& Efield,
            amrex::MultiFab const * Gfield,
            int lev, amrex::Real dt );

        template< typename T_Algo >
        void EvolveECartesian (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& Jfield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
            amrex::MultiFab const* Ffield,
            int lev, amrex::Real dt );

        template< typename T_Algo >
        void EvolveFCartesian (
            amrex::MultiFab* Ffield,
            ablastr::fields::VectorField Efield,
            amrex::MultiFab* rhofield,
            int rhocomp,
            amrex::Real dt );

        template< typename T_Algo >
        void EvolveGCartesian (
            amrex::MultiFab* Gfield,
            ablastr::fields::VectorField const& Bfield,
            amrex::Real dt);

        void EvolveRhoCartesianECT (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& edge_lengths,
            ablastr::fields::VectorField const& face_areas,
            ablastr::fields::VectorField const& ECTRhofield, int lev);

        void EvolveBCartesianECT (
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& face_areas,
            ablastr::fields::VectorField const& area_mod,
            ablastr::fields::VectorField const& ECTRhofield,
            ablastr::fields::VectorField const& Venl,
            std::array< std::unique_ptr<amrex::iMultiFab>, 3 >& flag_info_cell,
            std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 >& borrowing,
            int lev, amrex::Real dt
        );

        template< typename T_Algo >
        void ComputeDivECartesian (
            ablastr::fields::VectorField const & Efield,
            amrex::MultiFab& divE );

        template< typename T_Algo, typename T_MacroAlgo >
        void MacroscopicEvolveECartesian (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& Bfield,
            ablastr::fields::VectorField const& Jfield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
            amrex::Real dt,
            std::unique_ptr<MacroscopicProperties> const& macroscopic_properties);

        template< typename T_Algo >
        void EvolveBPMLCartesian (
            std::array< amrex::MultiFab*, 3 > Bfield,
            ablastr::fields::VectorField Efield,
            amrex::Real dt,
            bool dive_cleaning);

        template< typename T_Algo >
        void EvolveEPMLCartesian (
            ablastr::fields::VectorField Efield,
            std::array< amrex::MultiFab*, 3 > Bfield,
            std::array< amrex::MultiFab*, 3 > Jfield,
            std::array< amrex::MultiFab*, 3 > edge_lengths,
            amrex::MultiFab* Ffield,
            MultiSigmaBox const& sigba,
            amrex::Real dt, bool pml_has_particles );

        template< typename T_Algo >
        void EvolveFPMLCartesian ( amrex::MultiFab* Ffield,
                                   ablastr::fields::VectorField Efield,
                                   amrex::Real dt );

        template<typename T_Algo>
        void HybridPICSolveECartesian (
            ablastr::fields::VectorField const& Efield,
            ablastr::fields::VectorField const& Jfield,
            ablastr::fields::VectorField const& Jifield,
            ablastr::fields::VectorField const& Bfield,
            amrex::MultiFab const& rhofield,
            amrex::MultiFab const& Pefield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
            int lev, HybridPICModel const* hybrid_model,
            bool solve_for_Faraday );

        template<typename T_Algo>
        void CalculateCurrentAmpereCartesian (
            ablastr::fields::VectorField& Jfield,
            ablastr::fields::VectorField const& Bfield,
            std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_E,
            int lev
        );
#endif

};

#endif // WARPX_FINITE_DIFFERENCE_SOLVER_H_

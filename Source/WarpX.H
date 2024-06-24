/* Copyright 2016-2020 Andrew Myers, Ann Almgren, Aurore Blelly
 *                     Axel Huebl, Burlen Loring, David Grote
 *                     Glenn Richardson, Junmin Gu, Luca Fedeli
 *                     Mathieu Lobet, Maxence Thevenet, Michael Rowan
 *                     Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 *                     Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_H_
#define WARPX_H_

#include "BoundaryConditions/PML_fwd.H"
#include "Diagnostics/MultiDiagnostics_fwd.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags_fwd.H"
#include "EmbeddedBoundary/WarpXFaceInfoBox_fwd.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver_fwd.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties_fwd.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel_fwd.H"
#include "Filter/NCIGodfreyFilter_fwd.H"
#include "Initialization/ExternalField_fwd.H"
#include "Particles/ParticleBoundaryBuffer_fwd.H"
#include "Particles/MultiParticleContainer_fwd.H"
#include "Particles/WarpXParticleContainer_fwd.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "Fluids/WarpXFluidContainer_fwd.H"

#ifdef WARPX_USE_FFT
#   ifdef WARPX_DIM_RZ
#       include "FieldSolver/SpectralSolver/SpectralSolverRZ_fwd.H"
#       include "BoundaryConditions/PML_RZ_fwd.H"
#   else
#       include "FieldSolver/SpectralSolver/SpectralSolver_fwd.H"
#   endif
#endif
#include "AcceleratorLattice/AcceleratorLattice.H"
#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "FieldSolver/Fields.H"
#include "FieldSolver/ElectrostaticSolver.H"
#include "FieldSolver/MagnetostaticSolver/MagnetostaticSolver.H"
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "Filter/BilinearFilter.H"
#include "Parallelization/GuardCellManager.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/export.H"

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_Array.H>
#include <AMReX_Config.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>
#include <AMReX_VisMF.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class WARPX_EXPORT WarpX
    : public amrex::AmrCore
{
public:

    static WarpX& GetInstance ();

    static void ResetInstance ();

    /**
     * \brief
     * This method has to be called at the end of the simulation. It deletes the WarpX instance.
     */
    static void Finalize();

    /** Destructor */
    ~WarpX () override;

    /** Copy constructor */
    WarpX ( WarpX const &) = delete;
    /** Copy operator */
    WarpX& operator= ( WarpX const & ) = delete;

    /** Move constructor */
    WarpX ( WarpX && ) = default;
    /** Move operator */
    WarpX& operator= ( WarpX && ) = default;

    [[nodiscard]] int Verbose () const { return verbose; }

    void InitData ();

    void Evolve (int numsteps = -1);

    //
    // Functions used by implicit solvers
    //
    void ImplicitPreRHSOp ( amrex::Real  cur_time,
                            amrex::Real  a_full_dt,
                            int          a_nl_iter,
                            bool         a_from_jacobian );
    void SaveParticlesAtImplicitStepStart ();
    void FinishImplicitParticleUpdate ();
    void SetElectricFieldAndApplyBCs ( const WarpXSolverVec&  a_E );
    void UpdateMagneticFieldAndApplyBCs ( const amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > >& a_Bn,
                               amrex::Real a_thetadt );
    void ApplyMagneticFieldBCs ();
    void FinishMagneticFieldAndApplyBCs ( const amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > >& a_Bn,
                               amrex::Real a_theta );
    void FinishImplicitField ( amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > >& Field_fp,
                         const amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > >& Field_n,
                         amrex::Real theta );
    void ImplicitComputeRHSE (         amrex::Real dt, WarpXSolverVec& a_Erhs_vec);
    void ImplicitComputeRHSE (int lev, amrex::Real dt, WarpXSolverVec& a_Erhs_vec);
    void ImplicitComputeRHSE (int lev, PatchType patch_type, amrex::Real dt, WarpXSolverVec& a_Erhs_vec);

    MultiParticleContainer& GetPartContainer () { return *mypc; }
    MultiFluidContainer& GetFluidContainer () { return *myfl; }
    MacroscopicProperties& GetMacroscopicProperties () { return *m_macroscopic_properties; }
    HybridPICModel& GetHybridPICModel () { return *m_hybrid_pic_model; }
    [[nodiscard]] HybridPICModel * get_pointer_HybridPICModel () const { return m_hybrid_pic_model.get(); }
    MultiDiagnostics& GetMultiDiags () {return *multi_diags;}
#ifdef AMREX_USE_EB
    amrex::Vector<std::unique_ptr<amrex::MultiFab> >& GetDistanceToEB () {return m_distance_to_eb;}
#endif
    ParticleBoundaryBuffer& GetParticleBoundaryBuffer () { return *m_particle_boundary_buffer; }

    static void shiftMF (amrex::MultiFab& mf, const amrex::Geometry& geom,
                         int num_shift, int dir, int lev, bool update_cost_flag,
                         amrex::Real external_field=0.0, bool useparser = false,
                         amrex::ParserExecutor<3> const& field_parser={});

    /**
     * \brief
     * If an authors' string is specified in the inputfile, this method returns that string.
     * Otherwise, it returns an empty string.
     */
    [[nodiscard]] std::string GetAuthors () const { return m_authors; }

    /** Maximum level up to which the externally defined electric and magnetic fields are initialized.
     *  The default value is set to the max levels in the simulation.
     *  if lev > maxlevel_extEMfield_init, the fields on those levels will have a default value of 0
     */
    int maxlevel_extEMfield_init;

    // Algorithms
    //! Integer that corresponds to the current deposition algorithm (Esirkepov, direct, Vay, Villasenor)
    static short current_deposition_algo;
    //! Integer that corresponds to the charge deposition algorithm (only standard deposition)
    static short charge_deposition_algo;
    //! Integer that corresponds to the field gathering algorithm (energy-conserving, momentum-conserving)
    static short field_gathering_algo;
    //! Integer that corresponds to the particle push algorithm (Boris, Vay, Higuera-Cary)
    static short particle_pusher_algo;
    //! Integer that corresponds to the type of Maxwell solver (Yee, CKC, PSATD, ECT)
    static short electromagnetic_solver_id;
    //! Integer that corresponds to the evolve scheme (explicit, semi_implicit_em, theta_implicit_em)
    static short evolve_scheme;
    //! Maximum iterations used for self-consistent particle update in implicit particle-suppressed evolve schemes
    static int max_particle_its_in_implicit_scheme;
    //! Relative tolerance used for self-consistent particle update in implicit particle-suppressed evolve schemes
    static amrex::ParticleReal particle_tol_in_implicit_scheme;
    /** Records a number corresponding to the load balance cost update strategy
     *  being used (0 or 1 corresponding to timers or heuristic).
     */
    static short load_balance_costs_update_algo;
    //! Integer that corresponds to electromagnetic Maxwell solver (vacuum - 0, macroscopic - 1)
    static int em_solver_medium;
    /** Integer that correspond to macroscopic Maxwell solver algorithm
     *  (BackwardEuler - 0, Lax-Wendroff - 1)
     */
    static int macroscopic_solver_algo;
    /** Integers that correspond to boundary condition applied to fields at the
     *  lower domain boundaries
     *  (0 to 6 correspond to PML, Periodic, PEC, PMC, Damped, Absorbing Silver-Mueller, None)
     */
    static amrex::Vector<FieldBoundaryType> field_boundary_lo;
    /** Integers that correspond to boundary condition applied to fields at the
     *  upper domain boundaries
     *  (0 to 6 correspond to PML, Periodic, PEC, PMC, Damped, Absorbing Silver-Mueller, None)
     */
    static amrex::Vector<FieldBoundaryType> field_boundary_hi;
    /** Integers that correspond to boundary condition applied to particles at the
     *  lower domain boundaries
     *  (0 to 4 correspond to Absorbing, Open, Reflecting, Periodic, Thermal)
     */
    static amrex::Vector<ParticleBoundaryType> particle_boundary_lo;
    /** Integers that correspond to boundary condition applied to particles at the
     *  upper domain boundaries
     *  (0 to 4 correspond to Absorbing, Open, Reflecting, Periodic, Thermal)
     */
    static amrex::Vector<ParticleBoundaryType> particle_boundary_hi;

    //! Integer that corresponds to the order of the PSATD solution
    //! (whether the PSATD equations are derived from first-order or
    //! second-order solution)
    static short psatd_solution_type;

    //! Integers that correspond to the time dependency of J (constant, linear)
    //! and rho (linear, quadratic) for the PSATD algorithm
    static short J_in_time;
    static short rho_in_time;

    //! If true, the current is deposited on a nodal grid and then centered onto a staggered grid
    //! using finite centering of order given by #current_centering_nox, #current_centering_noy,
    //! and #current_centering_noz
    bool do_current_centering = false;

    //! If true, a correction is applied to the current in Fourier space,
    //  to satisfy the continuity equation and charge conservation
    bool current_correction;

    //! If true, the PSATD update equation for E contains both J and rho
    //! (default is false for standard PSATD and true for Galilean PSATD)
    bool update_with_rho = false;

    //! perform field communications in single precision
    static bool do_single_precision_comms;

    //! used shared memory algorithm for charge deposition
    static bool do_shared_mem_charge_deposition;

    //! use shared memory algorithm for current deposition
    static bool do_shared_mem_current_deposition;

    //! number of threads to use per block in shared deposition
    static int shared_mem_current_tpb;

    //! tileSize to use for shared current deposition operations
    static amrex::IntVect shared_tilesize;

    //! Whether to fill guard cells when computing inverse FFTs of fields
    static amrex::IntVect m_fill_guards_fields;

    //! Whether to fill guard cells when computing inverse FFTs of currents
    static amrex::IntVect m_fill_guards_current;

    //! Solve additional Maxwell equation for F in order to control errors in Gauss' law
    //! (useful when using current deposition algorithms that are not charge-conserving)
    static bool do_dive_cleaning;
    //! Solve additional Maxwell equation for G in order to control errors in magnetic Gauss' law
    static bool do_divb_cleaning;

    //! Order of the particle shape factors (splines) along x
    static int nox;
    //! Order of the particle shape factors (splines) along y
    static int noy;
    //! Order of the particle shape factors (splines) along z
    static int noz;

    //! Order of finite centering of fields (from staggered grid to nodal grid), along x
    static int field_centering_nox;
    //! Order of finite centering of fields (from staggered grid to nodal grid), along y
    static int field_centering_noy;
    //! Order of finite centering of fields (from staggered grid to nodal grid), along z
    static int field_centering_noz;

    //! Order of finite centering of currents (from nodal grid to staggered grid), along x
    static int current_centering_nox;
    //! Order of finite centering of currents (from nodal grid to staggered grid), along y
    static int current_centering_noy;
    //! Order of finite centering of currents (from nodal grid to staggered grid), along z
    static int current_centering_noz;

    //! Number of modes for the RZ multi-mode version
    static int n_rz_azimuthal_modes;
    //! Number of MultiFab components
    //! (with the RZ multi-mode version, each mode has a real and imaginary component,
    //! except mode 0 which is purely real: component 0 is mode 0, odd components are
    //! the real parts, even components are the imaginary parts)
    static int ncomps;

    //! If true, a Numerical Cherenkov Instability (NCI) corrector is applied
    //! (for simulations using the FDTD Maxwell solver)
    static bool use_fdtd_nci_corr;
    //! If true (Galerkin method): The fields are interpolated directly from the staggered
    //! grid to the particle positions (with reduced interpolation order in the parallel
    //! direction). This scheme is energy conserving in the limit of infinitely small time
    //! steps.
    //! Otherwise, "momentum conserving" (in the same limit): The fields are first
    //! interpolated from the staggered grid points to the corners of each cell, and then
    //! from the cell corners to the particle position (with the same order of interpolation
    //! in all directions). This scheme is momentum conserving in the limit of infinitely
    //! small time steps.
    static bool galerkin_interpolation;

    //! Flag whether the Verboncoeur correction is applied to the current and charge density
    //! on the axis when using RZ.
    static bool verboncoeur_axis_correction;

    //! If true, a bilinear filter is used to smooth charge and currents
    static bool use_filter;
    //! If true, the bilinear filtering of charge and currents is done in Fourier space
    static bool use_kspace_filter;
    //! If true, a compensation step is added to the bilinear filtering of charge and currents
    static bool use_filter_compensation;

    //! If true, the initial conditions from random number generators are serialized (useful for reproducible testing with OpenMP)
    static bool serialize_initial_conditions;

    //! Lorentz factor of the boosted frame in which a boosted-frame simulation is run
    static amrex::Real gamma_boost;
    //! Beta value corresponding to the Lorentz factor of the boosted frame of the simulation
    static amrex::Real beta_boost;
    //! Direction of the Lorentz transform that defines the boosted frame of the simulation
    static amrex::Vector<int> boost_direction;
    //! If specified, the maximum number of iterations is computed automatically so that
    //! the lower end of the simulation domain along z reaches #zmax_plasma_to_compute_max_step
    //! in the boosted frame
    static amrex::Real zmax_plasma_to_compute_max_step;
    //! Set to true if #zmax_plasma_to_compute_max_step is specified, in which case
    //! the maximum number of iterations is computed automatically so that the lower end of the
    //! simulation domain along z reaches #zmax_plasma_to_compute_max_step in the boosted frame
    static bool do_compute_max_step_from_zmax;
    //! store initial value of zmin_domain_boost because WarpX::computeMaxStepBoostAccelerator
    //! needs the initial value of zmin_domain_boost, even if restarting from a checkpoint file
    static amrex::Real zmin_domain_boost_step_0;

    //! If true, the code will compute max_step from the back transformed diagnostics
    static bool compute_max_step_from_btd;

    static bool do_dynamic_scheduling;
    static bool refine_plasma;

    static utils::parser::IntervalsParser sort_intervals;
    static amrex::IntVect sort_bin_size;

    //! If true, particles will be sorted in the order x -> y -> z -> ppc for faster deposition
    static bool sort_particles_for_deposition;
    //! Specifies the type of grid used for the above sorting, i.e. cell-centered, nodal, or mixed
    static amrex::IntVect sort_idx_type;

    static bool do_subcycling;
    static bool do_multi_J;
    static int do_multi_J_n_depositions;

    static bool do_device_synchronize;
    static bool safe_guard_cells;

    //! With mesh refinement, particles located inside a refinement patch, but within
    //! #n_field_gather_buffer cells of the edge of the patch, will gather the fields
    //! from the lower refinement level instead of the refinement patch itself
    static int n_field_gather_buffer;
    //! With mesh refinement, particles located inside a refinement patch, but within
    //! #n_current_deposition_buffer cells of the edge of the patch, will deposit their charge
    //! and current onto the lower refinement level instead of the refinement patch itself
    static int n_current_deposition_buffer;

    //! Integer that corresponds to the type of grid used in the simulation
    //! (collocated, staggered, hybrid)
    static short grid_type;

    // Global rho nodal flag to know about rho index type when rho MultiFab is not allocated
    amrex::IntVect m_rho_nodal_flag;

    /**
     * \brief
     * Allocate and optionally initialize the MultiFab. This also adds the MultiFab
     * to the map of MultiFabs (used to ease the access to MultiFabs from the Python
     * interface
     *
     * \param[out] mf The MultiFab unique pointer to be allocated
     * \param[in] ba The BoxArray describing the MultiFab
     * \param[in] dm The DistributionMapping describing the MultiFab
     * \param[in] ncomp The number of components in the MultiFab
     * \param[in] ngrow The number of guard cells in the MultiFab
     * \param[in] level The refinement level
     * \param[in] name The name of the MultiFab to use in the map
     * \param[in] initial_value The optional initial value
     */
    static void AllocInitMultiFab (
        std::unique_ptr<amrex::MultiFab>& mf,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomp,
        const amrex::IntVect& ngrow,
        int level,
        const std::string& name,
        std::optional<const amrex::Real> initial_value = {});

    /**
     * \brief
     * Allocate and optionally initialize the iMultiFab. This also adds the iMultiFab
     * to the map of MultiFabs (used to ease the access to MultiFabs from the Python
     * interface
     *
     * \param[out] mf The iMultiFab unique pointer to be allocated
     * \param[in] ba The BoxArray describing the iMultiFab
     * \param[in] dm The DistributionMapping describing the iMultiFab
     * \param[in] ncomp The number of components in the iMultiFab
     * \param[in] ngrow The number of guard cells in the iMultiFab
     * \param[in] level The refinement level
     * \param[in] name The name of the iMultiFab to use in the map
     * \param[in] initial_value The optional initial value
     */
    static void AllocInitMultiFab (
        std::unique_ptr<amrex::iMultiFab>& mf,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomp,
        const amrex::IntVect& ngrow,
        int level,
        const std::string& name,
        std::optional<const int> initial_value = {});

    /**
     * \brief
     * Create an alias of a MultiFab, adding the alias to the MultiFab map
     * \param[out] mf The MultiFab to create
     * \param[in] mf_to_alias The MultiFab to alias
     * \param[in] scomp The starting component to be aliased
     * \param[in] ncomp The number of components to alias
     * \param[in] level The refinement level
     * \param[in] name The name of the MultiFab to use in the map
     * \param[in] initial_value optional initial value for MultiFab
     */
    static void AliasInitMultiFab (
        std::unique_ptr<amrex::MultiFab>& mf,
        const amrex::MultiFab& mf_to_alias,
        int scomp,
        int ncomp,
        int level,
        const std::string& name,
        std::optional<const amrex::Real> initial_value);

    /**
     * \brief
     * Allocate the MultiFab so that is like the specified MultiFab (same ba and dm)
     * and optionally initialize it. This also adds the MultiFab
     * to the map of MultiFabs (used to ease the access to MultiFabs from the Python
     * interface
     *
     * \param mf[out] The MultiFab unique pointer to be allocated
     * \param mf_model[in] The MultiFab to model
     * \param name[in] The name of the MultiFab to use in the map
     * \param initial_value[in] The optional initial value
     */
    static void AllocInitMultiFabFromModel (
        std::unique_ptr<amrex::MultiFab>& mf,
        amrex::MultiFab& mf_model,
        int level,
        const std::string& name,
        std::optional<const amrex::Real> initial_value = {});

    // Maps of all of the MultiFabs and iMultiFabs used (this can include MFs from other classes)
    // This is a convenience for the Python interface, allowing all MultiFabs
    // to be easily referenced from Python.
    static std::map<std::string, amrex::MultiFab *> multifab_map;
    static std::map<std::string, amrex::iMultiFab *> imultifab_map;

    /**
     * \brief
     * Check if a field is initialized.
     *
     * \param field_type[in] the field type
     * \param lev[in] the mesh refinement level
     * \param direction[in] the field component (0 by default)
     *
     * \return true if the field is initialized, false otherwise
     */
    [[nodiscard]] bool
    isFieldInitialized (warpx::fields::FieldType field_type, int lev, int direction = 0) const;

    /**
     * \brief
     * Get a pointer to the field data.
     *
     * \param field_type[in] the field type
     * \param lev[in] the mesh refinement level
     * \param direction[in] the field component (0 by default)
     *
     * \return the pointer to an amrex::MultiFab containing the field data
     */
    [[nodiscard]] amrex::MultiFab*
    getFieldPointer (warpx::fields::FieldType field_type, int lev, int direction = 0) const;

    /**
     * \brief
     * For vector fields, get an array of three pointers to the field data.
     *
     * \param field_type[in] the field type
     * \param lev[in] the mesh refinement level
     *
     * \return an array of three pointers amrex::MultiFab* containing the field data
     */
    [[nodiscard]] std::array<const amrex::MultiFab* const, 3>
    getFieldPointerArray (warpx::fields::FieldType field_type, int lev) const;

    /**
     * \brief
     * Get a constant reference to the field data.
     *
     * \param field_type[in] the field type
     * \param lev[in] the mesh refinement level
     * \param direction[in] the field component (0 by default)
     *
     * \return a constant refernce to an amrex::MultiFab containing the field data
     */
    [[nodiscard]] const amrex::MultiFab&
    getField(warpx::fields::FieldType field_type, int lev, int direction = 0) const;

    /**
     * \brief
     * Get a constant reference to the specified vector field on the different MR levels
     *
     * \param field_type[in] the field type
     *
     * \return a vector (which one element per MR level) of arrays of three pointers (for 3 vector components) amrex::MultiFab* containing the field data
     */
    [[nodiscard]] const amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>,3>>&
    getMultiLevelField(warpx::fields::FieldType field_type) const;

    [[nodiscard]] bool DoPML () const {return do_pml;}
    [[nodiscard]] bool DoFluidSpecies () const {return do_fluid_species;}

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    const PML_RZ* getPMLRZ() {return pml_rz[0].get();}
#endif

    /** get low-high-low-high-... vector for each direction indicating if mother grid PMLs are enabled */
    [[nodiscard]] std::vector<bool> getPMLdirections() const;

    static amrex::LayoutData<amrex::Real>* getCosts (int lev);

    void setLoadBalanceEfficiency (int lev, amrex::Real efficiency);

    amrex::Real getLoadBalanceEfficiency (int lev);

    static amrex::IntVect filter_npass_each_dir;
    BilinearFilter bilinear_filter;
    amrex::Vector< std::unique_ptr<NCIGodfreyFilter> > nci_godfrey_filter_exeybz;
    amrex::Vector< std::unique_ptr<NCIGodfreyFilter> > nci_godfrey_filter_bxbyez;

    amrex::Real time_of_last_gal_shift = 0;
    amrex::Vector<amrex::Real> m_v_galilean = amrex::Vector<amrex::Real>(3, amrex::Real(0.));
    amrex::Array<amrex::Real,3> m_galilean_shift = {{0}};

    amrex::Vector<amrex::Real> m_v_comoving = amrex::Vector<amrex::Real>(3, amrex::Real(0.));

    static int num_mirrors;
    amrex::Vector<amrex::Real> mirror_z;
    amrex::Vector<amrex::Real> mirror_z_width;
    amrex::Vector<int> mirror_z_npoints;

    /// object with all reduced diagnostics, similar to MultiParticleContainer for species.
    std::unique_ptr<MultiReducedDiags> reduced_diags;

    void applyMirrors(amrex::Real time);

    /** Determine the timestep of the simulation. */
    void ComputeDt ();

    /** Print main PIC parameters to stdout */
    void PrintMainPICparameters ();

    /** Write a file that record all inputs: inputs file + command line options */
    void WriteUsedInputsFile () const;

    /** Print dt and dx,dy,dz */
    void PrintDtDxDyDz ();

    /**
     * \brief
     * Compute the last time step of the simulation
     * Calls computeMaxStepBoostAccelerator() if required.
     */
    void ComputeMaxStep ();
    // Compute max_step automatically for simulations in a boosted frame.
    void computeMaxStepBoostAccelerator();

    /** \brief Move the moving window
     * \param step Time step
     * \param move_j whether the current (and the charge, if allocated) is shifted or not
     */
    int  MoveWindow (int step, bool move_j);

    /**
     * \brief
     * This function shifts the boundary of the grid by 'm_v_galilean*dt'.
     * In doding so, only positions attributes are changed while fields remain unchanged.
     */
    void ShiftGalileanBoundary ();

    /**
     * \brief Update injection position for continuous injection of
     *        particles and lasers (loops over species and lasers).
     */
    void UpdateInjectionPosition (amrex::Real dt);

    void ResetProbDomain (const amrex::RealBox& rb);
    void EvolveE (         amrex::Real dt);
    void EvolveE (int lev, amrex::Real dt);
    void EvolveB (         amrex::Real dt, DtType dt_type);
    void EvolveB (int lev, amrex::Real dt, DtType dt_type);
    void EvolveF (         amrex::Real dt, DtType dt_type);
    void EvolveF (int lev, amrex::Real dt, DtType dt_type);
    void EvolveG (         amrex::Real dt, DtType dt_type);
    void EvolveG (int lev, amrex::Real dt, DtType dt_type);
    void EvolveB (int lev, PatchType patch_type, amrex::Real dt, DtType dt_type);
    void EvolveE (int lev, PatchType patch_type, amrex::Real dt);
    void EvolveF (int lev, PatchType patch_type, amrex::Real dt, DtType dt_type);
    void EvolveG (int lev, PatchType patch_type, amrex::Real dt, DtType dt_type);

    void MacroscopicEvolveE (         amrex::Real dt);
    void MacroscopicEvolveE (int lev, amrex::Real dt);
    void MacroscopicEvolveE (int lev, PatchType patch_type, amrex::Real dt);

    /**
     * \brief Hybrid-PIC field evolve function.
     * This function contains the logic involved in evolving the electric and
     * magnetic fields in the hybrid PIC algorithm.
     */
    void HybridPICEvolveFields ();

    /**
     * \brief Hybrid-PIC initial deposition function.
     * The hybrid-PIC algorithm uses the charge and current density from both
     * the current and previous step when updating the fields. This function
     * deposits the initial ion charge and current (before the first particle
     * push), to be used in the ``HybridPICEvolveFields()`` function.
     */
    void HybridPICDepositInitialRhoAndJ ();

    /** apply QED correction on electric field
     *
     * \param dt vector of time steps (for all levels)
     */
    void Hybrid_QED_Push (         amrex::Vector<amrex::Real> dt);

    /** apply QED correction on electric field for level lev
     *
     * \param lev mesh refinement level
     * \param dt time step
     */
    void Hybrid_QED_Push (int lev, amrex::Real dt);

    /** apply QED correction on electric field for level lev and patch type patch_type
     *
     * \param lev mesh refinement level
     * \param patch_type which MR patch: PatchType::fine or PatchType::coarse
     * \param dt time step
     */
    void Hybrid_QED_Push (int lev, PatchType patch_type, amrex::Real dt);

    static amrex::Real quantum_xi_c2;

    /** Check and potentially compute load balancing
     */
    void CheckLoadBalance (int step);

    /** \brief perform load balance; compute and communicate new `amrex::DistributionMapping`
     */
    void LoadBalance ();

    /** \brief resets costs to zero
     */
    void ResetCosts ();

    /** Perform running average of the LB costs
     *
     * Only needed for timers cost update, heuristic load balance considers the
     * instantaneous costs.
     * This gives more importance to most recent costs.
     */
    void RescaleCosts (int step);

    /** \brief returns the load balance interval
     */
    [[nodiscard]] utils::parser::IntervalsParser get_load_balance_intervals () const
    {
        return load_balance_intervals;
    }

    /**
     * \brief Private function for spectral solver
     * Applies a damping factor in the guards cells that extend
     * beyond the extent of the domain, reducing fluctuations that
     * can appear in parallel simulations. This will be called
     * when FieldBoundaryType is set to damped. Vector version.
     */
    void DampFieldsInGuards (int lev,
                             const std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
                             const std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield);

    /**
     * \brief Private function for spectral solver
     * Applies a damping factor in the guards cells that extend
     * beyond the extent of the domain, reducing fluctuations that
     * can appear in parallel simulations. This will be called
     * when FieldBoundaryType is set to damped. Scalar version.
     */
    void DampFieldsInGuards (int lev, std::unique_ptr<amrex::MultiFab>& mf);

#ifdef WARPX_DIM_RZ
    void ApplyInverseVolumeScalingToCurrentDensity(amrex::MultiFab* Jx,
                                                   amrex::MultiFab* Jy,
                                                   amrex::MultiFab* Jz,
                                                   int lev);

    void ApplyInverseVolumeScalingToChargeDensity(amrex::MultiFab* Rho,
                                                  int lev);
#endif

    /**
     * \brief If a PEC boundary conditions is used the charge
     * density deposited in the guard cells have to be reflected back into
     * the simulation domain. This function performs that logic.
     */
    void ApplyRhofieldBoundary (int lev, amrex::MultiFab* Rho,
                                PatchType patch_type);

    /**
     * \brief If a PEC boundary conditions is used the current
     * density deposited in the guard cells have to be reflected back into
     * the simulation domain. This function performs that logic.
     */
    void ApplyJfieldBoundary (int lev, amrex::MultiFab* Jx,
                              amrex::MultiFab* Jy, amrex::MultiFab* Jz,
                              PatchType patch_type);

    void ApplyEfieldBoundary (int lev, PatchType patch_type);
    void ApplyBfieldBoundary (int lev, PatchType patch_type, DtType dt_type);

#ifdef WARPX_DIM_RZ
    // Applies the boundary conditions that are specific to the axis when in RZ.
    void ApplyFieldBoundaryOnAxis (amrex::MultiFab* Er, amrex::MultiFab* Et, amrex::MultiFab* Ez, int lev) const;
#endif

    /**
     * \brief When the Ohm's law solver is used, the electron pressure values
     * on PEC boundaries are set to enforce a zero derivative Neumann condition,
     * otherwise the E_perp values have large spikes (that grows as 1/dx). This
     * has to be done since the E-field is algebraically calculated instead of
     * evolved which means the 1/dx growth is not avoided by multiplication
     * with dt as happens in the B-field evolve.
     */
    void ApplyElectronPressureBoundary (int lev, PatchType patch_type);

    void DampPML ();
    void DampPML (int lev);
    void DampPML (int lev, PatchType patch_type);
    void DampPML_Cartesian (int lev, PatchType patch_type);

    void DampJPML ();
    void DampJPML (int lev);
    void DampJPML (int lev, PatchType patch_type);

    void CopyJPML ();
    bool isAnyBoundaryPML();
    /** True if any of the particle boundary condition type is Thermal */
    static bool isAnyParticleBoundaryThermal();

    PML* GetPML (int lev);
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    PML_RZ* GetPML_RZ (int lev);
#endif

    /** Run the ionization module on all species */
    void doFieldIonization ();
    /** Run the ionization module on all species at level lev
     * \param lev level
     */
    void doFieldIonization (int lev);

#ifdef WARPX_QED
    /** Run the QED module on all species */
    void doQEDEvents ();
    /** Run the QED module on all species at level lev
     * \param lev level
     */
    void doQEDEvents (int lev);
#endif

    void PushParticlesandDeposit (int lev, amrex::Real cur_time, DtType a_dt_type=DtType::Full, bool skip_current=false,
                                 PushType push_type=PushType::Explicit);
    void PushParticlesandDeposit (amrex::Real cur_time, bool skip_current=false,
                                 PushType push_type=PushType::Explicit);

    // This function does aux(lev) = fp(lev) + I(aux(lev-1)-cp(lev)).
    // Caller must make sure fp and cp have ghost cells filled.
    void UpdateAuxilaryData ();
    void UpdateAuxilaryDataStagToNodal ();
    void UpdateAuxilaryDataSameType ();

    /**
     * \brief This function is called if \c warpx.do_current_centering = 1 and
     * it centers the currents from a nodal grid to a staggered grid (Yee) using
     * finite-order interpolation based on the Fornberg coefficients.
     *
     * \param[in,out] dst destination \c MultiFab where the results of the finite-order centering are stored
     * \param[in] src source \c MultiFab that contains the values of the nodal current to be centered
     */
    void UpdateCurrentNodalToStag (amrex::MultiFab& dst, amrex::MultiFab const& src);

    // Fill boundary cells including coarse/fine boundaries
    void FillBoundaryB   (amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryE   (amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryB_avg   (amrex::IntVect ng);
    void FillBoundaryE_avg   (amrex::IntVect ng);

    void FillBoundaryF   (amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryG   (amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryAux (amrex::IntVect ng);
    void FillBoundaryE   (int lev, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryB   (int lev, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryE_avg   (int lev, amrex::IntVect ng);
    void FillBoundaryB_avg   (int lev, amrex::IntVect ng);

    void FillBoundaryF   (int lev, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryG   (int lev, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryAux (int lev, amrex::IntVect ng);

    /**
     * \brief Synchronize J and rho:
     * filter (if used), exchange guard cells, interpolate across MR levels
     * and apply boundary conditions.
     * Contains separate calls to WarpX::SyncCurrent and WarpX::SyncRho.
     */
    void SyncCurrentAndRho ();

    /**
     * \brief Apply filter and sum guard cells across MR levels.
     * If current centering is used, center the current from a nodal grid
     * to a staggered grid. For each MR level beyond level 0, interpolate
     * the fine-patch current onto the coarse-patch current at the same level.
     * Then, for each MR level, including level 0, apply filter and sum guard
     * cells across levels.
     *
     * \param[in,out] J_fp reference to fine-patch current \c MultiFab (all MR levels)
     * \param[in,out] J_cp reference to coarse-patch current \c MultiFab (all MR levels)
     * \param[in,out] J_buffer reference to buffer current \c MultiFab (all MR levels)
     */
    void SyncCurrent (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_buffer);

    void SyncRho ();

    void SyncRho (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_buffer);

    [[nodiscard]] amrex::Vector<int> getnsubsteps () const {return nsubsteps;}
    [[nodiscard]] int getnsubsteps (int lev) const {return nsubsteps[lev];}
    [[nodiscard]] amrex::Vector<int> getistep () const {return istep;}
    [[nodiscard]] int getistep (int lev) const {return istep[lev];}
    void setistep (int lev, int ii) {istep[lev] = ii;}
    [[nodiscard]] amrex::Vector<amrex::Real> gett_old () const {return t_old;}
    [[nodiscard]] amrex::Real gett_old (int lev) const {return t_old[lev];}
    [[nodiscard]] amrex::Vector<amrex::Real> gett_new () const {return t_new;}
    [[nodiscard]] amrex::Real gett_new (int lev) const {return t_new[lev];}
    void sett_new (int lev, amrex::Real time) {t_new[lev] = time;}
    [[nodiscard]] amrex::Vector<amrex::Real> getdt () const {return dt;}
    [[nodiscard]] amrex::Real getdt (int lev) const {return dt.at(lev);}
    [[nodiscard]] int getdo_moving_window() const {return do_moving_window;}
    [[nodiscard]] amrex::Real getmoving_window_x() const {return moving_window_x;}
    [[nodiscard]] bool getis_synchronized() const {return is_synchronized;}

    [[nodiscard]] int maxStep () const {return max_step;}
    void updateMaxStep (const int new_max_step) {max_step = new_max_step;}
    [[nodiscard]] amrex::Real stopTime () const {return stop_time;}
    void updateStopTime (const amrex::Real new_stop_time) {stop_time = new_stop_time;}

    void AverageAndPackFields( amrex::Vector<std::string>& varnames,
        amrex::Vector<amrex::MultiFab>& mf_avg, amrex::IntVect ngrow) const;

    void prepareFields( int step, amrex::Vector<std::string>& varnames,
        amrex::Vector<amrex::MultiFab>& mf_avg,
        amrex::Vector<const amrex::MultiFab*>& output_mf,
        amrex::Vector<amrex::Geometry>& output_geom ) const;

    static std::array<amrex::Real,3> CellSize (int lev);
    static amrex::XDim3 InvCellSize (int lev);
    static amrex::RealBox getRealBox(const amrex::Box& bx, int lev);

    /**
     * \brief Return the lower corner of the box in real units.
     * \param bx The box
     * \param lev The refinement level of the box
     * \param time_shift_delta The time relative to the current time at which to calculate the position
     *                         (when v_galilean is not zero)
     * \return An array of the position coordinates
     */
    static amrex::XDim3 LowerCorner (const amrex::Box& bx, int lev, amrex::Real time_shift_delta);
    /**
     * \brief Return the upper corner of the box in real units.
     * \param bx The box
     * \param lev The refinement level of the box
     * \param time_shift_delta The time relative to the current time at which to calculate the position
     *                         (when v_galilean is not zero)
     * \return An array of the position coordinates
     */
    static amrex::XDim3 UpperCorner (const amrex::Box& bx, int lev, amrex::Real time_shift_delta);

    static amrex::IntVect RefRatio (int lev);

    static const amrex::iMultiFab* CurrentBufferMasks (int lev);
    static const amrex::iMultiFab* GatherBufferMasks (int lev);

    static int electrostatic_solver_id;
    static int poisson_solver_id;

    // Parameters for lab frame electrostatic
    static amrex::Real self_fields_required_precision;
    static amrex::Real self_fields_absolute_tolerance;
    static int self_fields_max_iters;
    static int self_fields_verbosity;

    static int do_moving_window; // boolean
    static int start_moving_window_step; // the first step to move window
    static int end_moving_window_step; // the last step to move window
    /** Returns true if the moving window is active for the provided step
     *
     * @param step time step
     * @return true if active, else false
     */
    static int moving_window_active (int const step) {
        bool const step_before_end = (step < end_moving_window_step) || (end_moving_window_step < 0);
        bool const step_after_start = (step >= start_moving_window_step);
        return do_moving_window && step_before_end && step_after_start;
    }
    static int moving_window_dir;
    static amrex::Real moving_window_v;
    static bool fft_do_time_averaging;

    // these should be private, but can't due to Cuda limitations
    static void ComputeDivB (amrex::MultiFab& divB, int dcomp,
                             const std::array<const amrex::MultiFab* const, 3>& B,
                             const std::array<amrex::Real,3>& dx);

    static void ComputeDivB (amrex::MultiFab& divB, int dcomp,
                             const std::array<const amrex::MultiFab* const, 3>& B,
                             const std::array<amrex::Real,3>& dx, amrex::IntVect ngrow);

    void ComputeDivE(amrex::MultiFab& divE, int lev);

    [[nodiscard]] amrex::IntVect getngEB() const { return guard_cells.ng_alloc_EB; }
    [[nodiscard]] amrex::IntVect getngF() const { return guard_cells.ng_alloc_F; }
    [[nodiscard]] amrex::IntVect getngUpdateAux() const { return guard_cells.ng_UpdateAux; }
    [[nodiscard]] amrex::IntVect get_ng_depos_J() const {return guard_cells.ng_depos_J;}
    [[nodiscard]] amrex::IntVect get_ng_depos_rho() const {return guard_cells.ng_depos_rho;}
    [[nodiscard]] amrex::IntVect get_ng_fieldgather () const {return guard_cells.ng_FieldGather;}

    /** Coarsest-level Domain Decomposition
     *
     * If specified, the domain will be chopped into the exact number
     * of pieces in each dimension as specified by this parameter.
     *
     * @return the number of MPI processes per dimension if specified, otherwise a 0-vector
     */
    [[nodiscard]] amrex::IntVect get_numprocs() const {return numprocs;}

    bool m_boundary_potential_specified = false;
    ElectrostaticSolver::PoissonBoundaryHandler m_poisson_boundary_handler;
    void ComputeSpaceChargeField (bool reset_fields);
    void AddBoundaryField ();
    void AddSpaceChargeField (WarpXParticleContainer& pc);
    void AddSpaceChargeFieldLabFrame ();
    void computePhi (const amrex::Vector<std::unique_ptr<amrex::MultiFab> >& rho,
                     amrex::Vector<std::unique_ptr<amrex::MultiFab> >& phi,
                     std::array<amrex::Real, 3> beta = {{0,0,0}},
                     amrex::Real required_precision=amrex::Real(1.e-11),
                     amrex::Real absolute_tolerance=amrex::Real(0.0),
                     int max_iters=200,
                     int verbosity=2) const;

    void setPhiBC (amrex::Vector<std::unique_ptr<amrex::MultiFab> >& phi ) const;

    void computeE (amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3> >& E,
                   const amrex::Vector<std::unique_ptr<amrex::MultiFab> >& phi,
                   std::array<amrex::Real, 3> beta = {{0,0,0}} ) const;
    void computeB (amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3> >& B,
                   const amrex::Vector<std::unique_ptr<amrex::MultiFab> >& phi,
                   std::array<amrex::Real, 3> beta = {{0,0,0}} ) const;
    void computePhiTriDiagonal (const amrex::Vector<std::unique_ptr<amrex::MultiFab> >& rho,
                                      amrex::Vector<std::unique_ptr<amrex::MultiFab> >& phi) const;

    // Magnetostatic Solver Interface
    MagnetostaticSolver::VectorPoissonBoundaryHandler m_vector_poisson_boundary_handler;
    void ComputeMagnetostaticField ();
    void AddMagnetostaticFieldLabFrame ();
    void computeVectorPotential (const amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> >& curr,
                                 amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> >& A,
                                 amrex::Real required_precision=amrex::Real(1.e-11),
                                 amrex::Real absolute_tolerance=amrex::Real(0.0),
                                 int max_iters=200,
                                 int verbosity=2) const;

    void setVectorPotentialBC (amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> >& A) const;

    /**
     * \brief
     * This function initializes the E and B fields on each level
     * using the parser and the user-defined function for the external fields.
     * The subroutine will parse the x_/y_z_external_grid_function and
     * then, the field multifab is initialized based on the (x,y,z) position
     * on the staggered yee-grid or cell-centered grid, in the interior cells
     * and guard cells.
     *
     * \param[in] mfx x-component of the field to be initialized
     * \param[in] mfy y-component of the field to be initialized
     * \param[in] mfz z-component of the field to be initialized
     * \param[in] xfield_parser parser function to initialize x-field
     * \param[in] yfield_parser parser function to initialize y-field
     * \param[in] zfield_parser parser function to initialize z-field
     * \param[in] edge_lengths edge lengths information
     * \param[in] face_areas face areas information
     * \param[in] field flag indicating which field is being initialized ('E' for electric, 'B' for magnetic)
     * \param[in] lev level of the Multifabs that is initialized
     * \param[in] patch_type PatchType on which the field is initialized (fine or coarse)
     */
    void InitializeExternalFieldsOnGridUsingParser (
         amrex::MultiFab *mfx, amrex::MultiFab *mfy, amrex::MultiFab *mfz,
         amrex::ParserExecutor<3> const& xfield_parser,
         amrex::ParserExecutor<3> const& yfield_parser,
         amrex::ParserExecutor<3> const& zfield_parser,
         std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
         std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& face_areas,
         char field,
         int lev, PatchType patch_type);

    /**
     * \brief Load field values from a user-specified openPMD file,
     * for the fields Ex, Ey, Ez, Bx, By, Bz
     */
    void LoadExternalFieldsFromFile (int lev);

    /**
     * \brief Load field values from a user-specified openPMD file
     * for a specific field (specified by `F_name`)
     */
    void ReadExternalFieldFromFile (
         const std::string& read_fields_from_path, amrex::MultiFab* mf,
         const std::string& F_name, const std::string& F_component);

    /**
     * \brief
     * This function initializes and calculates grid quantities used along with
     * EBs such as edge lengths, face areas, distance to EB, etc. It also
     * appropriately communicates EB data to guard cells.
     *
     * \param[in] lev level of the Multifabs that is initialized
     */
    void InitializeEBGridData(int lev);

    /** \brief adds particle and cell contributions in cells to compute heuristic
     * cost in each box on each level, and records in `costs`
     * @param[in] costs vector of (`unique_ptr` to) vectors; expected to be initialized
     * to correct number of boxes and boxes per level
     */
    void ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > >& costs);

    void ApplyFilterandSumBoundaryRho (int lev, int glev, amrex::MultiFab& rho, int icomp, int ncomp);

    /**
     * \brief Returns an array of coefficients (Fornberg coefficients), corresponding
     * to the weight of each point in a finite-difference approximation of a derivative
     * (up to order \c n_order).
     *
     * \param[in] n_order order of the finite-difference approximation
     * \param[in] a_grid_type type of grid (collocated or not)
     */
    static amrex::Vector<amrex::Real> getFornbergStencilCoefficients(int n_order, short a_grid_type);

    // Device vectors of stencil coefficients used for finite-order centering of fields
    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_x;
    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_y;
    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_z;

    // Device vectors of stencil coefficients used for finite-order centering of currents
    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_x;
    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_y;
    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_z;

    // This needs to be public for CUDA.
    //! Tagging cells for refinement
    void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int /*ngrow*/) final;

    // Return the accelerator lattice instance defined at the given refinement level
    const AcceleratorLattice& get_accelerator_lattice (int lev) {return *(m_accelerator_lattice[lev]);}

    // for cuda
    void BuildBufferMasksInBox ( amrex::Box tbx, amrex::IArrayBox &buffer_mask,
                                 const amrex::IArrayBox &guard_mask, int ng );
#ifdef AMREX_USE_EB
    amrex::EBFArrayBoxFactory const& fieldEBFactory (int lev) const noexcept {
        return static_cast<amrex::EBFArrayBoxFactory const&>(*m_field_factory[lev]);
    }
#endif

    void InitEB ();

#ifdef AMREX_USE_EB
    /**
    * \brief Compute the length of the mesh edges. Here the length is a value in [0, 1].
    *        An edge of length 0 is fully covered.
    */
    static void ComputeEdgeLengths (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& edge_lengths,
                                    const amrex::EBFArrayBoxFactory& eb_fact);
    /**
    * \brief Compute the area of the mesh faces. Here the area is a value in [0, 1].
    *        An edge of area 0 is fully covered.
    */
    static void ComputeFaceAreas (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& face_areas,
                                  const amrex::EBFArrayBoxFactory& eb_fact);

    /**
    * \brief Scale the edges lengths by the mesh width to obtain the real lengths.
    */
    static void ScaleEdges (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& edge_lengths,
                            const std::array<amrex::Real,3>& cell_size);
    /**
    * \brief Scale the edges areas by the mesh width to obtain the real areas.
    */
    static void ScaleAreas (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& face_areas,
                            const std::array<amrex::Real,3>& cell_size);
    /**
    * \brief Initialize information for cell extensions.
    *        The flags convention for m_flag_info_face is as follows
    *          - 0 for unstable cells
    *          - 1 for stable cells which have not been intruded
    *          - 2 for stable cells which have been intruded
    *        Here we cannot know if a cell is intruded or not so we initialize all stable cells with 1
    */
    void MarkCells();
#endif

    /**
    * \brief Compute the level set function used for particle-boundary interaction.
    */
    void ComputeDistanceToEB ();
    /**
    * \brief Auxiliary function to count the amount of faces which still need to be extended
    */
    amrex::Array1D<int, 0, 2> CountExtFaces();
    /**
    * \brief Main function computing the cell extension. Where possible it computes one-way
    *       extensions and, when this is not possible, it does eight-ways extensions.
    */
    void ComputeFaceExtensions();
    /**
    * \brief Initialize the memory for the FaceInfoBoxes
    */
    void InitBorrowing();
    /**
    * \brief Shrink the vectors in the FaceInfoBoxes
    */
    void ShrinkBorrowing();
    /**
    * \brief Do the one-way extension
    */
    void ComputeOneWayExtensions();
    /**
    * \brief Do the eight-ways extension
    */
    void ComputeEightWaysExtensions();
    /**
    * \brief Whenever an unstable cell cannot be extended we increase its area to be the minimal for stability.
    *        This is the method Benkler-Chavannes-Kuster method and it is less accurate than the regular ECT but it
    *        still works better than staircasing. (see https://ieeexplore.ieee.org/document/1638381)
    *
    * @param idim Integer indicating the dimension (x->0, y->1, z->2) for which the BCK correction is done
    *
    */
    void ApplyBCKCorrection(int idim);

    /**
     * \brief Subtract the average of the cumulative sums of the preliminary current D
     *        from the current J (computed from D according to the Vay deposition scheme)
     */
    void PSATDSubtractCurrentPartialSumsAvg ();

#ifdef WARPX_USE_FFT

#   ifdef WARPX_DIM_RZ
        SpectralSolverRZ&
#   else
        SpectralSolver&
#   endif
            get_spectral_solver_fp (int lev) {return *spectral_solver_fp[lev];}
#endif

    FiniteDifferenceSolver * get_pointer_fdtd_solver_fp (int lev) { return m_fdtd_solver_fp[lev].get(); }

protected:

    /**
     * \brief
     *  This function initializes E, B, rho, and F, at all the levels
     *  of the multifab. rho and F are initialized with 0.
     *  The E and B fields are initialized using user-defined inputs.
     *  The initialization type is set using "B_ext_grid_init_style"
     *  and "E_ext_grid_init_style". The initialization style is set to "default"
     *  if not explicitly defined by the user, and the E and B fields are
     *  initialized with E_external_grid and B_external_grid, respectively, each with
     *  a default value of 0.
     *  If the initialization type for the E and B field is "constant",
     *  then, the E and B fields at all the levels are initialized with
     *  user-defined values for E_external_grid and B_external_grid.
     *  If the initialization type for B-field is set to
     *  "parse_ext_grid_function", then, the parser is used to read
     *  Bx_external_grid_function(x,y,z), By_external_grid_function(x,y,z),
     *  and Bz_external_grid_function(x,y,z).
     *  Similarly, if the E-field initialization type is set to
     *  "parse_ext_grid_function", then, the parser is used to read
     *  Ex_external_grid_function(x,y,z), Ey_external_grid_function(x,y,z),
     *  and Ex_external_grid_function(x,y,z). The parser for the E and B
     *  initialization assumes that the function has three independent
     *  variables, at max, namely, x, y, z. However, any number of constants
     *  can be used in the function used to define the E and B fields on the grid.
     */
    void InitLevelData (int lev, amrex::Real time);

    //! Use this function to override the Level 0 grids made by AMReX.
    //! This function is called in amrex::AmrCore::InitFromScratch.
    void PostProcessBaseGrids (amrex::BoxArray& ba0) const final;

    //! Make a new level from scratch using provided BoxArray and
    //! DistributionMapping.  Only used during initialization.  Called
    //! by AmrCoreInitFromScratch.
    void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& new_grids,
                                          const amrex::DistributionMapping& new_dmap) final;

    //! Make a new level using provided BoxArray and
    //! DistributionMapping and fill with interpolated coarse level
    //! data.  Called by AmrCore::regrid.
    void MakeNewLevelFromCoarse (int /*lev*/, amrex::Real /*time*/, const amrex::BoxArray& /*ba*/,
                                         const amrex::DistributionMapping& /*dm*/) final;

    //! Remake an existing level using provided BoxArray and
    //! DistributionMapping and fill with existing fine and coarse
    //! data.  Called by AmrCore::regrid.
    void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm) final;

    //! Delete level data.  Called by AmrCore::regrid.
    void ClearLevel (int lev) final;

private:

    /**
     * \brief
     *  WarpX constructor. This method should not be called directly, but rather through
     * the static member function MakeWarpX(). MakeWarpX() is called by GetInstance ()
     * if an instance of the WarpX class does not currently exist.
     */
    WarpX ();

    /**
     * \brief
     * This method creates a new instance of the WarpX class.
     */
    static void MakeWarpX ();

    // Singleton is used when the code is run from python
    static WarpX* m_instance;

    //! Complete the asynchronous broadcast of signal flags, and initiate a checkpoint if requested
    void HandleSignals ();

    void FillBoundaryB (int lev, PatchType patch_type, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryE (int lev, PatchType patch_type, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryF (int lev, PatchType patch_type, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);
    void FillBoundaryG (int lev, PatchType patch_type, amrex::IntVect ng, std::optional<bool> nodal_sync = std::nullopt);

    void FillBoundaryB_avg (int lev, PatchType patch_type, amrex::IntVect ng);
    void FillBoundaryE_avg (int lev, PatchType patch_type, amrex::IntVect ng);

    void AddExternalFields ();

    void OneStep_nosub (amrex::Real cur_time);
    void OneStep_sub1 (amrex::Real cur_time);

    /**
     * \brief Perform one PIC iteration, with the multiple J deposition per time step
     */
    void OneStep_multiJ (amrex::Real cur_time);

    void RestrictCurrentFromFineToCoarsePatch (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
        int lev);
    void AddCurrentFromFineLevelandSumBoundary (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_buffer,
        int lev);
    void StoreCurrent (int lev);
    void RestoreCurrent (int lev);
    void ApplyFilterJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& current,
        int lev,
        int idim);
    void ApplyFilterJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& current,
        int lev);
    void SumBoundaryJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& current,
        int lev,
        int idim,
        const amrex::Periodicity& period);
    void SumBoundaryJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& current,
        int lev,
        const amrex::Periodicity& period);
    void NodalSyncJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
        int lev,
        PatchType patch_type);

    void RestrictRhoFromFineToCoarsePatch (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        int lev);
    void ApplyFilterandSumBoundaryRho (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        int lev,
        PatchType patch_type,
        int icomp,
        int ncomp);
    void AddRhoFromFineLevelandSumBoundary (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_buffer,
        int lev,
        int icomp,
        int ncomp);
    void NodalSyncRho (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        int lev,
        PatchType patch_type,
        int icomp,
        int ncomp);

    void ReadParameters ();

    /** This function queries deprecated input parameters and abort
     *  the run if one of them is specified. */
    void BackwardCompatibility ();

    void InitFromScratch ();

    void AllocLevelData (int lev, const amrex::BoxArray& ba,
                         const amrex::DistributionMapping& dm);

    [[nodiscard]] amrex::DistributionMapping
    GetRestartDMap (const std::string& chkfile, const amrex::BoxArray& ba, int lev) const;

    void InitFromCheckpoint ();
    void PostRestart ();

    void InitPML ();
    void ComputePMLFactors ();

    void InitFilter ();

    void InitDiagnostics ();

    void InitNCICorrector ();

    /**
     * \brief Check that the number of guard cells is smaller than the number of valid cells,
     * for all available MultiFabs, and abort otherwise.
     */
    void CheckGuardCells();

    /**
     * \brief Checks for known numerical issues involving different WarpX modules
     */
    void CheckKnownIssues();

    /** Check the requested resources and write performance hints */
    void PerformanceHints ();

    void BuildBufferMasks ();

    [[nodiscard]] const amrex::iMultiFab* getCurrentBufferMasks (int lev) const {
        return current_buffer_masks[lev].get();
    }

    [[nodiscard]] const amrex::iMultiFab* getGatherBufferMasks (int lev) const
    {
        return gather_buffer_masks[lev].get();
    }

    /**
     * \brief Re-orders the Fornberg coefficients so that they can be used more conveniently for
     * finite-order centering operations. For example, for finite-order centering of order 6,
     * the Fornberg coefficients \c (c_0,c_1,c_2) are re-ordered as \c (c_2,c_1,c_0,c_0,c_1,c_2).
     *
     * \param[in,out] ordered_coeffs host vector where the re-ordered Fornberg coefficients will be stored
     * \param[in] unordered_coeffs host vector storing the original sequence of Fornberg coefficients
     * \param[in] order order of the finite-order centering along a given direction
     */
    void ReorderFornbergCoefficients (amrex::Vector<amrex::Real>& ordered_coeffs,
                                      amrex::Vector<amrex::Real>& unordered_coeffs,
                                      int order);
    /**
     * \brief Allocates and initializes the stencil coefficients used for the finite-order centering
     * of fields and currents, and stores them in the given device vectors.
     *
     * \param[in,out] device_centering_stencil_coeffs_x device vector where the stencil coefficients along x will be stored
     * \param[in,out] device_centering_stencil_coeffs_y device vector where the stencil coefficients along y will be stored
     * \param[in,out] device_centering_stencil_coeffs_z device vector where the stencil coefficients along z will be stored
     * \param[in] centering_nox order of the finite-order centering along x
     * \param[in] centering_noy order of the finite-order centering along y
     * \param[in] centering_noz order of the finite-order centering along z
     * \param[in] a_grid_type type of grid (collocated or not)
     */
    void AllocateCenteringCoefficients (amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_x,
                                        amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_y,
                                        amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_z,
                                        int centering_nox,
                                        int centering_noy,
                                        int centering_noz,
                                        short a_grid_type);

    void AllocLevelMFs (int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                        const amrex::IntVect& ngEB, amrex::IntVect& ngJ,
                        const amrex::IntVect& ngRho, const amrex::IntVect& ngF,
                        const amrex::IntVect& ngG, bool aux_is_nodal);

#ifdef WARPX_USE_FFT
#   ifdef WARPX_DIM_RZ
    void AllocLevelSpectralSolverRZ (amrex::Vector<std::unique_ptr<SpectralSolverRZ>>& spectral_solver,
                                     int lev,
                                     const amrex::BoxArray& realspace_ba,
                                     const amrex::DistributionMapping& dm,
                                     const std::array<amrex::Real,3>& dx);
#   else
    void AllocLevelSpectralSolver (amrex::Vector<std::unique_ptr<SpectralSolver>>& spectral_solver,
                                   int lev,
                                   const amrex::BoxArray& realspace_ba,
                                   const amrex::DistributionMapping& dm,
                                   const std::array<amrex::Real,3>& dx,
                                   bool pml_flag=false);
#   endif
#endif

    //! Author of an input file / simulation setup
    std::string m_authors;

    amrex::Vector<int> istep;      // which step?
    amrex::Vector<int> nsubsteps;  // how many substeps on each level?

    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> dt;

    // Particle container
    std::unique_ptr<MultiParticleContainer> mypc;
    std::unique_ptr<MultiDiagnostics> multi_diags;

    // Fluid container
    bool do_fluid_species = false;
    std::unique_ptr<MultiFluidContainer> myfl;

    //
    // Fields: First array for level, second for direction
    //

    // Full solution
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_aux;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_aux;

    // Fine patch
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > F_fp;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > G_fp;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > rho_fp;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > phi_fp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_fp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_fp_vay;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_fp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_fp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_avg_fp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_avg_fp;

    // Memory buffers for computing magnetostatic fields
    // Vector Potential A and previous step.  Time buffer needed for computing dA/dt to first order
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > vector_potential_fp_nodal;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > vector_potential_grad_buf_e_stag;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > vector_potential_grad_buf_b_stag;

    // Same as Bfield_fp/Efield_fp for reading external field data
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_fp_external;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_fp_external;

    //! EB: Lengths of the mesh edges
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > m_edge_lengths;
    //! EB: Areas of the mesh faces
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > m_face_areas;

    /** EB: for every mesh face flag_info_face contains a:
     *          * 0 if the face needs to be extended
     *          * 1 if the face is large enough to lend area to other faces
     *          * 2 if the face is actually intruded by other face
     * It is initialized in WarpX::MarkCells
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>, 3 > > m_flag_info_face;
    /** EB: for every mesh face face flag_ext_face contains a:
     *          * 1 if the face needs to be extended
     *          * 0 otherwise
     * It is initialized in WarpX::MarkCells and then modified in WarpX::ComputeOneWayExtensions
     * and in WarpX::ComputeEightWaysExtensions
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>, 3 > > m_flag_ext_face;
    /** EB: m_area_mod contains the modified areas of the mesh faces, i.e. if a face is enlarged it
     * contains the area of the enlarged face
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > m_area_mod;
    /** EB: m_borrowing contains the info about the enlarged cells, i.e. for every enlarged cell it
     * contains the info of which neighbors are being intruded (and the amount of borrowed area).
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::LayoutData<FaceInfoBox> >, 3 > > m_borrowing;

    /** ECTRhofield is needed only by the ect
     * solver and it contains the electromotive force density for every mesh face.
     * The name ECTRhofield has been used to comply with the notation of the paper
     * https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4463918 (page 9, equation 4
     * and below).
     * Although it's called rho it has nothing to do with the charge density!
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > ECTRhofield;
    /** Venl contains the electromotive force for every mesh face, i.e. every entry is
     * the corresponding entry in ECTRhofield multiplied by the total area (possibly with enlargement)
     * This is only used for the ECT solver.*/
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Venl;

    //EB level set
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > m_distance_to_eb;

    // store fine patch
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_store;

    // Nodal MultiFab for nodal current deposition if warpx.do_current_centering = 1
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>> current_fp_nodal;

    // Coarse patch
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > F_cp;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > G_cp;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > rho_cp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_cp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_cp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_cp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_avg_cp;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_avg_cp;

    // Copy of the coarse aux
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3 > > Efield_cax;
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_cax;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > current_buffer_masks;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > gather_buffer_masks;

    // If charge/current deposition buffers are used
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_buf;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > charge_buf;

    /**
     * \brief
     * Get a pointer to the field data. Does not check if the pointer
     * is not nullptr.
     *
     * \param field_type[in] the field type
     * \param lev[in] the mesh refinement level
     * \param direction[in] the field component (0 by default)
     *
     * \return the pointer to an amrex::MultiFab containing the field data
     */
    [[nodiscard]] amrex::MultiFab*
    getFieldPointerUnchecked (warpx::fields::FieldType field_type, int lev, int direction = 0) const;

    // PML
    int do_pml = 0;
    int do_silver_mueller = 0;
    int pml_ncell = 10;
    int pml_delta = 10;
    int pml_has_particles = 0;
    int do_pml_j_damping = 0;
    int do_pml_in_domain = 0;
    bool do_similar_dm_pml = true;
    bool do_pml_dive_cleaning; // default set in WarpX.cpp
    bool do_pml_divb_cleaning; // default set in WarpX.cpp
    amrex::Vector<amrex::IntVect> do_pml_Lo;
    amrex::Vector<amrex::IntVect> do_pml_Hi;
    amrex::Vector<std::unique_ptr<PML> > pml;
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    amrex::Vector<std::unique_ptr<PML_RZ> > pml_rz;
#endif
    amrex::Real v_particle_pml;

    // External fields parameters
    std::unique_ptr<ExternalFieldParams> m_p_ext_field_params;

    amrex::Real moving_window_x = std::numeric_limits<amrex::Real>::max();

    // Plasma injection parameters
    int warpx_do_continuous_injection = 0;
    int num_injected_species = -1;
    amrex::Vector<int> injected_plasma_species;

    std::optional<amrex::Real> m_const_dt;

    // Macroscopic properties
    std::unique_ptr<MacroscopicProperties> m_macroscopic_properties;

    // Hybrid PIC algorithm parameters
    std::unique_ptr<HybridPICModel> m_hybrid_pic_model;

    // Load balancing
    /** Load balancing intervals that reads the "load_balance_intervals" string int the input file
     * for getting steps at which load balancing is performed */
    utils::parser::IntervalsParser load_balance_intervals;
    /** Collection of LayoutData to keep track of weights used in load balancing
     * routines. Contains timer-based or heuristic-based costs depending on input option */
    amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > > costs;
    /** Load balance with 'space filling curve' strategy. */
    int load_balance_with_sfc = 0;
    /** Controls the maximum number of boxes that can be assigned to a rank during
     * load balance via the 'knapsack' strategy; e.g., if there are 4 boxes per rank,
     * `load_balance_knapsack_factor=2` limits the maximum number of boxes that can
     * be assigned to a rank to 8. */
    amrex::Real load_balance_knapsack_factor = amrex::Real(1.24);
    /** Threshold value that controls whether to adopt the proposed distribution
     * mapping during load balancing.  The new distribution mapping is adopted
     * if the ratio of proposed distribution mapping efficiency to current
     * distribution mapping efficiency is larger than the threshold; 'efficiency'
     * here means the average cost per MPI rank.  */
    amrex::Real load_balance_efficiency_ratio_threshold = amrex::Real(1.1);
    /** Current load balance efficiency for each level.  */
    amrex::Vector<amrex::Real> load_balance_efficiency;
    /** Weight factor for cells in `Heuristic` costs update.
     * Default values on GPU are determined from single-GPU tests on Summit.
     * The problem setup for these tests is an empty (i.e. no particles) domain
     * of size 256 by 256 by 256 cells, from which the average time per iteration
     * per cell is computed. */
    amrex::Real costs_heuristic_cells_wt = amrex::Real(0);
    /** Weight factor for particles in `Heuristic` costs update.
     * Default values on GPU are determined from single-GPU tests on Summit.
     * The problem setup for these tests is a high-ppc (27 particles per cell)
     * uniform plasma on a domain of size 128 by 128 by 128, from which the approximate
     * time per iteration per particle is computed. */
    amrex::Real costs_heuristic_particles_wt = amrex::Real(0);

    // Determines timesteps for override sync
    utils::parser::IntervalsParser override_sync_intervals;

    // Other runtime parameters
    int verbose = 1;

    bool use_hybrid_QED = false;

    int max_step   = std::numeric_limits<int>::max();
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();

    int regrid_int = -1;

    amrex::Real cfl = amrex::Real(0.999);

    std::string restart_chkfile;

    /** When `true`, write the diagnostics after restart at the time of the restart. */
    bool write_diagnostics_on_restart = false;

    amrex::VisMF::Header::Version plotfile_headerversion  = amrex::VisMF::Header::Version_v1;
    amrex::VisMF::Header::Version slice_plotfile_headerversion  = amrex::VisMF::Header::Version_v1;

    bool use_single_read = true;
    bool use_single_write = true;
    int mffile_nstreams = 4;
    int field_io_nfiles = 1024;
    int particle_io_nfiles = 1024;

    amrex::RealVect fine_tag_lo;
    amrex::RealVect fine_tag_hi;
    //! User-defined parser to define refinement patches
    std::unique_ptr<amrex::Parser> ref_patch_parser;

    bool is_synchronized = true;

    // Synchronization of nodal points
    static constexpr bool sync_nodal_points = true;

    guardCellManager guard_cells;

    //Slice Parameters
    int slice_max_grid_size;
    int slice_plot_int = -1;
    amrex::RealBox slice_realbox;
    amrex::IntVect slice_cr_ratio;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > F_slice;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > G_slice;
    amrex::Vector<            std::unique_ptr<amrex::MultiFab>      > rho_slice;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > current_slice;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Efield_slice;
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > Bfield_slice;

    bool fft_periodic_single_box = false;
    int nox_fft = 16;
    int noy_fft = 16;
    int noz_fft = 16;

    //! Domain decomposition on Level 0
    amrex::IntVect numprocs{0};

    //! particle buffer for scraped particles on the boundaries
    std::unique_ptr<ParticleBoundaryBuffer> m_particle_boundary_buffer;

    // Accelerator lattice elements
    amrex::Vector< std::unique_ptr<AcceleratorLattice> > m_accelerator_lattice;

    //
    // Embedded Boundary
    //

    // Factory for field data
    amrex::Vector<std::unique_ptr<amrex::FabFactory<amrex::FArrayBox> > > m_field_factory;

    [[nodiscard]]
    amrex::FabFactory<amrex::FArrayBox> const& fieldFactory (int lev) const noexcept
    {
        return *m_field_factory[lev];
    }

    /** Stop the simulation at the end of the current step due to a received Unix signal?
     */
    bool m_exit_loop_due_to_interrupt_signal = false;

    /** Stop the simulation at the end of the current step?
     */
    [[nodiscard]]
    bool checkStopSimulation (amrex::Real cur_time);

    /** Print Unused Parameter Warnings after Step 1
     *
     * Instead of waiting for a simulation to end, we already do an early "unused parameter check"
     * after step 1 to inform users early of potential issues with their simulation setup.
     */
    void checkEarlyUnusedParams ();

    /** Perform essential particle house keeping at boundaries
     *
     * Inject, communicate, scrape and sort particles.
     *
     * @param step current step
     * @param cur_time current time
     * @param num_moved number of cells the moving window moved
     */
    void HandleParticlesAtBoundaries (int step, amrex::Real cur_time, int num_moved);

    void ScrapeParticles ();

    /** Update the E and B fields in the explicit em PIC scheme.
     *
     * At the beginning, we have B^{n} and E^{n}.
     * Particles have p^{n} and x^{n}.
     */
    void ExplicitFillBoundaryEBUpdateAux ();

    void PushPSATD ();

#ifdef WARPX_USE_FFT

    /**
     * \brief Forward FFT of E,B on all mesh refinement levels
     *
     * \param E_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch electric field to be transformed
     * \param B_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch magnetic field to be transformed
     * \param E_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch electric field to be transformed
     * \param B_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch magnetic field to be transformed
     */
    void PSATDForwardTransformEB (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_cp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_cp);

    /**
     * \brief Backward FFT of E,B on all mesh refinement levels,
     *        with field damping in the guard cells (if needed)
     *
     * \param E_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch electric field to be transformed
     * \param B_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch magnetic field to be transformed
     * \param E_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch electric field to be transformed
     * \param B_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch magnetic field to be transformed
     */
    void PSATDBackwardTransformEB (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_cp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_cp);

    /**
     * \brief Backward FFT of averaged E,B on all mesh refinement levels
     *
     * \param E_avg_fp Vector of three-dimensional arrays (for each level)
     *                 storing the fine patch averaged electric field to be transformed
     * \param B_avg_fp Vector of three-dimensional arrays (for each level)
     *                 storing the fine patch averaged magnetic field to be transformed
     * \param E_avg_cp Vector of three-dimensional arrays (for each level)
     *                 storing the coarse patch averaged electric field to be transformed
     * \param B_avg_cp Vector of three-dimensional arrays (for each level)
     *                 storing the coarse patch averaged magnetic field to be transformed
     */
    void PSATDBackwardTransformEBavg (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_avg_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_avg_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& E_avg_cp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& B_avg_cp);

    /**
     * \brief Forward FFT of J on all mesh refinement levels,
     *        with k-space filtering (if needed)
     *
     * \param J_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch current to be transformed
     * \param J_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch current to be transformed
     * \param[in] apply_kspace_filter Control whether to apply filtering
     *                                (only used in RZ geometry to avoid double filtering)
     */
    void PSATDForwardTransformJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp,
        bool apply_kspace_filter=true);

    /**
     * \brief Backward FFT of J on all mesh refinement levels
     *
     * \param J_fp Vector of three-dimensional arrays (for each level)
     *             storing the fine patch current to be transformed
     * \param J_cp Vector of three-dimensional arrays (for each level)
     *             storing the coarse patch current to be transformed
     */
    void PSATDBackwardTransformJ (
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_fp,
        const amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>,3>>& J_cp);

    /**
     * \brief Forward FFT of rho on all mesh refinement levels,
     *        with k-space filtering (if needed)
     *
     * \param charge_fp Vector (for each level) storing the fine patch charge to be transformed
     * \param charge_cp Vector (for each level) storing the coarse patch charge to be transformed
     * \param[in] icomp index of fourth component (0 for rho_old, 1 for rho_new)
     * \param[in] dcomp index of spectral component (0 for rho_old, 1 for rho_new)
     * \param[in] apply_kspace_filter Control whether to apply filtering
     *                                (only used in RZ geometry to avoid double filtering)
     */
    void PSATDForwardTransformRho (
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_fp,
        const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& charge_cp,
        int icomp, int dcomp, bool apply_kspace_filter=true);

    /**
     * \brief Copy rho_new to rho_old in spectral space (when rho is linear in time)
     */
    void PSATDMoveRhoNewToRhoOld ();

    /**
     * \brief Copy J_new to J_old in spectral space (when J is linear in time)
     */
    void PSATDMoveJNewToJOld ();

    /**
     * \brief Forward FFT of F on all mesh refinement levels
     */
    void PSATDForwardTransformF ();

    /**
     * \brief Backward FFT of F on all mesh refinement levels
     */
    void PSATDBackwardTransformF ();

    /**
     * \brief Forward FFT of G on all mesh refinement levels
     */
    void PSATDForwardTransformG ();

    /**
     * \brief Backward FFT of G on all mesh refinement levels
     */
    void PSATDBackwardTransformG ();

    /**
     * \brief Correct current in Fourier space so that the continuity equation is satisfied
     */
    void PSATDCurrentCorrection ();

    /**
     * \brief Vay deposition in Fourier space (https://doi.org/10.1016/j.jcp.2013.03.010)
     */
    void PSATDVayDeposition ();

    /**
     * \brief Update all necessary fields in spectral space
     */
    void PSATDPushSpectralFields ();

    /**
     * \brief Scale averaged E,B fields to account for time integration
     *
     * \param[in] scale_factor scalar to multiply each field component by
     */
    void PSATDScaleAverageFields (amrex::Real scale_factor);

    /**
     * \brief Set averaged E,B fields to zero before new iteration
     */
    void PSATDEraseAverageFields ();

#   ifdef WARPX_DIM_RZ
        amrex::Vector<std::unique_ptr<SpectralSolverRZ>> spectral_solver_fp;
        amrex::Vector<std::unique_ptr<SpectralSolverRZ>> spectral_solver_cp;
#   else
        amrex::Vector<std::unique_ptr<SpectralSolver>> spectral_solver_fp;
        amrex::Vector<std::unique_ptr<SpectralSolver>> spectral_solver_cp;
#   endif

#endif

    amrex::Vector<std::unique_ptr<FiniteDifferenceSolver>> m_fdtd_solver_fp;
    amrex::Vector<std::unique_ptr<FiniteDifferenceSolver>> m_fdtd_solver_cp;

    // implicit solver object
    std::unique_ptr<ImplicitSolver> m_implicit_solver;

};

#endif

/* Copyright 2019-2020 Andrew Myers, Axel Huebl, David Grote
 * Ligia Diana Amorim, Luca Fedeli, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 * Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PhysicalParticleContainer_H_
#define WARPX_PhysicalParticleContainer_H_

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Initialization/PlasmaInjector.H"
#include "Particles/ElementaryProcess/Ionization.H"
#ifdef WARPX_QED
#    include "Particles/ElementaryProcess/QEDPairGeneration.H"
#    include "Particles/ElementaryProcess/QEDPhotonEmission.H"
#    include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper_fwd.H"
#    include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper_fwd.H"
#endif
#include "Particles/Gather/ScaleFields.H"
#include "Particles/Resampling/Resampling.H"
#include "WarpXParticleContainer.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <memory>
#include <string>

/**
 * PhysicalParticleContainer is the ParticleContainer class containing plasma
 * particles (if a simulation has 2 plasma species, say "electrons" and
 * "ions"), they will be two instances of PhysicalParticleContainer.
 *
 * PhysicalParticleContainer inherits from WarpXParticleContainer.
 */
class PhysicalParticleContainer
    : public WarpXParticleContainer
{
public:

    PhysicalParticleContainer (amrex::AmrCore* amr_core,
                               int ispecies,
                               const std::string& name);

    PhysicalParticleContainer (amrex::AmrCore* amr_core);

    /** This function queries deprecated input parameters and abort
     *  the run if one of them is specified. */
    void BackwardCompatibility ();

    ~PhysicalParticleContainer () override = default;

    PhysicalParticleContainer (PhysicalParticleContainer const &)              = delete;
    PhysicalParticleContainer& operator= (PhysicalParticleContainer const & )  = delete;
    PhysicalParticleContainer(PhysicalParticleContainer&& )                    = default;
    PhysicalParticleContainer& operator=(PhysicalParticleContainer&& )         = default;

    void InitData () override;

    void ReadHeader (std::istream& is) override;

    void WriteHeader (std::ostream& os) const override;

    void InitIonizationModule () override;

    /*
     * \brief Returns a pointer to the i'th plasma injector.
     */
    PlasmaInjector* GetPlasmaInjector (int i) override;

    /**
     * \brief Evolve is the central function PhysicalParticleContainer that
     * advances plasma particles for a time dt (typically one timestep).
     *
     * \param fields the WarpX field register
     * \param lev level on which particles are living
     * \param current_fp_string current coarse or fine patch identifier in fields
     * \param t current physical time
     * \param dt time step by which particles are advanced
     * \param a_dt_type type of time step (used for sub-cycling)
     * \param skip_deposition Skip the charge and current deposition.
     * \param push_type Type of particle push, explicit or implicit. Defaults to explicit
     *
     * Evolve iterates over particle iterator (each box) and performs filtering,
     * field gather, particle push and current deposition for all particles
     * in the box.
     */
    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real t,
                 amrex::Real dt,
                 DtType a_dt_type=DtType::Full,
                 bool skip_deposition=false,
                 PushType push_type=PushType::Explicit) override;

    virtual void PushPX (WarpXParIter& pti,
                         amrex::FArrayBox const * exfab,
                         amrex::FArrayBox const * eyfab,
                         amrex::FArrayBox const * ezfab,
                         amrex::FArrayBox const * bxfab,
                         amrex::FArrayBox const * byfab,
                         amrex::FArrayBox const * bzfab,
                         amrex::IntVect ngEB, int /*e_is_nodal*/,
                         long offset,
                         long np_to_push,
                         int lev, int gather_lev,
                         amrex::Real dt, ScaleFields scaleFields,
                         DtType a_dt_type=DtType::Full);

    void ImplicitPushXP (WarpXParIter& pti,
                         amrex::FArrayBox const * exfab,
                         amrex::FArrayBox const * eyfab,
                         amrex::FArrayBox const * ezfab,
                         amrex::FArrayBox const * bxfab,
                         amrex::FArrayBox const * byfab,
                         amrex::FArrayBox const * bzfab,
                         amrex::IntVect ngEB, int /*e_is_nodal*/,
                         long offset,
                         long np_to_push,
                         int lev, int gather_lev,
                         amrex::Real dt, ScaleFields scaleFields,
                         DtType a_dt_type=DtType::Full);

    void PushP (int lev, amrex::Real dt,
                        const amrex::MultiFab& Ex,
                        const amrex::MultiFab& Ey,
                        const amrex::MultiFab& Ez,
                        const amrex::MultiFab& Bx,
                        const amrex::MultiFab& By,
                        const amrex::MultiFab& Bz) override;

    void PartitionParticlesInBuffers (
                        long& nfine_current,
                        long& nfine_gather,
                        long np,
                        WarpXParIter& pti,
                        int lev,
                        amrex::iMultiFab const* current_masks,
                        amrex::iMultiFab const* gather_masks );

    void PostRestart () final {}

    void SplitParticles (int lev);

    IonizationFilterFunc getIonizationFunc (const WarpXParIter& pti,
                                            int lev,
                                            amrex::IntVect ngEB,
                                            const amrex::FArrayBox& Ex,
                                            const amrex::FArrayBox& Ey,
                                            const amrex::FArrayBox& Ez,
                                            const amrex::FArrayBox& Bx,
                                            const amrex::FArrayBox& By,
                                            const amrex::FArrayBox& Bz);

    // Inject particles in Box 'part_box'
    virtual void AddParticles (int lev);

    /**
     * Create new macroparticles for this species, with a fixed
     * number of particles per cell (in the cells of `part_realbox`).
     * The new particles are only created inside the intersection of `part_realbox`
     * with the local grid for the current proc.
     * @param[in] the PlasmaInjector instance holding the input parameters
     * @param[in] lev the index of the refinement level
     * @param[in] part_realbox the box in which new particles should be created
     * (this box should correspond to an integer number of cells in each direction,
     * but its boundaries need not be aligned with the actual cells of the simulation)
    */
    void AddPlasma (PlasmaInjector const& plasma_injector, int lev, amrex::RealBox part_realbox = amrex::RealBox());

    /**
     * Create new macroparticles for this species, with a fixed
     * number of particles per cell in a plane.
     * @param[in] the PlasmaInjector instance holding the input parameters
     * @param[in] dt time step size, used to partially advance the particles
    */
    void AddPlasmaFlux (PlasmaInjector const& plasma_injector, amrex::Real dt);

    void MapParticletoBoostedFrame (amrex::ParticleReal& x, amrex::ParticleReal& y, amrex::ParticleReal& z,
                                    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
                                    amrex::Real t_lab = 0.) const;

    void AddGaussianBeam (PlasmaInjector const& plasma_injector);

    /** Load a particle beam from an external file
     * @param[in] the PlasmaInjector instance holding the input parameters
     * @param[in] q_tot total charge of the particle species to be initialized
     * @param[in] z_shift optional shift to the z position of particles (useful for boosted frame runs)
     */
    void AddPlasmaFromFile (PlasmaInjector & plasma_injector,
                            amrex::ParticleReal q_tot,
                            amrex::ParticleReal z_shift);

    void CheckAndAddParticle (
        amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z,
        amrex::ParticleReal ux, amrex::ParticleReal uy, amrex::ParticleReal uz,
        amrex::ParticleReal weight,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_x,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_y,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_z,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_ux,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_uy,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_uz,
        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_w,
        amrex::Real t_lab= 0.) const;

    /**
     * \brief Default initialize runtime attributes in a tile. This routine does not initialize the
     * first n_external_attr_real real attributes and the first n_external_attr_int integer
     * attributes, which have been in principle externally set elsewhere.
     *
     * @param[inout] pinned_tile the tile in which attributes are initialized
     * @param[in] n_external_attr_real The number of real attributes that have been externally set.
     * These are NOT initialized by this function.
     * @param[in] n_external_attr_int The number of integer attributes that have been externally set.
     * These are NOT initialized by this function.
     * @param[in] engine the random engine, used in initialization of QED optical depths
     */
    void DefaultInitializeRuntimeAttributes (
        typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType& pinned_tile,
        int n_external_attr_real,
        int n_external_attr_int) final;

/**
 * \brief Apply NCI Godfrey filter to all components of E and B before gather
 * \param lev MR level
 * \param box box onto which the filter is applied
 * \param exeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field Ex
 * \param eyeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field Ey
 * \param ezeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field Ez
 * \param bxeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field Bx
 * \param byeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field By
 * \param bzeli safeguard Elixir object (to avoid de-allocating too early
            --between ParIter iterations-- on GPU) for field Bz
 * \param filtered_Ex Array containing filtered value
 * \param filtered_Ey Array containing filtered value
 * \param filtered_Ez Array containing filtered value
 * \param filtered_Bx Array containing filtered value
 * \param filtered_By Array containing filtered value
 * \param filtered_Bz Array containing filtered value
 * \param Ex Field array before filtering (not modified)
 * \param Ey Field array before filtering (not modified)
 * \param Ez Field array before filtering (not modified)
 * \param Bx Field array before filtering (not modified)
 * \param By Field array before filtering (not modified)
 * \param Bz Field array before filtering (not modified)
 * \param ex_ptr pointer to the Ex field (modified)
 * \param ey_ptr pointer to the Ey field (modified)
 * \param ez_ptr pointer to the Ez field (modified)
 * \param bx_ptr pointer to the Bx field (modified)
 * \param by_ptr pointer to the By field (modified)
 * \param bz_ptr pointer to the Bz field (modified)
 *
 * The NCI Godfrey filter is applied on Ex, the result is stored in filtered_Ex
 * and the pointer exfab is modified (before this function is called, it points to Ex
 * and after this function is called, it points to Ex_filtered)
 */
    void applyNCIFilter (
        int lev, const amrex::Box& box,
        amrex::Elixir& exeli, amrex::Elixir& eyeli, amrex::Elixir& ezeli,
        amrex::Elixir& bxeli, amrex::Elixir& byeli, amrex::Elixir& bzeli,
        amrex::FArrayBox& filtered_Ex, amrex::FArrayBox& filtered_Ey,
        amrex::FArrayBox& filtered_Ez, amrex::FArrayBox& filtered_Bx,
        amrex::FArrayBox& filtered_By, amrex::FArrayBox& filtered_Bz,
        const amrex::FArrayBox& Ex, const amrex::FArrayBox& Ey,
        const amrex::FArrayBox& Ez, const amrex::FArrayBox& Bx,
        const amrex::FArrayBox& By, const amrex::FArrayBox& Bz,
        amrex::FArrayBox const * & ex_ptr, amrex::FArrayBox const * & ey_ptr,
        amrex::FArrayBox const * & ez_ptr, amrex::FArrayBox const * & bx_ptr,
        amrex::FArrayBox const * & by_ptr, amrex::FArrayBox const * & bz_ptr);

    /**
    * \brief This function determines if resampling should be done for the current species, and
    * if so, performs the resampling.
    *
    * @param[in] timestep the current timestep.
    */
    void resample (int timestep, bool verbose=true) final;

#ifdef WARPX_QED
    //Functions decleared in WarpXParticleContainer.H
    //containers for which QED processes could be relevant
    //are expected to override these functions

    /**
     * Tells if this PhysicalParticleContainer has Quantum
     * Synchrotron process enabled
     * @return true if process is enabled
     */
    bool has_quantum_sync () const override;

    /**
     * Tells if this PhysicalParticleContainer has Breit
     * Wheeler process enabled
     * @return true if process is enabled
     */
    bool has_breit_wheeler () const override;

    /**
     * Acquires a shared smart pointer to a BreitWheelerEngine
     * @param[in] ptr the pointer
     */
    void set_breit_wheeler_engine_ptr
        (const std::shared_ptr<BreitWheelerEngine>& ptr) override;

    /**
     * Acquires a shared smart pointer to a QuantumSynchrotronEngine
     * @param[in] ptr the pointer
     */
    void set_quantum_sync_engine_ptr
        (const std::shared_ptr<QuantumSynchrotronEngine>& ptr) override;
    //__________

    BreitWheelerEngine* get_breit_wheeler_engine_ptr () const override {
        return m_shr_p_bw_engine.get();
    }

    QuantumSynchrotronEngine* get_quantum_sync_engine_ptr () const override {
        return m_shr_p_qs_engine.get();
    }

    PhotonEmissionFilterFunc getPhotonEmissionFilterFunc ();

    PairGenerationFilterFunc getPairGenerationFilterFunc ();
#endif

    std::vector<std::string> getUserIntAttribs () const override {
        return m_user_int_attribs;
    }

    std::vector<std::string> getUserRealAttribs () const override {
        return m_user_real_attribs;
    }

    amrex::Vector< amrex::Parser* > getUserIntAttribParser () const override {
        return GetVecOfPtrs(m_user_int_attrib_parser);
    }

    amrex::Vector< amrex::Parser* > getUserRealAttribParser () const override {
        return GetVecOfPtrs(m_user_real_attrib_parser);
    }

protected:

    /*
       Finds the box defining the region where refine injection should be used, if that
       option is enabled. Currently this only works for numLevels() == 2 and static mesh
       refinement.
    */
    bool findRefinedInjectionBox (amrex::Box& fine_injection_box, amrex::IntVect& rrfac);

    std::string species_name;
    std::vector<std::unique_ptr<PlasmaInjector>> plasma_injectors;

    // When true, adjust the transverse particle positions accounting
    // for the difference between the Lorentz transformed time of the
    // particle and the time of the boosted frame.
    bool boost_adjust_transverse_positions = false;
    bool do_backward_propagation = false;
    bool m_rz_random_theta = true;

    // Impose t_lab from the openPMD file for externally loaded species
    bool impose_t_lab_from_file = false;

    Resampling m_resampler;

    // Inject particles during the whole simulation
    void ContinuousInjection (const amrex::RealBox& injection_box) override;

    // Continuously inject a flux of particles from a defined surface
    void ContinuousFluxInjection (amrex::Real t, amrex::Real dt) override;

    //This function return true if the PhysicalParticleContainer contains electrons
    //or positrons, false otherwise

    //When true PhysicalParticleContainer tries to use a pusher including
    //radiation reaction
    bool do_classical_radiation_reaction = false;

    // A flag to enable saving of the previous timestep positions
    bool m_save_previous_position = false;

#ifdef WARPX_QED
    // A flag to enable quantum_synchrotron process for leptons
    bool m_do_qed_quantum_sync = false;

    // A flag to enable breit_wheeler process [photons only!!]
    bool m_do_qed_breit_wheeler = false;

    // A smart pointer to an instance of a Quantum Synchrotron engine
    std::shared_ptr<QuantumSynchrotronEngine> m_shr_p_qs_engine;

    // A smart pointer to an instance of a Breit Wheeler engine [photons only!]
    std::shared_ptr<BreitWheelerEngine> m_shr_p_bw_engine;
#endif
    /* Vector of user-defined integer attributes for species, species_name */
    std::vector<std::string> m_user_int_attribs;
    /* Vector of user-defined real attributes for species, species_name */
    std::vector<std::string> m_user_real_attribs;
    /* Vector of user-defined parser for initializing user-defined integer attributes */
    amrex::Vector< std::unique_ptr<amrex::Parser> > m_user_int_attrib_parser;
    /* Vector of user-defined parser for initializing user-defined real attributes */
    amrex::Vector< std::unique_ptr<amrex::Parser> > m_user_real_attrib_parser;

};

#endif

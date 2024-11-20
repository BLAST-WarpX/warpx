/* Copyright 2019-2020 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Jean-Luc Vay, Junmin Gu
 * Luca Fedeli, Mathieu Lobet, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 * Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ParticleContainer_H_
#define WARPX_ParticleContainer_H_

#include "MultiParticleContainer_fwd.H"

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Particles/Collision/CollisionHandler.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper_fwd.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper_fwd.H"
#endif
#include "PhysicalParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpXParticleContainer.H"
#include "ParticleBoundaries.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_GpuControl.H>
#include <AMReX_INT.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <algorithm>
#include <array>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

/**
 * The class MultiParticleContainer holds multiple instances of the polymorphic
 * class WarpXParticleContainer, stored in its member variable "allcontainers".
 * The class WarpX typically has a single (pointer to an) instance of
 * MultiParticleContainer.
 *
 * MultiParticleContainer typically has two types of functions:
 * - Functions that loop over all instances of WarpXParticleContainer in
 *   allcontainers and calls the corresponding function (for instance,
 *   MultiParticleContainer::Evolve loops over all particles containers and
 *   calls the corresponding WarpXParticleContainer::Evolve function).
 * - Functions that specifically handle multiple species (for instance
 *   ReadParameters or mapSpeciesProduct).
 */
class MultiParticleContainer
{

public:

    MultiParticleContainer (amrex::AmrCore* amr_core);

    ~MultiParticleContainer() = default;

    MultiParticleContainer (MultiParticleContainer const &)             = delete;
    MultiParticleContainer& operator= (MultiParticleContainer const & ) = delete;
    MultiParticleContainer(MultiParticleContainer&& )                   = default;
    MultiParticleContainer& operator=(MultiParticleContainer&& )        = default;

    [[nodiscard]] WarpXParticleContainer&
    GetParticleContainer (int index) const {return *allcontainers[index];}

    [[nodiscard]] WarpXParticleContainer*
    GetParticleContainerPtr (int index) const {return allcontainers[index].get();}

    [[nodiscard]] WarpXParticleContainer&
    GetParticleContainerFromName (const std::string& name) const;

    std::array<amrex::ParticleReal, 3> meanParticleVelocity(int index) {
        return allcontainers[index]->meanParticleVelocity();
    }

    amrex::ParticleReal maxParticleVelocity();

    void AllocData ();

    void InitData ();

    void InitMultiPhysicsModules ();

    /**
    * \brief This evolves all the particles by one PIC time step, including current deposition, the
    * field solve, and pushing the particles, for all the species in the MultiParticleContainer.
    * This is the electromagnetic version.
    */
    void Evolve (
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        std::string const& current_fp_string,
        amrex::Real t,
        amrex::Real dt,
        DtType a_dt_type=DtType::Full,
        bool skip_deposition=false,
        PushType push_type=PushType::Explicit
    );

    /**
    * \brief This pushes the particle positions by one time step for all the species in the
    * MultiParticleContainer.
    */
    void PushX (amrex::Real dt);

    /**
    * This pushes the particle momenta by dt for all the species in the
    * MultiParticleContainer. It is used to desynchronize the particles after initialization
    * or when restarting from a checkpoint.  It is also used to synchronize particles at the
    * the end of the run.  This is the electromagnetic version.
    */
    void PushP (int lev, amrex::Real dt,
                const amrex::MultiFab& Ex, const amrex::MultiFab& Ey, const amrex::MultiFab& Ez,
                const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bz);

    /**
    * \brief This returns a MultiFAB filled with zeros. It is used to return the charge density
    * when there is no particle species.
    *
    * @param[in] lev the index of the refinement level.
    */
    std::unique_ptr<amrex::MultiFab> GetZeroChargeDensity(int lev);

    /**
     * \brief Deposit charge density.
     *
     * \param[in,out] rho vector of charge densities (one pointer to MultiFab per mesh refinement level)
     * \param[in] relative_time Time at which to deposit rho, relative to the time of the
     *                          current positions of the particles. When different than 0,
     *                          the particle position will be temporarily modified to match
     *                          the time of the deposition.
     */
    void
    DepositCharge (const ablastr::fields::MultiLevelScalarField& rho,
                   amrex::Real relative_time);

    /**
     * \brief Deposit current density.
     *
     * \param[in,out] J vector of current densities (one three-dimensional array of pointers
     *                to MultiFabs per mesh refinement level)
     * \param[in] dt Time step for particle level
     * \param[in] relative_time Time at which to deposit J, relative to the time of the
     *                          current positions of the particles. When different than 0,
     *                          the particle position will be temporarily modified to match
     *                          the time of the deposition.
     */
    void
    DepositCurrent (ablastr::fields::MultiLevelVectorField const& J,
                    amrex::Real dt, amrex::Real relative_time);

    ///
    /// This deposits the particle charge onto a node-centered MultiFab and returns a unique ptr
    /// to it. The charge density is accumulated over all the particles in the MultiParticleContainer

    std::unique_ptr<amrex::MultiFab> GetChargeDensity(int lev, bool local = false);

    void doFieldIonization (int lev,
                            const amrex::MultiFab& Ex, const amrex::MultiFab& Ey, const amrex::MultiFab& Ez,
                            const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bz);

    void doCollisions (amrex::Real cur_time, amrex::Real dt);

    /**
    * \brief This function loops over all species and performs resampling if appropriate.
    *
    * @param[in] timestep the current timestep.
    */
    void doResampling (int timestep, bool verbose);

#ifdef WARPX_QED
    /** If Schwinger process is activated, this function is called at every
     * timestep in Evolve and is used to create Schwinger electron-positron pairs.
     * Within this function we loop over all cells to calculate the number of
     * created physical pairs. If this number is higher than 0, we create a single
     * particle per species in this cell, with a weight corresponding to the number of physical
     * particles.
     */
    void doQEDSchwinger ();

    /** This function computes the box outside which Schwinger process is disabled. The box is
     * defined by m_qed_schwinger_xmin/xmax/ymin/ymax/zmin/zmax and the warpx level 0 geometry
     * object (to make the link between Real and int quantities).
     */
    [[nodiscard]] amrex::Box ComputeSchwingerGlobalBox () const;
#endif

    void Restart (const std::string& dir);

    void PostRestart ();

    void ReadHeader (std::istream& is);

    void WriteHeader (std::ostream& os) const;

    void SortParticlesByBin (amrex::IntVect bin_size);

    void Redistribute ();

    void defineAllParticleTiles ();

    void deleteInvalidParticles ();

    void RedistributeLocal (int num_ghost);

    /** Apply BC. For now, just discard particles outside the domain, regardless
     *  of the whole simulation BC. */
    void ApplyBoundaryConditions ();

    /**
    * \brief This returns a vector filled with zeros whose size is the number of boxes in the
    * simulation boxarray. It is used to return the number of particles in each grid when there is
    * no particle species.
    *
    * @param[in] lev the index of the refinement level.
    */
    [[nodiscard]] amrex::Vector<amrex::Long> GetZeroParticlesInGrid(int lev) const;

    [[nodiscard]] amrex::Vector<amrex::Long> NumberOfParticlesInGrid(int lev) const;

    void Increment (amrex::MultiFab& mf, int lev);

    void SetParticleBoxArray (int lev, amrex::BoxArray& new_ba);
    void SetParticleDistributionMap (int lev, amrex::DistributionMapping& new_dm);

    [[nodiscard]] int nSpecies () const {return static_cast<int>(species_names.size());}
    [[nodiscard]] int nLasers () const {return static_cast<int>(lasers_names.size());}
    [[nodiscard]] int nContainers () const {return static_cast<int>(allcontainers.size());}

    /** Whether back-transformed diagnostics need to be performed for any plasma species.
     *
     * \param[in] do_back_transformed_particles The parameter to set if back-transformed particles are set to true/false
     */
    void SetDoBackTransformedParticles (bool do_back_transformed_particles);
    /** Whether back-transformed diagnostics is set for species with species_name.
     *
     * \param[in] species_name The species for which back-transformed particles is set.
     * \param[in] do_back_transformed_particles The parameter to set if back-transformed particles are set to true/false
     */
    void SetDoBackTransformedParticles (const std::string& species_name, bool do_back_transformed_particles);

    [[nodiscard]] int nSpeciesDepositOnMainGrid () const
    {
        bool const onMainGrid = true;
        auto const & v = m_deposit_on_main_grid;
        return static_cast<int>(std::count( v.begin(), v.end(), onMainGrid ));
    }

    [[nodiscard]] int nSpeciesGatherFromMainGrid() const
    {
        bool const fromMainGrid = true;
        auto const & v = m_gather_from_main_grid;
        return static_cast<int>(std::count( v.begin(), v.end(), fromMainGrid ));
    }

    // Inject particles during the simulation (for particles entering the
    // simulation domain after some iterations, due to flowing plasma and/or
    // moving window).
    void ContinuousInjection(const amrex::RealBox& injection_box) const;

    /**
     * \brief  Update antenna position for continuous injection of lasers
     *         in a boosted frame. Empty function for containers other than lasers.
     */
    void UpdateAntennaPosition(amrex::Real dt) const;

    [[nodiscard]] int doContinuousInjection() const;

    // Inject particles from a surface during the simulation
    void ContinuousFluxInjection(amrex::Real t, amrex::Real dt) const;

    [[nodiscard]] std::vector<std::string> GetSpeciesNames() const { return species_names; }

    [[nodiscard]] std::vector<std::string> GetLasersNames() const { return lasers_names; }

    [[nodiscard]] std::vector<std::string> GetSpeciesAndLasersNames() const
    {
        std::vector<std::string> tmp = species_names;
        tmp.insert(tmp.end(), lasers_names.begin(), lasers_names.end());
        return tmp;
    }

    PhysicalParticleContainer& GetPCtmp () { return *pc_tmp; }

    void ScrapeParticlesAtEB (ablastr::fields::MultiLevelScalarField const& distance_to_eb);

    std::string m_B_ext_particle_s = "none";
    std::string m_E_ext_particle_s = "none";
    // Parser for B_external on the particle
    std::unique_ptr<amrex::Parser> m_Bx_particle_parser;
    std::unique_ptr<amrex::Parser> m_By_particle_parser;
    std::unique_ptr<amrex::Parser> m_Bz_particle_parser;
    // Parser for E_external on the particle
    std::unique_ptr<amrex::Parser> m_Ex_particle_parser;
    std::unique_ptr<amrex::Parser> m_Ey_particle_parser;
    std::unique_ptr<amrex::Parser> m_Ez_particle_parser;

    amrex::ParticleReal m_repeated_plasma_lens_period;
    amrex::Vector<amrex::ParticleReal> h_repeated_plasma_lens_starts;
    amrex::Vector<amrex::ParticleReal> h_repeated_plasma_lens_lengths;
    amrex::Vector<amrex::ParticleReal> h_repeated_plasma_lens_strengths_E;
    amrex::Vector<amrex::ParticleReal> h_repeated_plasma_lens_strengths_B;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_starts;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_lengths;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_strengths_E;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_strengths_B;

#ifdef WARPX_QED
    /**
    * \brief Performs QED events (Breit-Wheeler process and photon emission)
    */
    void doQedEvents (int lev,
                      const amrex::MultiFab& Ex,
                      const amrex::MultiFab& Ey,
                      const amrex::MultiFab& Ez,
                      const amrex::MultiFab& Bx,
                      const amrex::MultiFab& By,
                      const amrex::MultiFab& Bz);
#endif

    [[nodiscard]] int getSpeciesID (const std::string& product_str) const;

    amrex::Vector<std::unique_ptr<WarpXParticleContainer>>::iterator begin() {return allcontainers.begin();}
    amrex::Vector<std::unique_ptr<WarpXParticleContainer>>::iterator end() {return allcontainers.end();}

protected:

#ifdef WARPX_QED
    /**
    * \brief Performs Breit-Wheeler process for the species for which it is enabled
    */
    void doQedBreitWheeler (int lev,
                            const amrex::MultiFab& Ex,
                            const amrex::MultiFab& Ey,
                            const amrex::MultiFab& Ez,
                            const amrex::MultiFab& Bx,
                            const amrex::MultiFab& By,
                            const amrex::MultiFab& Bz);

    /**
    * \brief Performs QED photon emission for the species for which it is enabled
    */
    void doQedQuantumSync (int lev,
                           const amrex::MultiFab& Ex,
                           const amrex::MultiFab& Ey,
                           const amrex::MultiFab& Ez,
                           const amrex::MultiFab& Bx,
                           const amrex::MultiFab& By,
                           const amrex::MultiFab& Bz);
#endif

    // Particle container types
    enum struct PCTypes {Physical, RigidInjected, Photon};

    std::vector<std::string> species_names;

    std::vector<std::string> lasers_names;

    std::unique_ptr<CollisionHandler> collisionhandler;

    //! instead of depositing (current, charge) on the finest patch level, deposit to the coarsest grid
    std::vector<bool> m_deposit_on_main_grid;
    std::vector<bool> m_laser_deposit_on_main_grid;

    //! instead of gathering fields from the finest patch level, gather from the coarsest
    std::vector<bool> m_gather_from_main_grid;

    std::vector<PCTypes> species_types;

    template<typename ...Args>
    [[nodiscard]]
    amrex::MFItInfo getMFItInfo (const WarpXParticleContainer& pc_src,
                                 Args const&... pc_dsts) const noexcept
    {
        amrex::MFItInfo info;

        MFItInfoCheckTiling(pc_src, pc_dsts...);

        if (WarpXParticleContainer::do_tiling && amrex::Gpu::notInLaunchRegion()) {
            info.EnableTiling(WarpXParticleContainer::tile_size);
        }

#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#endif

        return info;
    }


#ifdef WARPX_QED
    // The QED engines
    std::shared_ptr<BreitWheelerEngine> m_shr_p_bw_engine;
    std::shared_ptr<QuantumSynchrotronEngine> m_shr_p_qs_engine;
    //_______________________________

    /**
     * Initialize QED engines and provides smart pointers
     * to species who need QED processes
     */
    void InitQED ();

    //Variables to store how many species need a QED process
    int m_nspecies_quantum_sync = 0;
    int m_nspecies_breit_wheeler = 0;
    //________

    static constexpr auto m_default_quantum_sync_photon_creation_energy_threshold =
        static_cast<amrex::ParticleReal>(
        2.0 * PhysConst::m_e * PhysConst::c * PhysConst::c );  /*!< Default value of the energy threshold for photon creation in Quantum Synchrotron process.*/


    amrex::ParticleReal m_quantum_sync_photon_creation_energy_threshold =
        m_default_quantum_sync_photon_creation_energy_threshold; /*!< Energy threshold for photon creation in Quantum Synchrotron process.*/

    /**
     * Returns the number of species having Quantum Synchrotron process enabled
     */
    [[nodiscard]] int NSpeciesQuantumSync() const { return  m_nspecies_quantum_sync;}

    /**
     * Returns the number of species having Breit Wheeler process enabled
     */
    [[nodiscard]] int NSpeciesBreitWheeler() const { return m_nspecies_breit_wheeler;}

    /**
     * Initializes the Quantum Synchrotron engine
     */
    void InitQuantumSync ();

    /**
     * Initializes the Quantum Synchrotron engine
     */
    void InitBreitWheeler ();

    /**
     * Called by InitQuantumSync if a new table has
     * to be generated.
     */
    void QuantumSyncGenerateTable();

    /**
     * Called by InitBreitWheeler if a new table has
     * to be generated.
     */
    void BreitWheelerGenerateTable();

    /** Whether or not to activate Schwinger process */
    bool m_do_qed_schwinger = false;
    /** Name of Schwinger electron product species */
    std::string m_qed_schwinger_ele_product_name;
    /** Name of Schwinger positron product species */
    std::string m_qed_schwinger_pos_product_name;
    /** Index for Schwinger electron product species in allcontainers */
    int m_qed_schwinger_ele_product;
    /** Index for Schwinger positron product species in allcontainers */
    int m_qed_schwinger_pos_product;
    /** Transverse size used in 2D Schwinger pair production rate calculations */
    amrex::Real m_qed_schwinger_y_size;
    /** If the number of physical Schwinger pairs created within a cell is
     * higher than this threshold we use a Gaussian distribution rather than
     * a Poisson distribution for the pair production rate calculations
     */
    int m_qed_schwinger_threshold_poisson_gaussian = 25;
    /** The 6 following variables are spatial boundaries beyond which Schwinger process is
     *  deactivated
     */
    amrex::Real m_qed_schwinger_xmin = std::numeric_limits<amrex::Real>::lowest();
    amrex::Real m_qed_schwinger_xmax = std::numeric_limits<amrex::Real>::max();
    amrex::Real m_qed_schwinger_ymin = std::numeric_limits<amrex::Real>::lowest();
    amrex::Real m_qed_schwinger_ymax = std::numeric_limits<amrex::Real>::max();
    amrex::Real m_qed_schwinger_zmin = std::numeric_limits<amrex::Real>::lowest();
    amrex::Real m_qed_schwinger_zmax = std::numeric_limits<amrex::Real>::max();

#endif

private:

    // physical particles (+ laser)
    amrex::Vector<std::unique_ptr<WarpXParticleContainer>> allcontainers;
    // Temporary particle container, used e.g. for particle splitting.
    std::unique_ptr<PhysicalParticleContainer> pc_tmp;

    void ReadParameters ();

    void mapSpeciesProduct ();

    bool m_do_back_transformed_particles = false;

    void MFItInfoCheckTiling(const WarpXParticleContainer& /*pc_src*/) const noexcept
    {}

    template<typename First, typename ...Args>
    void MFItInfoCheckTiling(const WarpXParticleContainer& pc_src,
                             First const& pc_dst, Args const&... others) const noexcept
    {
        if (WarpXParticleContainer::do_tiling && amrex::Gpu::notInLaunchRegion()) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pc_dst.do_tiling,
                "For particle creation processes, either all or none of the "
                "particle species must use tiling.");
        }

        MFItInfoCheckTiling(pc_src, others...);
    }

    /**
    * Should be called right after mapSpeciesProduct in InitData.
    * It checks the physical correctness of product particle species
    * selected by the user for ionization process.
    */
    void CheckIonizationProductSpecies();

#ifdef WARPX_QED
    /**
    * Should be called right after mapSpeciesProduct in InitData.
    * It checks the physical correctness of product particle species
    * selected by the user for QED processes.
    */
    void CheckQEDProductSpecies();
#endif


};
#endif /*WARPX_ParticleContainer_H_*/

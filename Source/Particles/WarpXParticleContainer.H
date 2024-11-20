/* Copyright 2019-2020 Andrew Myers, Axel Huebl, David Grote
 * Jean-Luc Vay, Junmin Gu, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 * Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_WarpXParticleContainer_H_
#define WARPX_WarpXParticleContainer_H_

#include "WarpXParticleContainer_fwd.H"

#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Initialization/PlasmaInjector.H"
#include "Particles/ParticleBoundaries.H"
#include "SpeciesPhysicalProperties.H"

#ifdef WARPX_QED
#    include "ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper_fwd.H"
#    include "ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper_fwd.H"
#endif
#include "MultiParticleContainer_fwd.H"
#include "NamedComponentParticleContainer.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_INT.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <array>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <utility>


class WarpXParIter
    : public amrex::ParIterSoA<PIdx::nattribs, 0>
{
public:
    using amrex::ParIterSoA<PIdx::nattribs, 0>::ParIterSoA;

    WarpXParIter (ContainerType& pc, int level);

    WarpXParIter (ContainerType& pc, int level, amrex::MFItInfo& info);

    [[nodiscard]] const std::array<RealVector, PIdx::nattribs>& GetAttribs () const
    {
        return GetStructOfArrays().GetRealData();
    }

    [[nodiscard]] std::array<RealVector, PIdx::nattribs>& GetAttribs ()
    {
        return GetStructOfArrays().GetRealData();
    }

    [[nodiscard]] const RealVector& GetAttribs (int comp) const
    {
        return GetStructOfArrays().GetRealData(comp);
    }

    [[nodiscard]] RealVector& GetAttribs (int comp)
    {
        return GetStructOfArrays().GetRealData(comp);
    }

    [[nodiscard]] IntVector& GetiAttribs (int comp)
    {
        return GetStructOfArrays().GetIntData(comp);
    }
};

/**
 * WarpXParticleContainer is the base polymorphic class from which all concrete
 * particle container classes (that store a collection of particles) derive. Derived
 * classes can be used for plasma particles, photon particles, or non-physical
 * particles (e.g., for the laser antenna).
 * It derives from amrex::ParticleContainerPureSoA<PIdx::nattribs>, where the
 * template arguments stand for the number of int and amrex::Real SoA
 * data in amrex::SoAParticle.
 *  - SoA amrex::Real: positions x, y, z, momentum ux, uy, uz, ... see PIdx for details;
 *                     more can be added at runtime
 *  - SoA int: 0 attributes by default, but can be added at runtime
 *  - SoA uint64_t: idcpu, a global 64bit index, with a 40bit local id and a 24bit cpu id
 *    (both set at creation)
 * the list.
 *
 * WarpXParticleContainer contains the main functions for initialization,
 * interaction with the grid (field gather and current deposition) and particle
 * push.
 *
 * Note: many functions are pure virtual (meaning they MUST be defined in
 * derived classes, e.g., Evolve) or actual functions (e.g. CurrentDeposition).
 */
class WarpXParticleContainer
    : public NamedComponentParticleContainer<amrex::DefaultAllocator>
{
public:
    friend MultiParticleContainer;

    // amrex::StructOfArrays with DiagIdx::nattribs amrex::ParticleReal components
    // and 0 int components for the particle data.
    using DiagnosticParticleData = amrex::StructOfArrays<DiagIdx::nattribs, 0>;
    // DiagnosticParticles is a vector, with one element per MR level.
    // DiagnosticParticles[lev] is typically a key-value pair where the key is
    // a pair [grid_index, tile_index], and the value is the corresponding
    // DiagnosticParticleData (see above) on this tile.
    using DiagnosticParticles = amrex::Vector<std::map<std::pair<int, int>, DiagnosticParticleData> >;

    WarpXParticleContainer (amrex::AmrCore* amr_core, int ispecies);
    ~WarpXParticleContainer() override = default;

    // Move and copy operations
    WarpXParticleContainer(const WarpXParticleContainer&)            = delete;
    WarpXParticleContainer& operator=(const WarpXParticleContainer&) = delete;
    WarpXParticleContainer(WarpXParticleContainer&&)                 = default;
    WarpXParticleContainer& operator=(WarpXParticleContainer&&)      = default;

    virtual void InitData () = 0;

    virtual void InitIonizationModule () {}

    /*
     * \brief Virtual function that returns a pointer to the plasma injector,
     *        for derived classes that define one (PhysicalParticleContainer).
     */
    virtual PlasmaInjector* GetPlasmaInjector (const int /*i*/) { return nullptr; }

    /**
     * Evolve is the central WarpXParticleContainer function that advances
     * particles for a time dt (typically one timestep). It is a pure virtual
     * function for flexibility.
     */
    virtual void Evolve (ablastr::fields::MultiFabRegister& fields,
                         int lev,
                         const std::string& current_fp_string,
                         amrex::Real t, amrex::Real dt, DtType a_dt_type=DtType::Full, bool skip_deposition=false,
                         PushType push_type=PushType::Explicit) = 0;

    virtual void PostRestart () = 0;

    void AllocData ();

    /**
     * \brief Virtual method to initialize runtime attributes. Must be overriden by each derived
     * class.
     */
    virtual void DefaultInitializeRuntimeAttributes (
        typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType& pinned_tile,
        int n_external_attr_real,
        int n_external_attr_int) = 0;

    ///
    /// This pushes the particle positions by one half time step.
    /// It is used to desynchronize the particles after initialization
    /// or when restarting from a checkpoint.
    ///
    void PushX (         amrex::Real dt);
    void PushX (int lev, amrex::Real dt);

    ///
    /// This pushes the particle momenta by dt.
    ///
    virtual void PushP (int lev, amrex::Real dt,
                        const amrex::MultiFab& Ex,
                        const amrex::MultiFab& Ey,
                        const amrex::MultiFab& Ez,
                        const amrex::MultiFab& Bx,
                        const amrex::MultiFab& By,
                        const amrex::MultiFab& Bz) = 0;

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
    void DepositCurrent (ablastr::fields::MultiLevelVectorField const & J,
                         amrex::Real dt, amrex::Real relative_time);

    /**
     * \brief Deposit charge density.
     *
     * \param[in,out] rho vector of charge densities (one pointer to MultiFab per mesh refinement level)
     * \param[in] local if false, exchange the data in the guard cells after the deposition
     * \param[in] reset if true, reset all values of rho to zero
     * \param[in] apply_boundary_and_scale_volume whether to apply boundary conditions and scale the final density by some volume norm in RZ geometry
     * \param[in] interpolate_across_levels whether to average down from the fine patch to the coarse patch
     * \param[in] icomp component of the MultiFab where rho is deposited (old, new)
     */
    void DepositCharge (const ablastr::fields::MultiLevelScalarField& rho,
                        bool local = false, bool reset = false,
                        bool apply_boundary_and_scale_volume = false,
                        bool interpolate_across_levels = true,
                        int icomp = 0);
    void DepositCharge (amrex::MultiFab* rho, int lev,
                        bool local = false, bool reset = false,
                        bool apply_boundary_and_scale_volume = false,
                        int icomp = 0);

    std::unique_ptr<amrex::MultiFab> GetChargeDensity(int lev, bool local = false);

    virtual void DepositCharge (WarpXParIter& pti,
                               RealVector const & wp,
                               const int* ion_lev,
                               amrex::MultiFab* rho,
                               int icomp,
                               long offset,
                               long np_to_deposit,
                               int thread_num,
                               int lev,
                               int depos_lev);

    virtual void DepositCurrent (WarpXParIter& pti,
                                RealVector const & wp,
                                RealVector const & uxp,
                                RealVector const & uyp,
                                RealVector const & uzp,
                                int const* ion_lev,
                                amrex::MultiFab* jx,
                                amrex::MultiFab* jy,
                                amrex::MultiFab* jz,
                                long offset,
                                long np_to_deposit,
                                int thread_num,
                                int lev,
                                int depos_lev,
                                amrex::Real dt,
                                amrex::Real relative_time,
                                PushType push_type);

    // If particles start outside of the domain, ContinuousInjection
    // makes sure that they are initialized when they enter the domain, and
    // NOT before. Virtual function, overriden by derived classes.
    // Current status:
    // PhysicalParticleContainer: implemented.
    // LaserParticleContainer: implemented.
    // RigidInjectedParticleContainer: not implemented.
    virtual void ContinuousInjection(const amrex::RealBox& /*injection_box*/) {}

    /**
     * \brief Update antenna position for continuous injection of lasers
     *        in a boosted frame. Empty function for containers other than lasers.
     */
    virtual void UpdateAntennaPosition(const amrex::Real /*dt*/) {}

    bool doContinuousInjection() const {return do_continuous_injection;}

    // Inject a continuous flux of particles from a defined plane
    virtual void ContinuousFluxInjection(amrex::Real /*t*/, amrex::Real /*dt*/) {}

    int getSpeciesId() const {return species_id;}

    ///
    /// This returns the total charge for all the particles in this ParticleContainer.
    /// This is needed when solving Poisson's equation with periodic boundary conditions.
    ///
    amrex::ParticleReal sumParticleCharge(bool local = false);
    amrex::ParticleReal sumParticleWeight(bool local = false);

    std::array<amrex::ParticleReal, 3> meanParticleVelocity(bool local = false);

    amrex::ParticleReal maxParticleVelocity(bool local = false);

    /**
     * \brief Adds n particles to the simulation
     *
     * @param[in] lev refinement level (unused)
     * @param[in] n the number of particles to add
     * @param[in] x x component of the position of particles to be added
     * @param[in] y y component of the position of particles to be added
     * @param[in] z z component of the position of particles to be added
     * @param[in] ux x component of the momentum of particles to be added
     * @param[in] uy y component of the momentum of particles to be added
     * @param[in] uz z component of the momentum of particles to be added
     * @param[in] nattr_real number of runtime real attributes to initialize with the attr_real
     * array. (particle weight is treated as a real attribute in this routine). The remaining
     * runtime real attributes are initialized in the method DefaultInitializeRuntimeAttributes.
     * @param[in] attr_real value of real attributes to initialize
     * @param[in] nattr_int number of runtime int attributes to initialize with the attr_int array.
     * The remaining runtime int attributes are initialized in the method
     * DefaultInitializeRuntimeAttributes.
     * @param[in] attr_int value of int attributes to initialize
     * @param[in] uniqueparticles if true, each MPI rank calling this function creates n
     * particles. Else, all MPI ranks work together to create n particles in total.
     * @param[in] id if different than -1, this id will be assigned to the particles (used for
     * particle tagging in some routines, e.g. SplitParticle)
     */
    void AddNParticles (int lev, long n,
                        amrex::Vector<amrex::ParticleReal> const & x,
                        amrex::Vector<amrex::ParticleReal> const & y,
                        amrex::Vector<amrex::ParticleReal> const & z,
                        amrex::Vector<amrex::ParticleReal> const & ux,
                        amrex::Vector<amrex::ParticleReal> const & uy,
                        amrex::Vector<amrex::ParticleReal> const & uz,
                        int nattr_real,
                        amrex::Vector<amrex::Vector<amrex::ParticleReal>> const & attr_real,
                        int nattr_int, amrex::Vector<amrex::Vector<int>> const & attr_int,
                        int uniqueparticles, amrex::Long id=-1);

    /** Remove particles with invalid ID
    *
    * This is a local operation (no MPI communication) and is thus preferable over `ReDistribute`
    * (which also removes invalid particles, but involves MPI communications at the same time)
    */
    void deleteInvalidParticles ();

    virtual void ReadHeader (std::istream& is) = 0;

    virtual void WriteHeader (std::ostream& os) const = 0;

    static void ReadParameters ();

    static void BackwardCompatibility ();

    /** \brief Apply particle BC.
     */
    void ApplyBoundaryConditions ();

    bool do_splitting = false;
    int do_not_deposit = 0;
    bool initialize_self_fields = false;
    amrex::Real self_fields_required_precision = amrex::Real(1.e-11);
    amrex::Real self_fields_absolute_tolerance = amrex::Real(0.0);
    int self_fields_max_iters = 200;
    int self_fields_verbosity = 2;

    // External fields added to particle fields.
    amrex::Vector<amrex::ParticleReal> m_B_external_particle;
    amrex::Vector<amrex::ParticleReal> m_E_external_particle;

    //! Current injection position
    amrex::Real m_current_injection_position = 0.;

    // split along diagonals (0) or axes (1)
    int split_type = 0;

    /** Whether back-transformed diagnostics need to be performed for a particular species.
     *
     * \param[in] do_back_transformed_particles The parameter to set if back-transformed particles are set to true/false
     */
    void SetDoBackTransformedParticles(const bool do_back_transformed_particles) {
        m_do_back_transformed_particles = do_back_transformed_particles;
    }

    //amrex::Real getCharge () {return charge;}
    amrex::ParticleReal getCharge () const {return charge;}
    //amrex::Real getMass () {return mass;}
    amrex::ParticleReal getMass () const {return mass;}

    int DoFieldIonization() const { return do_field_ionization; }

#ifdef WARPX_QED
    //Species for which QED effects are relevant should override these methods
    virtual bool has_quantum_sync() const {return false;}
    virtual bool has_breit_wheeler() const {return false;}

    int DoQED() const { return has_quantum_sync() || has_breit_wheeler(); }
#else
    int DoQED() const { return false; }
#endif

    /* \brief This function tests if the current species
    *  is of a given PhysicalSpecies (specified as a template parameter).
    * @tparam PhysSpec the PhysicalSpecies to test against
    * @return the result of the test
    */
    template<PhysicalSpecies PhysSpec>
    bool AmIA () const noexcept {return (physical_species == PhysSpec);}

    /**
    * \brief This function returns a string containing the name of the species type
    */
    std::string getSpeciesTypeName () const {return species::get_name(physical_species);}

    /**
     * \brief Virtual method to resample the species. Overriden by PhysicalParticleContainer only.
     * Empty body is here because making the method purely virtual would mean that we need to
     * override the method for every derived class. Note that in practice this function is never
     * called because resample() is only called for PhysicalParticleContainers.
     */
    virtual void resample (const int /*timestep*/, bool /*verbose*/) {}

    /**
     * When using runtime components, AMReX requires to touch all tiles
     * in serial and create particles tiles with runtime components if
     * they do not exist (or if they were defined by default, i.e.,
     * without runtime component).
     */
    void defineAllParticleTiles () noexcept;

    virtual std::vector<std::string> getUserIntAttribs () const { return std::vector<std::string>{}; }

    virtual std::vector<std::string> getUserRealAttribs () const { return std::vector<std::string>{}; }

    virtual amrex::Vector< amrex::Parser* > getUserIntAttribParser () const { return amrex::Vector<amrex::Parser*>{}; }

    virtual amrex::Vector< amrex::Parser* > getUserRealAttribParser () const { return amrex::Vector<amrex::Parser*>{}; }

#ifdef WARPX_QED
    virtual BreitWheelerEngine* get_breit_wheeler_engine_ptr () const {return nullptr;}
    virtual QuantumSynchrotronEngine* get_quantum_sync_engine_ptr () const {return nullptr;}
#endif

    void setDoNotPush (bool flag) { do_not_push = flag; }

protected:
    int species_id;

    amrex::ParticleReal charge;
    amrex::ParticleReal mass;
    PhysicalSpecies physical_species = PhysicalSpecies::unspecified;

    // Controls boundaries for particles exiting the domain
    ParticleBoundaries m_boundary_conditions;

    //! instead of depositing (current, charge) on the finest patch level, deposit to the coarsest grid
    bool m_deposit_on_main_grid = false;

    //! instead of gathering fields from the finest patch level, gather from the coarsest
    bool m_gather_from_main_grid = false;

    bool do_not_push = false;
    int do_not_gather = 0;

    // Whether to allow particles outside of the simulation domain to be
    // initialized when they enter the domain.
    // This is currently required because continuous injection does not
    // support all features allowed by direct injection.
    int do_continuous_injection = 0;

    int do_field_ionization = 0;
    int do_adk_correction = 0;
    int ionization_product;
    std::string ionization_product_name;
    int ion_atomic_number;
    int ionization_initial_level = 0;
    amrex::Gpu::DeviceVector<amrex::Real> ionization_energies;
    amrex::Gpu::DeviceVector<amrex::Real> adk_power;
    amrex::Gpu::DeviceVector<amrex::Real> adk_prefactor;
    amrex::Gpu::DeviceVector<amrex::Real> adk_exp_prefactor;
    /** for correction in Zhang et al., PRA 90, 043410 (2014). a1, a2, a3, Ecrit. */
    amrex::Gpu::DeviceVector<amrex::Real> adk_correction_factors;
    std::string physical_element;

    int do_resampling = 0;

    /** Whether back-transformed diagnostics is turned on for the corresponding species.*/
    bool m_do_back_transformed_particles = false;

#ifdef WARPX_QED
    //Species can receive a shared pointer to a QED engine (species for
    //which this is relevant should override these functions)
    virtual void
    set_breit_wheeler_engine_ptr(const std::shared_ptr<BreitWheelerEngine>&){}
    virtual void
    set_quantum_sync_engine_ptr(const std::shared_ptr<QuantumSynchrotronEngine>&){}

    int m_qed_breit_wheeler_ele_product;
    std::string m_qed_breit_wheeler_ele_product_name;
    int m_qed_breit_wheeler_pos_product;
    std::string m_qed_breit_wheeler_pos_product_name;
    int m_qed_quantum_sync_phot_product;
    std::string m_qed_quantum_sync_phot_product_name;

#endif
    amrex::Vector<amrex::FArrayBox> local_rho;
    amrex::Vector<amrex::FArrayBox> local_jx;
    amrex::Vector<amrex::FArrayBox> local_jy;
    amrex::Vector<amrex::FArrayBox> local_jz;

public:
    using PairIndex = std::pair<int, int>;
    using TmpParticleTile = std::array<amrex::Gpu::DeviceVector<amrex::ParticleReal>,
                                       TmpIdx::nattribs>;
    using TmpParticles = amrex::Vector<std::map<PairIndex, TmpParticleTile> >;

    TmpParticles getTmpParticleData () const noexcept {return tmp_particle_data;}

    int getIonizationInitialLevel () const noexcept {return ionization_initial_level;}

protected:
    TmpParticles tmp_particle_data;

private:
    void particlePostLocate(ParticleType& p, const amrex::ParticleLocData& pld, int lev) override;

};

#endif

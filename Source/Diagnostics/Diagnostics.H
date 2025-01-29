#ifndef WARPX_DIAGNOSTICS_H_
#define WARPX_DIAGNOSTICS_H_

#include "ParticleDiag/ParticleDiag.H"

#include "ComputeDiagFunctors/ComputeDiagFunctor_fwd.H"
#include "ComputeDiagFunctors/ComputeParticleDiagFunctor.H"
#include "FlushFormats/FlushFormat_fwd.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PinnedMemoryParticleContainer.H"

#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_AmrParticles.H>

#include <AMReX_BaseFwd.H>

#include <map>
#include <memory>
#include <string>
#include <vector>

/** All types of diagnostics. */
enum struct DiagTypes {Full, BackTransformed, BoundaryScraping, TimeAveraged};
/**
 * \brief base class for diagnostics.
 * Contains main routines to filter, compute and flush diagnostics.
 *
 * Each specific diagnostics derives from this class.
 */
class Diagnostics
{
public:
    /** Constructor
     *
     * @param i index of diagnostics in MultiDiagnostics::alldiags
     * @param name diagnostics name in the inputs file
     */
    Diagnostics (int i, std::string name, DiagTypes diag_type);

    /** Virtual Destructor to handle clean destruction of derived classes */
    virtual ~Diagnostics ();

    Diagnostics (Diagnostics const &)              = delete;
    Diagnostics& operator= (Diagnostics const & )  = delete;
    Diagnostics(Diagnostics&& )                    = default;
    Diagnostics& operator=(Diagnostics&& )         = default;

    /** Stores the diag type */
    DiagTypes m_diag_type;
    /** Pack (stack) all fields in the cell-centered output MultiFab m_mf_output.
     *
     * Fields are computed (e.g., cell-centered or back-transformed)
       on-the-fly using a functor. */
    void ComputeAndPack ();

    /** \brief Flush particle and field buffers to file using the FlushFormat member variable.
     *
     * This function should belong to class Diagnostics and not be virtual, as it flushes
     * the particle buffers (name not found yet) and the field buffers (m_mf_output), both
     * of which are members of Diagnostics. Yet, the implementation is left to derived classes
     * for now because:
     * - The functions underlying FlushFormat::WriteToFile expect a geometry object, which is
     *   WarpX::geom for FullDiagnostics but should be constructed for BTDiagnostics;
     * - The functions underlying FlushFormat::WriteToFile do not support writing a buffer to file
     *   multiple times yet.
     *  When these are fixed, the implementation of Flush should be in Diagnostics.cpp
     * \param[in] i_buffer index of the buffer data to be flushed.
     * \param[in] force_flush only used for BTD, whether to do a complete flush of the data
     * (including metadata listing the total number of particles) even if the snapshot is incomplete
     */
    virtual void Flush (int i_buffer, bool force_flush) = 0;
    /** Initialize pointers to main fields and allocate output multifab m_mf_output. */
    void InitData ();
    void InitDataBeforeRestart ();
    void InitDataAfterRestart ();
    /** Initialize functors that store pointers to the fields requested by the user.
     *
     * Derived classes MUST implement this function, and it must allocate m_all_field_functors
     * and fill it with ComputeDiagFunctor objects.
     * The functions is called at initialization and when the domain is decomposed
     * during the simulation to load-balance.
     * \param[in] lev level on which the vector of unique_ptrs to field functors is initialized.
     */
    virtual void InitializeFieldFunctors (int lev) = 0;
    /** Initialize functors that store pointers to the fields requested by the user.
     *
     * The function is called at initialization and when the domain is decomposed
     * during the simulation to load-balance.
     * \param[in] lev level on which the vector of unique_ptrs to field functors is initialized.
     */
    virtual void InitializeFieldFunctorsRZopenPMD ([[maybe_unused]] int lev) { }
    /** Initialize functors that store pointers to the species data requested by the user. */
    virtual void InitializeParticleFunctors () {}
    /** whether to compute and pack data in output buffers at this time step
     * \param[in] step current time step
     * \param[in] force_flush if true, return true for any step
     * \return bool, whether to compute and pack data
     */
    virtual bool DoComputeAndPack(int step, bool force_flush=false) = 0;
    /** whether to flush at this time step
     * \param[in] step current time step
     * \param[in] i_buffer index of the buffer data to be flushed.
     * \param[in] force_flush if true, return true for any step
     * \return bool, whether to flush
     */
    virtual bool DoDump (int step, int i_buffer, bool force_flush=false) = 0;
    /** Start a new iteration, i.e., dump has not been done yet. */
    void NewIteration () {m_already_done = false;}
    /** Perform necessary operations with user-defined diagnostic parameters
     *  to filter (coarsen, slice), compute (cell-center, back-transform),
     *  and flush the output data stored in buffers, m_mf_output.
     * \param[in] step current timestep
     * \param[in] force_flush used to force-fully write data stored in buffers.
     */
    void FilterComputePackFlush (int step, bool force_flush=false);
    /** Whether the last timestep is always dumped */
    [[nodiscard]] bool DoDumpLastTimestep () const {return  m_dump_last_timestep;}
    /** Returns the number of snapshots used in BTD. For Full-Diagnostics, the value is 1*/
    [[nodiscard]] int getnumbuffers() const {return m_num_buffers;}
    /** Time in lab-frame associated with the ith snapshot
     * \param[in] i_buffer index of the buffer
     */
    virtual amrex::Real gettlab ( [[maybe_unused]] int i_buffer) {return 0.;}
    /** Set time in lab-frame for the ith snapshot
     * \param[in] i_buffer index of the buffer
     * \param[in] tlab value used to set the lab-frame time for the ith buffer
     */
    virtual void settlab ( [[maybe_unused]] int i_buffer, [[maybe_unused]] amrex::Real tlab) { }
    /** Returns k-index in big end of the buffer box currently being filled for the ith snapshot in BTD
     * \param[in] i_buffer index of the buffer
     */
    virtual int get_buffer_k_index_hi ( [[maybe_unused]] int i_buffer) {return 0; }
    /** Set k-index of the buffer in the moving window direction currently being filled for the ith snapshot
     * \param[in] i_buffer index of the buffer
     * \param[in] kindex value used to set the k-index for the big end of the buffer box
     */
    virtual void set_buffer_k_index_hi ( [[maybe_unused]] int i_buffer, [[maybe_unused]] int kindex) { }
    /** Returns lo-end of the lab-frame physical domain for the ith snapshot in BTD
     * \param[in] i_buffer index of the buffer
     * \param[in] idim dimension along which the lo-end of the lab-frame physical domain is returned
     */
    virtual amrex::Real get_snapshot_domain_lo ([[maybe_unused]] int i_buffer, [[maybe_unused]] int idim) {return 0.; }
    /** Returns hi-end of the lab-frame physical domain for the ith snapshot in BTD
     * \param[in] i_buffer index of the buffer
     * \param[in] idim dimension along which the hi-end of the lab-frame physical domain is returned
     */
    virtual amrex::Real get_snapshot_domain_hi ([[maybe_unused]] int i_buffer, [[maybe_unused]] int idim) {return 0.; }
    /** Sets lo-end of the lab-frame physical domain for the ith snapshot in BTD
     * \param[in] i_buffer index of the buffer
     * \param[in] idim dimension along which the lo-end of the lab-frame physical domain is set
     * \param[in] domain_lab_lo value of the lo-end of the lab-frame physical domain for the ith snapshot
     */
    void setSnapshotDomainLo(int i_buffer, int idim,  amrex::Real domain_lab_lo) {m_snapshot_domain_lab[i_buffer].setLo(idim, domain_lab_lo); }
    /** Sets hi-end of the lab-frame physical domain for the ith snapshot in BTD
     * \param[in] i_buffer index of the buffer
     * \param[in] idim dimension along which the hi-end of the lab-frame physical domain is set
     * \param[in] domain_lab_hi value of the hi-end of the lab-frame physical domain for the ith snapshot
     */
    void setSnapshotDomainHi(int i_buffer, int idim,  amrex::Real domain_lab_hi) {m_snapshot_domain_lab[i_buffer].setHi(idim, domain_lab_hi); }
    /** Returns counter for the number of times buffer data for the ith snapshot has been flushed
     * \param[in] i_buffer index of the buffer
     */
    virtual int get_flush_counter ( [[maybe_unused]] int i_buffer) { return 0; }
    /** Sets counter for the number of times buffer data for the ith snapshot has been flushed to parameter input in the function
     * \param[in] i_buffer index of the buffer
     * \param[in] flush_counter value of the counter
     */
    virtual void set_flush_counter ( [[maybe_unused]] int i_buffer, [[maybe_unused]] int flush_counter) { }
    /** Returns 0/1 if the last valid Zslice for the ith snapshot has been filled (1) or not(0)
     * \param[in] i_buffer index of the buffer
     */
    virtual int get_last_valid_Zslice ( [[maybe_unused]] int i_buffer) { return 0; }
    /** Sets if the last valid Zslice for the ith snapshot has been filled (1) or not(0) with input parameter last_Valid_Zslice
     * \param[in] i_buffer index of the buffer
     * \param[in] last_valid_Zslice value to set m_lastValidZSlice in BTD for the ith snapshot (i_buffer)
     */
    virtual void set_last_valid_Zslice ( [[maybe_unused]] int i_buffer, [[maybe_unused]] int last_valid_Zslice) { }
    /** Returns 0/1 if the snapshot is fully filled and BTD for that snapshot is complete(1) or not(0)
     * \param[in] i_buffer index of the buffer
     */
    virtual int get_snapshot_full_flag ( [[maybe_unused]] int i_buffer) { return 0; }
    /** Sets the value for m_snapshot_full for the ith snapshot in BTD using parameter snapshot_full
     * \param[in] i_buffer index of the buffer
     * \param[in] snapshot_full value to set m_snapshot_full in BTD for the ith snapshot (i_buffer)
     */
    virtual void set_snapshot_full ( [[maybe_unused]] int i_buffer, [[maybe_unused]] int snapshot_full) { }

protected:
    /** Read Parameters of the base Diagnostics class */
    bool BaseReadParameters ();
    /** Whether to output a message when diagnostics are output */
    int m_verbose = 2;
    /** Initialize member variables of the base Diagnostics class. */
    void InitBaseData ();
    /** Initialize m_mf_output vectors and data required to construct the buffers
     * \param[in] i_buffer index of buffer for which the output MultiFab is defined.
     * \param[in] lev level on which the output MultiFab is defined
     */
    virtual void InitializeBufferData (int i_buffer, int lev, bool restart=false) = 0;
    /** Initialize member variables and arrays specific to the diagnostics in the
     *  derived classes.(FullDiagnostics, BTDiagnostics)
     */
    virtual void DerivedInitData () {}
    /** This function initialized particle buffers (not implemented in diagnostics, yet) */
    virtual void InitializeParticleBuffer () = 0;
    /** This function prepares buffer data as required for fields and particles. For back-transformed diagnostics,
     *  this function prepares the z coordinate in the boosted-frame and lab-frame.
     */
    virtual void PrepareBufferData () {}
    /** This function updates buffer data and computes the number of buffers filled in the output multifab as well
     *  as identifies if the last buffer has been filled as needed to close the output files.
     */
    virtual void UpdateBufferData () {}
    /** Prepare data (either fill-boundary or cell-centered data for
        back-transform diagnostics) to be processed for diagnostics.
     */
    virtual void PrepareFieldDataForOutput () {}
    /** The Particle Geometry, BoxArray, and RealBox are set for the lab-frame output */
    virtual void PrepareParticleDataForOutput () {}
    /** Update the physical extent of the diagnostic domain for moving window and
     *  galilean shift simulations
     *
     * \param[in] step current time step
     */
    virtual void MovingWindowAndGalileanDomainShift (int step) { amrex::ignore_unused(step); }
    /** Name of diagnostics: runtime parameter given in the input file. */
    std::string m_diag_name;
    /** Prefix for output directories */
    std::string m_file_prefix;
    /** Minimum number of digits to iteration number in file name */
    int m_file_min_digits = 6;
    /** Index of diagnostics in MultiDiagnostics::alldiags */
    int m_diag_index;
    /** Names of  each component requested by the user.
     * The list is appended with the average particle fields, if used.
     * In cylindrical geometry, this list is also appended with
     * automatically-constructed names for all modes of all fields.*/
    amrex::Vector< std::string > m_varnames;
    /** Names of plotfile fields requested by the user */
    amrex::Vector< std::string > m_varnames_fields;

    /** Names of particle field properties to output */
    amrex::Vector< std::string > m_pfield_varnames;
    /** Names of species for which to output particle field diagnostics */
    std::vector< std::string > m_pfield_species;
    /** Whether to do averaging for each of the particle field diagnostics */
    std::vector< bool > m_pfield_do_average;
    /** Species indices corresponding to elements of m_pfield_varnames */
    std::vector< int > m_pfield_species_index;
    /** List of the parser strings for the particle field diagnostics */
    std::vector< std::string > m_pfield_strings;
    /** Whether to use a filter function on particles before calculating particle field diagnostics */
    std::vector< bool> m_pfield_dofilter;
    /** List of parser strings for pre-average filtering for the particle field diagnostics */
    std::vector< std::string > m_pfield_filter_strings;

    /** If true, a dump is performed at the last timestep regardless of the required dump timesteps */
    bool m_dump_last_timestep = true;
    /** format for output files, "plotfile" or "openpmd" or "sensei" or "ascent"
     * The checkpoint format is applicable for FullDiagnostics only.
     */
    std::string m_format = "plotfile";
    /** Whether this iteration has already been dumped, to avoid writing data twice */
    int m_already_done = false;
    /** This class is responsible for flushing the data to file */
    std::unique_ptr<FlushFormat> m_flush_format;
    /** output multifab, where all fields are computed (cell-centered or back-transformed)
     *  and stacked.
     *  The first vector is for total number of snapshots. (=1 for FullDiagnostics)
     *  The second vector is loops over the total number of levels.
     */
    amrex::Vector< amrex::Vector< amrex::MultiFab > >  m_mf_output;
    /** summation multifab, where all fields (computed, cell-centered, and stacked)
     *  are summed for every step in a time averaging period.
     *  The first vector is for total number of snapshots. (=1 for FullDiagnostics)
     *  The second vector is loops over the total number of levels.
     */
    amrex::Vector< amrex::Vector <amrex::MultiFab > > m_sum_mf_output;

    /** Geometry that defines the domain attributes corresponding to output multifab.
     *  Specifically, the user-defined physical co-ordinates for the diagnostics
     *  is used to construct the geometry information for each MultiFab at
     *  the respective levels. This geometry will be used to write out
     *  plotfile data using the WriteToFile() function
     */
    amrex::Vector< amrex::Vector <amrex::Geometry> > m_geom_output;
    // a particle buffer here?
    int nlev; /**< number of levels to output */
    int nmax_lev; /**< max_level to allocate output multifab and vector of field functors. */
    /** Number of levels to be output*/
    int nlev_output;
    /** Names of species to write to output */
    std::vector< std::string > m_output_species_names;
    /** Names of all species in the simulation */
    std::vector< std::string > m_all_species_names;
    /** The first vector is for total number of snapshots. (=1 for FullDiagnostics)
        The second vector handles output for 1 species
      */
    amrex::Vector< amrex::Vector< ParticleDiag> > m_output_species;
    /** Vector of (pointers to) functors to compute output fields, per level,
     * per component. This allows for simple operations (averaging to
     * cell-center for standard EB fields) as well as more involved operations
     * (back-transformed diagnostics, filtering, reconstructing cartesian
     * fields in cylindrical). */
    amrex::Vector< amrex::Vector <std::unique_ptr<ComputeDiagFunctor > > > m_all_field_functors;
    /** Coarsening ratio such that, fields are averaged to the coarsened grid.
      * The ratio should render the grid to be coarsenable (as defined by AMReX). */
    amrex::IntVect m_crse_ratio = amrex::IntVect(1);
    /** Lower corner of the diagnostics output, in physical coordinates */
    amrex::Vector< amrex::Real> m_lo;
    /** Higher corner of the diagnostics output, in physical coordinates */
    amrex::Vector< amrex::Real> m_hi;
    /** Number of output buffers. The value is set to 1 for all FullDiagnostics */
    int m_num_buffers;
    /** Array of species indices that dump rho per species */
    amrex::Vector<int> m_rho_per_species_index;
    /** Array of species indices that dump temperature per species */
    amrex::Vector<int> m_T_per_species_index;
    /** Vector of particle buffer vectors for each snapshot */
    amrex::Vector< amrex::Vector<std::unique_ptr<PinnedMemoryParticleContainer> > > m_particles_buffer;
    /** Vector of pointers to functors to compute particle output per species*/
    amrex::Vector< std::unique_ptr<ComputeParticleDiagFunctor> > m_all_particle_functors;

    /** Vector of total number of particles in the buffer, per species, per snapshot.
     *  The first vector is for total number of snapshots and second vector loops
     *  over the total number of species selected for diagnostics.
     */
    amrex::Vector< amrex::Vector <int> > m_totalParticles_in_buffer;
    /** Vector of user-defined physical region for diagnostics in lab-frame
     *  for each back-transformed snapshot */
    amrex::Vector<amrex::RealBox> m_snapshot_domain_lab;

};

#endif // WARPX_DIAGNOSTICS_H_

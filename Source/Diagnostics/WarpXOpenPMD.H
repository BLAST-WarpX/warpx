/* Copyright 2019-2021 Axel Huebl, Junmin Gu, Maxence Thevenet
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_OPEN_PMD_H_
#define WARPX_OPEN_PMD_H_

#include "Particles/WarpXParticleContainer.H"
#include "Diagnostics/FlushFormats/FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_AmrParticles.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

//
//
class WarpXParticleCounter
{
public:
  using ParticleContainer = typename WarpXParticleContainer::ContainerLike<amrex::PinnedArenaAllocator>;
  using ParticleIter = typename amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;

  WarpXParticleCounter (ParticleContainer* pc);
  [[nodiscard]] unsigned long GetTotalNumParticles () const {return m_Total;}

  std::vector<unsigned long long> m_ParticleOffsetAtRank;
  std::vector<unsigned long long> m_ParticleSizeAtRank;
private:
  /** get the offset in the overall particle id collection
  *
  * @param[out] numParticles particles on this processor  / amrex fab
  * @param[out] offset particle offset over all, mpi-global amrex fabs
  * @param[out] sum number of all particles from all amrex fabs
  */
  void GetParticleOffsetOfProcessor (const long& numParticles,
                    unsigned long long& offset,
                    unsigned long long& sum)  const ;


  int m_MPIRank = 0;
  int m_MPISize = 1;

  unsigned long long m_Total = 0;

  std::vector<unsigned long long> m_ParticleCounterByLevel;
};


#ifdef WARPX_USE_OPENPMD
//
//
/** Writer logic for openPMD particles and fields */
class WarpXOpenPMDPlot
{
public:
  using ParticleContainer = typename WarpXParticleContainer::ContainerLike<amrex::PinnedArenaAllocator>;
  using ParticleIter = typename amrex::ParConstIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;

  /** Initialize openPMD I/O routines
   *
   * @param ie  iteration encoding from openPMD: "group, file, variable"
   * @param filetype file backend, e.g. "bp" or "h5"
   * @param operator_type openPMD-api backend operator (compressor) for ADIOS2
   * @param operator_parameters openPMD-api backend operator parameters for ADIOS2
   * @param engine_type ADIOS engine for output
   * @param engine_parameters map of parameters for the engine
   * @param fieldPMLdirections PML field solver, @see WarpX::getPMLdirections()
   * @param authors a string specifying the authors of the simulation (can be empty)
   */
  WarpXOpenPMDPlot (openPMD::IterationEncoding ie,
                    const std::string& filetype,
                    const std::string& operator_type,
                    const std::map< std::string, std::string >& operator_parameters,
                    const std::string& engine_type,
                    const std::map< std::string, std::string >& engine_parameters,
                    const std::vector<bool>& fieldPMLdirections,
                    const std::string& authors);

  ~WarpXOpenPMDPlot ();

  WarpXOpenPMDPlot ( WarpXOpenPMDPlot const &)             = delete;
  WarpXOpenPMDPlot& operator= ( WarpXOpenPMDPlot const & ) = delete;
  WarpXOpenPMDPlot ( WarpXOpenPMDPlot&& )                  = default;
  WarpXOpenPMDPlot& operator= ( WarpXOpenPMDPlot&& )       = default;

  /** Set Iteration Step for the series
   *
   * @note If an iteration has been written, then it will give a warning
   *
   */
  void SetStep (int ts, const std::string& dirPrefix, int file_min_digits,
                bool isBTD=false);

  /** Close the step
   *
   * Signal that no further updates will be written for the step.
   */
  void CloseStep (bool isBTD = false, bool isLastBTDFlush = false);

  void WriteOpenPMDParticles (
              const amrex::Vector<ParticleDiag>& particle_diags,
              amrex::Real time,
              bool use_pinned_pc = false,
              bool isBTD = false,
              bool isLastBTDFlush = false);

  /** Write out all openPMD fields for all active MR levels
   *
   * @param varnames variable names in each multifab
   * @param mf multifab for each level
   * @param geom for each level
   * @param output_levels the finest level to output, <= maxLevel
   * @param iteration the current iteration or reconstructed labframe station number
   * @param time the current simulation time in the lab frame
   * @param isBTD true if this is part of a back-transformed diagnostics (BTD) station flush;
                  in BTD, we write multiple times to the same iteration
   * @param full_BTD_snapshot the geometry of the full lab frame for BTD
   */
  void WriteOpenPMDFieldsAll (
              const std::vector<std::string>& varnames,
              const amrex::Vector<amrex::MultiFab>& mf,
              amrex::Vector<amrex::Geometry>& geom,
              int output_levels,
              int iteration,
              double time,
              bool isBTD = false,
              const amrex::Geometry& full_BTD_snapshot=amrex::Geometry() ) const;

  /** Return OpenPMD File type ("bp" or "h5" or "json")*/
  std::string OpenPMDFileType () { return m_OpenPMDFileType; }

private:
  void Init (openPMD::Access access, bool isBTD);


  /** Get the openPMD::Iteration object of the current Series
   *
   * We use this helper function to differentiate between efficient, temporally
   * sequentially increasing writes to iteration numbers and random-access
   * writes to iterations, e.g., as needed for back-transformed diagnostics.
   *
   * @param[in] iteration iteration number (lab-frame for BTD)
   * @param[in] isBTD is this a backtransformed diagnostics write?
   * @return the iteration object
   */
  [[nodiscard]] inline openPMD::Iteration GetIteration (int const iteration, bool const isBTD) const
  {
    if (isBTD)
    {
        return m_Series->iterations[iteration];
    } else {
        return m_Series->writeIterations()[iteration];
    }
  }


  /** This function does initial setup for the fields when interation is newly created
   *  @param[in] meshes   The meshes in a series
   *  @param[in] full_geom The geometry
   */
  void SetupFields (
      openPMD::Container< openPMD::Mesh >& meshes,
      amrex::Geometry& full_geom
  ) const;

  void SetupMeshComp (
      openPMD::Mesh& mesh,
      amrex::Geometry const& full_geom,
      std::string const& comp_name,
      std::string const& field_name,
      amrex::MultiFab const& mf,
      bool var_in_theta_mode
  ) const;

  /** Get Component Names from WarpX name
   *
   * Get component names of a field for openPMD-api book-keeping
   * Level is reflected as _lvl<meshLevel>
   *
   * @param[in] meshLevel     level of mesh
   * @param[in] varname       name from WarpX
   * @param[out] field_name   field name for openPMD-api output
   * @param[in] comp_name     comp name for openPMD-api output
   * @param[in] var_in_theta_mode indicate if this field will be output with theta
   *                              modes (instead of a reconstructed 2D slice)
   */
  void GetMeshCompNames (
      int meshLevel,
      const std::string& varname,
      std::string& field_name,
      std::string& comp_name,
      bool var_in_theta_mode
  ) const;

  /** This function sets up the entries for storing the particle positions and global IDs
  *
  * @param[in] currSpecies Corresponding openPMD species
  * @param[in] positionComponents user-selected components of the particle position
  * @param[in] np          Number of particles
  * @param[in] isBTD       Is this a back-transformed diagnostics output?
  */
  void SetupPos (
        openPMD::ParticleSpecies& currSpecies,
        std::vector<std::string> const & positionComponents,
        const unsigned long long& np,
        bool isBTD = false);

  /** This function sets constant particle records and ED-PIC attributes.
   *
   * Sets the entries for storing particle position offset, constant records (charge, mass) and ED-PIC attributes.
   *
   * @param[in] currSpecies Corresponding openPMD species
   * @param[in] positionComponents user-selected components of the particle position
   * @param[in] np          Number of particles
   * @param[in] charge      Charge of the particles (note: fix for ions)
   * @param[in] mass        Mass of the particles
   */
  void SetConstParticleRecordsEDPIC (
        openPMD::ParticleSpecies& currSpecies,
        std::vector<std::string> const & positionComponents,
        const unsigned long long& np,
        amrex::ParticleReal charge,
        amrex::ParticleReal mass);

  /** This function sets up the entries for particle properties
   *
   * @param[in] pc The particle container of the species
   * @param[in] currSpecies The openPMD species
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   * @param[in] np  Number of particles
   * @param[in] isBTD whether this is a back-transformed diagnostic
   */
  void SetupRealProperties (ParticleContainer const * pc,
               openPMD::ParticleSpecies& currSpecies,
               const amrex::Vector<int>& write_real_comp,
               const amrex::Vector<std::string>& real_comp_names,
               const amrex::Vector<int>& write_int_comp,
               const amrex::Vector<std::string>& int_comp_names,
               unsigned long long np, bool isBTD = false) const;

  /** This function saves the values of the entries for particle properties
   *
   * @param[in] pti WarpX particle iterator
   * @param[in] currSpecies The openPMD species to save to
   * @param[in] offset offset to start saving  the particle iterator contents
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   */
  void SaveRealProperty (ParticleIter& pti, //int, int,
            openPMD::ParticleSpecies& currSpecies,
            unsigned long long offset,
            const amrex::Vector<int>& write_real_comp,
            const amrex::Vector<std::string>& real_comp_names,
            const amrex::Vector<int>& write_int_comp,
            const amrex::Vector<std::string>& int_comp_names) const;

  /** This function saves the plot file
   *
   * @param[in] pc WarpX particle container
   * @param[in] name species name
   * @param[in] iteration timestep
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   * @param[in] charge         Charge of the particles (note: fix for ions)
   * @param[in] mass           Mass of the particles
   * @param[inout] ParticleFlushOffset previously flushed number of particles in BTD
   * @param[in] isBTD is this a backtransformed diagnostics (BTD) write?
   * @param[in] isLastBTDFlush is this the last time we will flush this BTD station?
   */
  void DumpToFile (ParticleContainer* pc,
            const std::string& name,
            int iteration,
            const amrex::Vector<int>& write_real_comp,
            const amrex::Vector<int>& write_int_comp,
            const amrex::Vector<std::string>& real_comp_names,
            const amrex::Vector<std::string>&  int_comp_names,
            amrex::ParticleReal charge,
            amrex::ParticleReal mass,
            bool isBTD = false,
            bool isLastBTDFlush = false);

  /** Get the openPMD-api filename for openPMD::Series
   *
   * No need for ts in the file name, openPMD handles steps (iterations).
   *
   * @param[inout] filepath the path and filename for openPMD::Series
   *               passes a prefix path in and appends the filename
   * @return pure filename w/o path
   */
  std::string GetFileName (std::string& filepath);

  std::unique_ptr<openPMD::Series> m_Series;

  /** This is the output directory
   *
   * This usually does not yet end in a `/`.
   * It does not yet include the file prefix of the openPMD series, which will
   * be appended by the GetFileName function.
   */
  std::string m_dirPrefix;

  /** This is the minimum number of digits in the step number that is used as the
   * suffix for file names when doing file based encoding */
  int m_file_min_digits;

  int m_MPIRank = 0;
  int m_MPISize = 1;

  openPMD::IterationEncoding m_Encoding = openPMD::IterationEncoding::fileBased;
  std::string m_OpenPMDFileType = "bp"; //! MPI-parallel openPMD backend: bp or h5
  std::string m_OpenPMDoptions = "{}"; //! JSON option string for openPMD::Series constructor
  int m_CurrentStep  = -1;

  // meta data
  std::vector< bool > m_fieldPMLdirections; //! @see WarpX::getPMLdirections()

  // The authors' string
  std::string m_authors;
};
#endif // WARPX_USE_OPENPMD

#endif // WARPX_OPEN_PMD_H

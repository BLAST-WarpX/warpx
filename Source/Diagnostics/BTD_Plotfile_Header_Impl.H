/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BTD_PLOTFILE_HEADER_IMPL_H
#define WARPX_BTD_PLOTFILE_HEADER_IMPL_H

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include <string>

/**
 * \brief Class to read, modify, and write plotfile header when
 * back-transformed diag format is selected as plotfile.
 * This class enables multiple BTD buffers to be interweaved and stitched into
 * a single plotfile with a single Header.
 */

class BTDPlotfileHeaderImpl
{
public:
    /** Constructor.
     * \param[in] Headerfile_path string containing path of Headerfile
     */
    BTDPlotfileHeaderImpl (std::string const& Headerfile_path);

    /** Returns the Header file version for plotfile */
    [[nodiscard]] std::string fileVersion () const noexcept {return m_file_version; }
    /** Returns the number of components written in the Headerfile */
    [[nodiscard]] int ncomp () const noexcept {return m_nComp; }
    /** Returns the names of components in the Headerfile */
    [[nodiscard]] const amrex::Vector<std::string>& varnames () const noexcept {return m_varnames; }
    /** Returns the number of dimensions in the Headerfile */
    [[nodiscard]] int spaceDim () const noexcept {return m_spacedim; }
    /** Returns the physical time in the simulation in the boosted-frame */
    [[nodiscard]] amrex::Real time () const noexcept {return m_time; }
    /** Returns finest level output in the Headerfile */
    [[nodiscard]] int finestLevel () const noexcept { return m_finest_level; }
    /** Returns the physical co-ordinates of the lower corner in dimension, idim,
     *  that corresponds the to the respective plotfile data
     */
    [[nodiscard]] amrex::Real problo (int dim) const noexcept {return m_prob_lo[dim]; }
    /** Returns the physical co-ordinates of the upper corner in dimension, idim,
     *  that corresponds the to the respective plotfile data.
     */
    [[nodiscard]] amrex::Real probhi (int dim) const noexcept {return m_prob_hi[dim]; }

    /** Returns the bounding box of the domain spanned in the plotfile */
    [[nodiscard]] amrex::Box probDomain () const noexcept {return m_prob_domain; }
    /** Returns timestep at which the plotfile was written */
    [[nodiscard]] int timestep () const noexcept {return m_timestep; }
    [[nodiscard]] int numFabs () const noexcept {return m_numFabs; }

    /** Returns physical co-ordinates of the lower-corner for the i-th Fab.
     *  \param[in] iFab id of the ith Fab in the list of Multifabs
     *  \returns   Array of lower-corner physical co-ordinates corresponding to the ith Fab
     */
    [[nodiscard]] amrex::Array<amrex::Real, AMREX_SPACEDIM> FabLo (int iFab) const noexcept {return m_glo[iFab]; }
    /** Returns physical co-ordinates of the lower-corner for the i-th Fab.
     *  \param[in] iFab id of the ith Fab in the list of Multifabs
     *  \returns   Array of lower-corner physical co-ordinates corresponding to the ith Fab
     */
    [[nodiscard]] amrex::Array<amrex::Real, AMREX_SPACEDIM> FabHi (int iFab) const noexcept {return m_ghi[iFab]; }
    /** Returns path to location of multifabs */
    [[nodiscard]] std::string CellPath () const noexcept {return m_CellPath; }

    /** Reads the Header file data for BTD */
    void ReadHeaderData ();
    /** Writes Header file data for BTD */
    void WriteHeader ();

    /** Sets the physical simulation time, m_time, in the Header file to a new_time.
     *  \param[in] new_time current time in the boosted-frame read from Header plotfile
     **/
    void set_time ( amrex::Real new_time) { m_time = new_time;}
    /** Sets the timestep, m_timestep, in the Header file to a new_timestep
     *  \param[in] new_timestep current timestep in the boosted-frame (read from the Header file of the newly flushed buffer)
     **/
    void set_timestep (int new_timestep) { m_timestep = new_timestep; }
    /** Set the ith-dimension physical co-ordinate of the lower corner.
     *  \param[in] dim dimension modified.
     *  \param[in] lo  lower-corner physical coordinate to be stored in dimension, dim.
     **/
    void set_problo (int dim, amrex::Real lo) { m_prob_lo[dim] = lo;}
    /** Set the ith-dimension physical co-ordinate of the upper corner.
     *  \param[in] dim dimension modified.
     *  \param[in] hi  upper-corner physical coordinate to be stored in dimension, dim.
     **/
    void set_probhi (int dim, amrex::Real hi) { m_prob_hi[dim] = hi;}
    /** Set the Domain box spanning the plotfile data in the modified plotfile and Header.
     * \param[in] domainBox spanning the domain corresponding to the plotfile.
     */
    void set_probDomain (amrex::Box domainBox) {m_prob_domain = domainBox; }
    /** Increments the number of fabs stored in the plotfile by one. */
    void IncrementNumFabs () { m_numFabs++;}
    void ResizeFabLo () { m_glo.resize(m_numFabs); }
    void ResizeFabHi () { m_ghi.resize(m_numFabs); }
    /** Append array of lower-corner physical coordinates corresponding to a new fab to
     *  the existing list of coordinates, m_glo.
     *  \param[in] newFabLo contains physical coordinates
     *                      of the newly appended fab-data to the plotfile.
     */
    void AppendNewFabLo (amrex::Array<amrex::Real, AMREX_SPACEDIM> newFabLo);
    /** Append array of upper-corner physical coordinates corresponding to a new fab to
     *  the existing list of coordinates, m_ghi.
     *  \param[in] newFabHi contains physical coordinates
     *                      of the newly appended fab-data to the plotfile.
     */
    void AppendNewFabHi (amrex::Array<amrex::Real, AMREX_SPACEDIM> newFabHi);
private:
    /** string containing path of the Header file. */
    std::string m_Header_path;
    /** String containing file version of the plotfile. */
    std::string m_file_version;
    /** Number of components in the plotfile. */
    int m_nComp;
    /** Names of components stored in the plotfile. */
    amrex::Vector<std::string> m_varnames;
    /** Number of dimensions stored in the plotfile, should be same as AMREX_SPACEDIM */
    int m_spacedim;
    /** Physical time */
    amrex::Real m_time;
    int m_finest_level, m_nlevel;
    /** Lower cornder physical coordinates of the domain spanned in the plotfile */
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_prob_lo {{AMREX_D_DECL(0.,0.,0.)}};
    /** Upper corner physical coordinates of the domain spanned in the plotfile */
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_prob_hi {{AMREX_D_DECL(1.,1.,1.)}};
    /** Cell size */
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_cell_size {{AMREX_D_DECL(1.,1.,1.)}};
    /** Box covering the span of the physical domain in the plotfile */
    amrex::Box m_prob_domain;
    int m_timestep;
    int m_coordsys;
    int m_bwidth;
    int m_cur_level;
    /** Number of Fabs in the plotfile. */
    int m_numFabs;
    /** Lower corner physical coordinates of each fab in the plotfile. */
    amrex::Vector<amrex::Array<amrex::Real, AMREX_SPACEDIM> > m_glo;
    /** Upper corner physical coordinates of each fab in the plotfile. */
    amrex::Vector<amrex::Array<amrex::Real, AMREX_SPACEDIM> > m_ghi;
    std::string m_CellPath;

};

/**
 * \brief Class to read, modify, and write MultiFab header in Level_0/Cell_H when
 * back-transformed diag format is selected as plotfile.
 * This class enables multiple fabs to be interweaved and stitched into
 * a single plotfile with a single Header, Cell_H.
 */
class BTDMultiFabHeaderImpl
{
    public:
    /** Constructor.
     * \param[in] Headerfile_path string containing path of Headerfile
     */
    BTDMultiFabHeaderImpl (std::string const& Headerfile_path);

    /** Reads the Multifab Header file and stores its data. */
    void ReadMultiFabHeader ();
    /** Writes the meta-data of the Multifab in a header file, with path, m_Header_path. */
    void WriteMultiFabHeader ();

    /** Returns size, m_ba_size, of the Box Array, m_ba.*/
    [[nodiscard]] int ba_size () const {return m_ba_size;}
    /** Returns box corresponding to the ith box in the BoxArray, m_ba.
     *  \param[in] ibox index of the box in the BoxArray.
     */
    [[nodiscard]] amrex::Box ba_box (int ibox) const {return m_ba[ibox]; }
    /** Returns prefix of the ith-fab on disk, i.e., ith fab of the MultiFab data.
     *  \param[in] ifab index of the ith fab in the MultiFab data.
     */
    std::string fodPrefix (int ifab) {return m_FabOnDiskPrefix[ifab]; }
    /** Returns name of the ith fab stored in the MultiFab. */
    std::string FabName (int ifab) {return m_fabname[ifab]; }
    /** Returns the starting byte of the ith fab data */
    int FabHead (int ifab) {return m_fabhead[ifab]; }
    /** Returns minimum value of all the components stored in the ith fab. */
    amrex::Vector<amrex::Real> minval(int ifab) { return m_minval[ifab];}
    /** Returns maximum value of all the components stored in the ith fab. */
    amrex::Vector<amrex::Real> maxval(int ifab) { return m_maxval[ifab];}
    void ResizeFabData ();
    /** Increments MultiFab size, m_ba_size, by add_mf_size.
     *  \param[in] add_mf_size number of new multifabs to be appended to the existing
     *             Box Array.
     */
    void IncreaseMultiFabSize (int add_mf_size) {m_ba_size += add_mf_size;}
    /** Set Box indices of the ith-box in Box Array, m_ba, to the new Box, ba_box.
     *  \param[in] ibox   index of the ith box in BoxArray, m_ba.
     *  \param[in] ba_box dimensions corresponding to the ith Fab.
     */
    void SetBox (int ibox, amrex::Box ba_box) { m_ba.set(ibox, ba_box); }
    /** Set Fab name of the ith fab to be written in the multifab Header file.*/
    void SetFabName (int ifab, const std::string& fodPrefix, const std::string& FabName,
                     int FabHead);
    /** Set minimum value of all the components for the ith fab. */
    void SetMinVal (int ifab, const amrex::Vector<amrex::Real>& minval);
    /** Set maximum value of all the components for the ith fab. */
    void SetMaxVal (int ifab, const amrex::Vector<amrex::Real>& maxval);
    private:
    /** Header file path */
    std::string m_Header_path;
    int m_vers;
    int m_how;
    /** number of components stored in the multifab. */
    int m_ncomp;
    /** number of guard cells in the multifab. */
    int m_ngrow;
    /** Size of the BoxArray, m_ba.*/
    int m_ba_size;
    /** BoxArray corresponding to the multifab stored in the plotfile.*/
    amrex::BoxArray m_ba;
    amrex::Vector<std::string> m_FabOnDiskPrefix;
    amrex::Vector<std::string> m_fabname;
    amrex::Vector<int> m_fabhead;
    /** The min of each component of each FAB in the BoxArray, m_ba.
     *  To access the min value of the ith fab and jth component [ifab][jcomp]*/
    amrex::Vector<amrex::Vector< amrex::Real> > m_minval;
    /** The max of each component of each FAB in the BoxArray, m_ba.
     *  To access the max value of the ith fab and jth component [ifab][jcomp]*/
    amrex::Vector<amrex::Vector< amrex::Real> > m_maxval;
    /** Copy values of the vector from the src vector, src, to dst vector. */
    void CopyVec (amrex::Vector<amrex::Real>& dst,
                  amrex::Vector<amrex::Real> src);
};

/**
 * \brief Class to read, modify, and write species header when
 * back-transformed diag format is selected as plotfile.
 * This class enables multiple BTD particle buffers to be interweaved and stitched into
 * a single plotfile with a single Header.
 */
class BTDSpeciesHeaderImpl
{
public:
    /** Constructor.
     * \param[in] Headerfile_path string containing path of Headerfile
     * \param[in] species_name string containing species name
     */
    BTDSpeciesHeaderImpl (std::string const& Headerfile_path, std::string const& species_name);

    /** Reads the Header file for BTD species*/
    void ReadHeader ();
    /** Writes the meta-data of species Header file, with path, m_Header_path*/
    void WriteHeader ();
    /** Set data Index of the data-file, DATAXXXXX, that the particles belong to*/
    void set_DataIndex (int lev, int box_id, int data_index);
    /** Add new_particles to existing to obtain the total number of particles of the species.
        \param[in] new_particles total particles in the new buffer
     */
    void AddTotalParticles (const int new_particles) { m_total_particles += new_particles;}
    /** Increment number of boxes in a box array by 1, with every flush. */
    void IncrementParticleBoxArraySize () { m_particleBoxArray_size[m_finestLevel]++; }
    /** Append particle info of the newly added box, namely,
        m_which_data, m_particles_per_box, m_offset_per_box.
     */
    void AppendParticleInfoForNewBox (int data_index, int particles_per_box, int offset);
    /** Vector of data indices of all particle data for each level*/
    amrex::Vector< amrex::Vector<int> > m_which_data;
    /** Vector of particles per box for each level*/
    amrex::Vector< amrex::Vector<int> > m_particles_per_box;
    /** Vector of particle offset per box of each particle data file for each level*/
    amrex::Vector< amrex::Vector<int> > m_offset_per_box;
    /** Vector of box array size per for each level*/
    amrex::Vector<int> m_particleBoxArray_size;
    /** Total number of particles of species, m_species_name, in the particle output*/
    int m_total_particles;
    /** Id of the next particle*/
    int m_nextid;
    /** finest level of the particle output*/
    int m_finestLevel;
    private:
    /** Path of the headerfile */
    std::string m_Header_path;
    /** Species name of the particles flushed out */
    std::string m_species_name;
    /** String containing file version of the species output */
    std::string m_file_version;
    /** Number of dimensions stored in the plotfile, should be same as AMREX_SPACEDIM */
    int m_spacedim;
    /** Number of real attributes*/
    int m_num_output_real;
    /** Number of integer attributes*/
    int m_num_output_int;
    /** The real attribute names*/
    amrex::Vector<std::string> m_real_comp_names;
    /** The real integer names*/
    amrex::Vector<std::string> m_int_comp_names;
    /** if the output is a checkpoint */
    bool m_is_checkpoint;
};

/**
 * \brief Class to read, modify, and write particle header file, Particle_H, when
 * back-transformed diag format is selected as plotfile.
 * This class enables multiple particle buffers to be interweaved and stitched into
 * a single plotfile with a single Particle_H file.
 */
class BTDParticleDataHeaderImpl
{
public:
    /** Constructor.
     * \param[in] Headerfile_path containing path of Headerfile
     */
    BTDParticleDataHeaderImpl (std::string const& Headerfile_path);

    /** Reads the particle header file at m_Header_path and stores its data*/
    void ReadHeader ();
    /** Writes the meta-data of particle box array in header file, with path, m_Header_path*/
    void WriteHeader () const;
    /** Returns the size of the box array, m_ba_size */
    [[nodiscard]] int ba_size () const {return m_ba_size; }
    /** Increases Box array size, m_ba_size, by add_size
     *  \param[in] add_size
     */
    void IncreaseBoxArraySize ( const int add_size) { m_ba_size += add_size;}
    /** Returns box corresponding to the ith box in the BoxArray, m_ba.
     *  \param[in] ibox index of the box in the BoxArray.
     */
    [[nodiscard]] amrex::Box ba_box (int ibox) const {return m_ba[ibox]; }
    /** Resize boxArray, m_ba, to size, m_ba_size. */
    void ResizeBoxArray () { m_ba.resize(m_ba_size); }
    /** Set Box indices of the ith-box in Box Array, m_ba, to the new Box, ba_box.
     *  \param[in] ibox index of the ith box in BoxArray, m_ba.
     *  \param[in] ba_box dimensions corresponding to the ith Fab.
     */
    void SetBox (int ibox, amrex::Box ba_box) { m_ba.set(ibox, ba_box); }
    /** Size of BoxArray, m_ba*/
    int m_ba_size;
    /** BoxArray for particle output*/
    amrex::BoxArray m_ba;
    /** string containing path of the particle output of species, species_name.*/
    std::string m_Header_path;
};

#endif //WARPX_BTD_PLOTFILE_HEADER_IMPL_H

/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BTDIAGNOSTICS_H_
#define WARPX_BTDIAGNOSTICS_H_

#include "Diagnostics.H"
#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Utils/WarpXConst.H"
#include "Utils/Parser/IntervalsParser.H"

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <limits>
#include <memory>
#include <string>

class BTDiagnostics final : public Diagnostics
{
public:

    BTDiagnostics (int i, const std::string& name, DiagTypes diag_type);

private:
    /** Whether to plot raw (i.e., NOT cell-centered) fields */
    bool m_plot_raw_fields = false;
    /** Whether to plot guard cells of raw fields */
    bool m_plot_raw_fields_guards = false;
    /** Read relevant parameters for BTD */
    void ReadParameters ();
    /** Determines timesteps at which BTD diagnostics are written to file */
    utils::parser::BTDIntervalsParser m_intervals;
    /** \brief Flush m_mf_output and particles to file.
     * Currently, a temporary customized output format for the buffer
     * data is implemented and called in this function.
     * After implementing OpenPMD and plotfile formats, this function
     * can be defined and implemented in Diagnostics.H, such that,
     * the function call to flush out buffer data for
     * FullDiagnostics and BTDiagnostics is the same */
    void Flush (int i_buffer, bool force_flush) override;
    /** whether to write output files at this time step
     *  The data is flushed when the buffer is full and/or
     *  when the simulation ends or when forced.
     * \param[in] step current time step
     * \param[in] i_buffer snapshot index
     * \param[in] force_flush if true, return true for any step
     *  The return bool is used to determine when to write buffer data out
     */
    bool DoDump (int step, int i_buffer, bool force_flush=false) override;
    /** whether to compute the back-transformed data and store buffer in this timestep
     *  The field-data is back-transformed from boosted-frame to lab-frame
     *  at every time step within the PIC loop. Back-transformation is not performed
     *  at initialization.
     * \param[in] step current time step, return true for any step >= 0
     * \param[in] force_flush if true, return true for any step
     */
    bool DoComputeAndPack (int step, bool force_flush=false) override;
    /** Initialize Data required to compute back-transformed diagnostics */
    void DerivedInitData () override;
    /** Initialize functors that store pointers to the fields requested by the user.
     *  Additionally, the cell-center functors that stores pointers to all fields,
     *  namely, Ex, Ey, Ez, Bx, By, Bz, jx, jy, jz, and rho is also initialized.
     * \param[in] lev level on which the vector of unique_ptrs to field functors
     *                is initialized.
     */
    void InitializeFieldFunctors (int lev) override;
    /** Initialize functors that store pointers to the fields in RZ requested by the user.
     *  Additionally, the cell-center functors that stores pointers to all fields,
     *  namely, Er, Et, Ez, Br, Bt, Bz, jr, jt, jz, and rho is also initialized.
     * \param[in] lev level on which the vector of unique_ptrs to field functors
     *                is initialized.
     */
    void InitializeFieldFunctorsRZopenPMD (int lev) override;
    /** Populating m_varnames with real and imaginary parts of each RZ mode.
     *  Modes are numbered from 0 to (nmodes-1) and mode 0 is purely real.
     *  Both m_cellcenter_varnames (storing cell-centered data) and
     *  m_varnames (storing back-transformed field data) are modified to include RZ modes to include
     *  field_modeid_real/imag. For example, for Er with two modes, the varnames are
     *  Er_0_real, Er_1_real, Er_1_imag
     * \param[in] field field-name
     * \param[in] ncomp number of rz components (if 2 modes, the ncomp is 2*nmodes-1)
     * \param[in] cellcenter_data if true, m_cellcenter_varnames are updated
                                  if false, m_varnames is updated
     */
    void AddRZModesToOutputNames (const std::string& field, int ncomp, bool cellcenter_data);
    /** This function allocates and initializes particle buffers for all the snapshots.
     * This is currently an empty function:
     * The particle containers required for this must be added to populate this function.
     */
    void InitializeParticleBuffer () override;
    /** This function prepares the current z coordinate in the boosted-frame and lab-frame as required by
        the particle and fields.
     */
    void PrepareBufferData () override;
    /** This function increments the buffer counter and identifies if the snapshot is fully populated */
    void UpdateBufferData () override;
    /** The cell-centered data for all fields, namely,
     *  Ex, Ey, Ez, Bx, By, Bz, jx, jy, jz, and rho is computed and stored in
     *  the multi-level cell-centered multifab, m_mf_cc. This MultiFab extends
     *  over the entire domain and is coarsened using the user-defined crse_ratio.
     *  For every lab-frame buffer, the data stored in this cell-centered MultiFab
     *  is sliced, back-transformed, and stored in the output multifab, m_mf_output.
     */
    void PrepareFieldDataForOutput () override;
    /** The Particle Geometry, BoxArray, and RealBox are set for the lab-frame output */
    void PrepareParticleDataForOutput() override;

    /** whether z-slice that corresponds to the buffer, i_buffer, is within the
     *  boosted-domain and lab-frame domains at level, lev.
     * \param[in] i_buffer index of the buffer for which the z-coordinates are checked
     *            Return true if the z co-ordinates stored in
     *            m_current_z_boost and m_current_z_lab for buffer, i_buffer,
     *            are within the bounds of the boosted-frame and lab-frame domain.
     * \param[in] lev the mesh-refinement level
     */
    bool GetZSliceInDomainFlag (int i_buffer, int lev);

    /** whether the k-index corresponding to the z-slice for the ith buffer, i_buffer,
     *  is within the bounds of the box in index-space
     * \param[in] i_buffer index of the buffer for which the k-index is checked.
     * \param[in] lev mesh-refinement level using which the k-index for the z-slice being filled in the ith buffer is computed
     *            Return true if the k-index of the z-slice is within the bounds of the box for the buffer being filled
     */
    bool GetKIndexInSnapshotBoxFlag (int i_buffer, int lev);

    /** Initialize buffer domain, buffer box and lab-frame parameters such as
     *  m_t_lab, and z-positions for the i^th snapshot, i_buffer, and level, lev.
     * \param[in] i_buffer i^th snapshot or buffer
     * \param[in] lev mesh-refinement level for which the field and/or particle buffer data is initialized.
     */
    void InitializeBufferData ( int i_buffer , int lev, bool restart=false) override;
    /** Whether to compute back-tranformed values for field-data.
     *  default value is true.
     */
    bool m_do_back_transformed_fields = true;
    /** Whether to compute back-tranformed values for particle-data
     *  default value is true.
     */
    bool m_do_back_transformed_particles = true;

    /** m_gamma_boost, is a copy of WarpX::gamma_boost
     *  That is, the Lorentz factor of the boosted frame in which the simulation is run.
     *  The direction for Lorentz transformation is assumed to be along
     *  ``warpx.boost_direction``, which is the same as the moving window direction.
     *  Currently, back-transformed diagnostics only works if the boost-direction and
     *  moving window direction are along the z-direction in Cartesian co-ordinates.
     */
    amrex::Real m_gamma_boost;
    amrex::Real m_beta_boost;
    /** m_moving_window_dir is a copy of WarpX::moving_window_dir
     *  Currently, back-transformed diagnostics only works if moving window is
     *  in z-direction for both 2D and 3D simulations in the Cartesian frame of reference.
     */
    int m_moving_window_dir;
    amrex::Real m_moving_window_beta;

    /** Number of back-transformed snapshots in the lab-frame requested by the user */
    int m_num_snapshots_lab = std::numeric_limits<int>::lowest();
    /** Time interval in lab-frame between the back-transformed snapshots */
    amrex::Real m_dt_snapshots_lab = std::numeric_limits<amrex::Real>::lowest();
    /** Distance between the back-transformed snapshots in lab-frame
      * m_dz_snapshots_lab = m_dt_snapshots_lab/c
      */
    amrex::Real m_dz_snapshots_lab = 0.0;

    /** Number of z-slices in each buffer of the snapshot */
    int m_buffer_size = 256;

    /** Vector of lab-frame time corresponding to each snapshot */
    amrex::Vector<amrex::Real> m_t_lab;
    /** Vector of physical region corresponding to the buffer that spans a part
     *  of the full back-transformed snapshot */
    amrex::Vector<amrex::RealBox> m_buffer_domain_lab;
    /** Vector of number of cells in the lab-frame for each back-transformed snapshot */
    amrex::Vector<amrex::IntVect> m_snapshot_ncells_lab;
    /** Vector of Box-dimension in boosted-frame index space
     *  for each back-transformed snapshot */
    amrex::Vector<amrex::Box> m_snapshot_box;
    /** Vector of Box-dimension in boosted-frame index space corresponding to the
     *  buffer that covers a part of the full backtransformed snapshot */
    amrex::Vector<amrex::Box> m_buffer_box;
    /** Vector of lab-frame z co-ordinate of each back-transformed snapshot
     *  at the current timestep */
    amrex::Vector<amrex::Real> m_current_z_lab;
    /** Vector of boosted-frame z co-ordinate corresponding to each back-transformed
        snapshot at the current timestep */
    amrex::Vector<amrex::Real> m_current_z_boost;
    /** Vector of previous boosted-frame z co-ordinate corresponding to each
        back-transformed snapshot*/
    amrex::Vector<amrex::Real> m_old_z_boost;
    /** Geometry that defines the domain attributes corresponding to the
     *  full snapshot in the back-transformed lab-frame.
     *  Specifically, the user-defined physical co-ordinates for the diagnostics
     *  is used to construct the geometry information for each snapshot at
     *  the respective levels. This geometry will be used to integrate all
     *  the buffers that were dumped out with partial field data
     *  in the back-transformed frame at a particle snapshot, t_lab.
     *  WriteToFile() for BTD will include this additional geometry information
     *  to guide the integration process.
     */
    amrex::Vector< amrex::Vector <amrex::Geometry> > m_geom_snapshot;
    /** Vector of counters tracking number of back-transformed z-slices filled
     *  in the output buffer multifab, m_mf_output, for each snapshot.
     *  When the buffer counter for a snapshot is equal to the maximum number
     *  of buffers (m_buffer_size), i.e., when the buffer is full, the data
     *  stored in the buffer multifab is flushed out and the counter is reset is zero.
     */
    amrex::Vector<int> m_buffer_counter;
    /** Vector of maximum number of buffer multifabs that need to be flushed to
     *  generate each lab-frame snapshot. At the final flush, m_buffer_flush_counter
     *  will be equal to the predicted m_max_buffer_multifabs for each snapshot.
     */
    amrex::Vector<int> m_max_buffer_multifabs;
    /** Vector of integers to indicate if the snapshot is full (0 not full, 1 for full).
     *  If the snapshot is full, then the snapshot files are closed.
     */
    amrex::Vector<int> m_snapshot_full;
    /** Vector of integers to store if the last valid slice in the lab-frame
     *  is being filled. When the value is 1, then the buffer is flushed, and
     *  m_snapshot_full is set to 1 for that snapshot index.
     */
    amrex::Vector<int> m_lastValidZSlice;
    /** Vector of counters tracking number of times the buffer of multifab is
     *  flushed out and emptied before being refilled again for each snapshot */
    amrex::Vector<int> m_buffer_flush_counter;
    /** Vector of k-indices in the hi-end along the moving window direction for the buffer being filled
     *  for each snapshot.
     */
    amrex::Vector<int> m_buffer_k_index_hi;
    /** Multi-level cell-centered multifab with all field-data components, namely,
     *  Ex, Ey, Ez, Bx, By, Bz, jx, jy, jz, and rho.
     *  This cell-centered data extending over the entire domain
     *  will be used by all snapshots to obtain lab-frame data at the respective
     *  z slice location.
     */
    std::string const m_cell_centered_data_name;
    /** Vector of pointers to compute cell-centered data, per level, per component
     *  using the coarsening-ratio provided by the user.
     */
    amrex::Vector< amrex::Vector <std::unique_ptr<ComputeDiagFunctor const> > > m_cell_center_functors;
    /** Define the cell-centered multi-component MultiFab at level, lev.
     *  This function is called when the buffer counter for a snapshot is zero,
     *  or when the buffer is empty. Checked using `buffer_empty(i_buffer)`
     *  \param[in] lev level at which the cell-centered MultiFab is defined.
     */
    void DefineCellCenteredMultiFab(int lev);
    /** Define the cell-centered multi-component output buffer MultiFab for
     *  snapshot, i_buffer, at level, lev
     *
     * \param[in] i_buffer buffer-id of the back-transformed snapshot
     * \param[in] lev      mesh-refinement level at which the output buffer MultiFab is defined
     */
    void DefineFieldBufferMultiFab (int i_buffer, int lev);

    /** Define the geometry object that spans the user-defined region for the
     *  ith snapshot, i_buffer, at level, lev.
     *
     * \param[in] i_buffer id of the back-transformed snapshot
     * \param[in] lev      level at which the geometry object is defined
     */
    void DefineSnapshotGeometry (int i_buffer, int lev);

    /** Compute and return z-position in the boosted-frame at the current timestep
      * \param[in] t_lab   lab-frame time of the snapshot
      * \param[in] t_boost boosted-frame time at level, lev
      */
    [[nodiscard]] amrex::Real UpdateCurrentZBoostCoordinate(amrex::Real t_lab, amrex::Real t_boost) const
    {
        const amrex::Real current_z_boost = (t_lab / m_gamma_boost - t_boost) * PhysConst::c / m_beta_boost;
        return current_z_boost;
    }
    /** Compute and return z-position in the lab-frame at the current timestep
      * \param[in] t_lab   lab-frame time of the snapshot
      * \param[in] t_boost boosted-frame time at level, lev
      */
    [[nodiscard]] amrex::Real UpdateCurrentZLabCoordinate(amrex::Real t_lab, amrex::Real t_boost) const
    {
        const amrex::Real current_z_lab = (t_lab - t_boost / m_gamma_boost ) * PhysConst::c / m_beta_boost;
        return current_z_lab;
    }
    /** Compute and return cell-size in z-dimension in the lab-frame at level, lev
     *  \param[in] dt         timestep in boosted-frame at level, lev
     *  \param[in] ref_ratio  refinement ratio in the z-direction at level, lev-1.
     *             The ref-ratio in the z-direction for single-level diagnostics is 1.
     */
    [[nodiscard]] amrex::Real dz_lab (amrex::Real dt, amrex::Real ref_ratio) const;
    /** Compute k-index corresponding to current lab-frame z co-ordinate (m_current_z_lab)
     *  for the ith buffer i_buffer, and at level, lev.
     * \param[in] i_buffer snapshot index
     * \param[in] lev      mesh-refinement level at which the lab-frame z-index is computed
     */
    [[nodiscard]] int k_index_zlab (int i_buffer, int lev) const;
    /** whether field buffer is full
     * \param[in] i_buffer buffer id for which the buffer size is checked.
     * returns bool = true is buffer is full, that is,
               when buffer counter is equal to m_buffer_size
     */
    bool buffer_full (int i_buffer) {
        return (k_index_zlab(i_buffer,0) == m_buffer_box[i_buffer].smallEnd(m_moving_window_dir));
    }

    /** whether field buffer is empty.
     * \param[in] i_buffer buffer id for which the buffer size is checked.
     * returns bool = true is buffer is empty i.e., when buffer counter is zero.
     */
    bool buffer_empty (int i_buffer) {
        return ( m_buffer_counter[i_buffer] == 0) ;
    }

    /** Vector of integers to indicate if is the first flush for a snapshot so that appropriate meta-data and geometry is written (0 if not, 1 if yes). */
    amrex::Vector<int> m_first_flush_after_restart;

    /** Resets integer for first flush to 0 for the ith snapshot (i_buffer).
     *  This function is called after flush the first buffer after restarting the simulation
     */
    void NullifyFirstFlush(int i_buffer) {m_first_flush_after_restart[i_buffer] = 0;}

    /** Vector of integers to indicate if the snapshot geometry is defined. */
    amrex::Vector<int> m_snapshot_geometry_defined;
    /** Vector of integers to indicate if the field buffer multifab, m_mf_output, is defined in DefineFieldBufferMultifab() function */
    amrex::Vector<int> m_field_buffer_multifab_defined;

    /** Reset buffer counter to zero.
     * \param[in] i_buffer snapshot index for which the counter is set to zero.
     */
    void ResetBufferCounter(int i_buffer) {
        m_buffer_counter[i_buffer] = 0;
    }
    /** Increment buffer counter by one when the buffer of a snapshot has been flushed.
     * \param[in] i_buffer snapshot index for which the counter is incremented.
     */
    void IncrementBufferFlushCounter(int i_buffer) {
        m_buffer_flush_counter[i_buffer]++;
    }
    /** Set Snapshot full status to 1 if the last valid zslice, m_lastValidZSlice,
     *  for the ith snapshot index, given by, i_buffer, is filled.
     *
     *  \param[in] i_buffer snapshot index
     */
    void SetSnapshotFullStatus (int i_buffer);
    /** Vector of field-data stored in the cell-centered MultiFab.
     *  All the fields are stored regardless of the specific fields to plot selected
     *  by the user.
     */
#ifdef WARPX_DIM_RZ
    amrex::Vector< std::string > m_cellcenter_varnames = {"Er", "Et", "Ez",
                                                          "Br", "Bt", "Bz",
                                                          "jr", "jt", "jz", "rho"};
    amrex::Vector< std::string > m_cellcenter_varnames_fields = {"Er", "Et", "Ez",
                                                                 "Br", "Bt", "Bz",
                                                                 "jr", "jt", "jz",
                                                                 "rho"};
#else
    amrex::Vector< std::string > m_cellcenter_varnames = {"Ex", "Ey", "Ez",
                                                          "Bx", "By", "Bz",
                                                          "jx", "jy", "jz", "rho"};
#endif


    /** Merge the lab-frame buffer multifabs so it can be visualized as
     *  a single plotfile
     */
    void MergeBuffersForPlotfile (int i_snapshot);
    /** Interleave lab-frame meta-data of the buffers to be consistent
     *  with the merged plotfile lab-frame data.
     */
    void InterleaveBufferAndSnapshotHeader (const std::string& buffer_Header,
                                            const std::string& snapshot_Header);
    /** Interleave meta-data of the buffer multifabs to be consistent
     *  with the merged plotfile lab-frame data.
     */
    void InterleaveFabArrayHeader (const std::string& Buffer_FabHeader_path,
                                  const std::string& snapshot_FabHeader_path,
                                  const std::string& newsnapshot_FabFilename);
    /** Interleave lab-frame metadata of the species header file in the buffers to
     *  be consistent with the merged plotfile lab-frame data
     */
    void InterleaveSpeciesHeader(const std::string& buffer_species_Header_path,
                                 const std::string& snapshot_species_Header_path,
                                 const std::string& species_name, int new_data_index);

    /** Interleave lab-frame metadata of the particle header file in the buffers to
     *  be consistent with the merged plotfile lab-frame data
     */
    void InterleaveParticleDataHeader(const std::string& buffer_ParticleHdrFilename,
                                      const std::string& snapshot_ParticleHdrFilename);
    /** Initialize particle functors for each species to compute the back-transformed
        lab-frame data. */
    void InitializeParticleFunctors () override;

    /** Reset total number of particles in the particle buffer to 0 for ith snapshot */
    void ResetTotalParticlesInBuffer(int i_buffer);
    /** Clear particle data stored in the particle buffer */
    void ClearParticleBuffer(int i_buffer);
    /** Redistributes particles to the buffer box array in the lab-frame */
    void RedistributeParticleBuffer (int i_buffer);
    void UpdateVarnamesForRZopenPMD();

    amrex::Real gettlab (int i_buffer) override {return m_t_lab[i_buffer];}
    void settlab (int i_buffer, amrex::Real tlab) override {m_t_lab[i_buffer] = tlab; }
    int get_buffer_k_index_hi (int i_buffer) override {return m_buffer_k_index_hi[i_buffer]; }
    void set_buffer_k_index_hi (int i_buffer, int kindex) override {m_buffer_k_index_hi[i_buffer] = kindex;}
    amrex::Real get_snapshot_domain_lo (int i_buffer, int idim) override {return m_snapshot_domain_lab[i_buffer].lo(idim); }
    amrex::Real get_snapshot_domain_hi (int i_buffer, int idim) override {return m_snapshot_domain_lab[i_buffer].hi(idim); }
    int get_flush_counter (int i_buffer) override {return m_buffer_flush_counter[i_buffer]; }
    void set_flush_counter ([[maybe_unused]] int i_buffer, int flush_counter) override { m_buffer_flush_counter[i_buffer] = flush_counter; }
    int get_last_valid_Zslice ( int i_buffer) override { return m_lastValidZSlice[i_buffer]; }
    void set_last_valid_Zslice ( int i_buffer, int last_valid_Zslice) override { m_lastValidZSlice[i_buffer] = last_valid_Zslice; }
    int get_snapshot_full_flag ( int i_buffer) override { return m_snapshot_full[i_buffer]; }
    void set_snapshot_full ( int i_buffer, int snapshot_full) override { m_snapshot_full[i_buffer] = snapshot_full; }
};
#endif // WARPX_BTDIAGNOSTICS_H_

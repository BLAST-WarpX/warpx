#ifndef WARPX_FULLDIAGNOSTICS_H_
#define WARPX_FULLDIAGNOSTICS_H_

#include "Diagnostics.H"
#include "Utils/Parser/IntervalsParser.H"

#include <AMReX_REAL.H>

#include <string>

class FullDiagnostics final : public Diagnostics
{
public:
    FullDiagnostics (int i, const std::string& name, DiagTypes diag_type);
    /** Type of time averaging for diagnostics (fields only)
     * None corresponds to instantaneous diags
     * Static corresponds to a fixed starting step for averaging,
     *     will average until the end, and dump out intermediate average results
     * Dynamic corresponds to a moving period for averaging where the start step
     *     is as many steps before the output interval as the averaging period is long.
     */
    enum struct TimeAverageType {None, Static, Dynamic};
private:
    /** Read user-requested parameters for full diagnostics */
    void ReadParameters ();
    /** Determines timesteps at which full diagnostics are written to file */
    utils::parser::IntervalsParser m_intervals;
    /** Whether to plot raw (i.e., NOT cell-centered) fields */
    bool m_plot_raw_fields = false;
    /** Whether to plot guard cells of raw fields */
    bool m_plot_raw_fields_guards = false;
    /** Whether to dump the RZ modes */
    bool m_dump_rz_modes = false;
    /** Whether the field solver deposits current density, if not it will be done
     * before writing the diagnostic.
     */
    bool m_solver_deposits_current = true;
    /** Whether the diagnostics are averaging data over time or not */
    TimeAverageType m_time_average_mode = TimeAverageType::None;
    /** Period to average fields over: in steps */
    int m_average_period_steps = -1;
    /** Period to average fields over: in seconds */
    amrex::Real m_average_period_time = -1.0;
    /** Time step to start averaging */
    int m_average_start_step = -1;
    /** Flush m_mf_output or m_sum_mf_output and particles to file for the i^th buffer */
    void Flush (int i_buffer, bool /* force_flush */) override;
    /** Flush raw data */
    void FlushRaw ();
    /** Initialize Data required to compute TimeAveraged diagnostics */
    void DerivedInitData () override;
    /** whether to compute and pack cell-centered data in m_mf_output
     * \param[in] step current time step
     * \param[in] force_flush if true, return true for any step since output must be
                  computed in this case
     * \return bool, whether to flush
     */
    bool DoComputeAndPack (int step, bool force_flush=false) override;
    /** whether to flush at this time step
     * \param[in] step current time step
     * \param[in] i_buffer index of a back-transformed snapshot
     * \param[in] force_flush if true, return true for any step
     * \return bool, whether to flush
     */
    bool DoDump (int step, int i_buffer, bool force_flush=false) override;
    /** Append varnames with names for all modes of a field
     * \param[in] field field name (includes component, e.g., Er)
     * \param[in] ncomp number of components (modes, real and imag)
     */
    void AddRZModesToOutputNames (const std::string& field, int ncomp);
    /** Define the cell-centered multifab m_mf_output depending on user-defined
      * lo and hi and coarsening ratio. This MultiFab may have a different BoxArray and
      * DistributionMap than field MultiFabs in the simulation.
      *
      * \param[in] i_buffer index of a back-transformed snapshot
      * \param[in] lev level on which source multifabs are defined
      */
    void InitializeBufferData ( int i_buffer, int lev, bool restart=false ) override;
    /** Initialize functors that store pointers to the RZ fields requested by the user.
     * \param[in] lev level on which the vector of unique_ptrs to field functors is initialized.
     */
    void InitializeFieldFunctorsRZopenPMD (int lev) override;
    /** Append m_all_field_functors[lev] with all modes of all components of E B and j
     * and add name to varnames.
     * \param[in] lev level on which source multifabs are defined.
     */
    void AddRZModesToDiags (int lev);
    /** Initialize functors that store pointers to the fields requested by the user.
      * \param[in] lev level on which the vector of unique_ptrs to field functors is initialized.
      */
    void InitializeFieldFunctors (int lev) override;
    void InitializeParticleBuffer () override;
    /** Prepare field data to be used for diagnostics */
    void PrepareFieldDataForOutput () override;
    /** Prepare particle data to be used for diagnostics. */
    void PrepareParticleDataForOutput() override {}
    /** Update the physical extent of the diagnostic domain for moving window and
     *  galilean shift simulations
     *
     * \param[in] step current time step
     */
    void MovingWindowAndGalileanDomainShift (int step) override;
    /** This function queries deprecated input parameters and abort
     *  the run if one of them is specified.
     */
    void BackwardCompatibility ();
};

#endif // WARPX_FULLDIAGNOSTICS_H_

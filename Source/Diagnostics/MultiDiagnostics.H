#ifndef WARPX_MULTIDIAGNOSTICS_H_
#define WARPX_MULTIDIAGNOSTICS_H_

#include "Diagnostics.H"

#include "MultiDiagnostics_fwd.H"

#include <AMReX_Vector.H>

#include <memory>
#include <string>
#include <vector>


/**
 * \brief This class contains a vector of all diagnostics in the simulation.
 */
class MultiDiagnostics
{
public:
    MultiDiagnostics ();
    /** \brief Read Input parameters. Called in constructor. */
    void ReadParameters ();
    /** \brief Loop over diags in alldiags and call their InitDiags */
    void InitData ();
    /** \brief Called at each iteration. Compute diags and flush. */
    void FilterComputePackFlush (int step, bool force_flush=false, bool BackTransform=false);
    /** \brief Called only at the last iteration. Loop over each diag and if m_dump_last_timestep
     *         is true, compute diags and flush with force_flush=true. */
    void FilterComputePackFlushLastTimestep (int step);
    /** \brief Loop over diags in all diags and call their InitializeFieldFunctors.
               Called when a new partitioning is generated at level, lev.
      * \param[in] lev level at this the field functors are initialized.
      */
    void InitializeFieldFunctors (int lev);
    /** Start a new iteration, i.e., dump has not been done yet. */
    void NewIteration ();
    Diagnostics& GetDiag(int idiag) {return *alldiags[idiag]; }
    [[nodiscard]] int GetTotalDiags() const {return ndiags;}
    DiagTypes diagstypes(int idiag) {return diags_types[idiag];}
private:
    /** Vector of pointers to all diagnostics */
    amrex::Vector<std::unique_ptr<Diagnostics> > alldiags;
    int ndiags = 0; /**< number of different diagnostics */
    std::vector<std::string> diags_names;
    /**Type of each diagnostics*/
    std::vector<DiagTypes> diags_types;
};

#endif // WARPX_MULTIDIAGNOSTICS_H_

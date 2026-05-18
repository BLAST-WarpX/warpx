/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_

#include "MultiReducedDiags_fwd.H"

#include "ReducedDiags.H"

#include <memory>
#include <string>
#include <vector>

/**
 * This class holds multiple instances of ReducedDiagnostics, and contains
 * general functions to initialize, compute, and write these diagnostics
 * to file.
 */
class MultiReducedDiags
{
public:

    /// Bool: whether or not reduced diagnostics are activated
    int m_plot_rd = 0;

    /// names of reduced diagnostics
    std::vector<std::string> m_rd_names;

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::unique_ptr<ReducedDiags>> m_multi_rd;

    /// constructor
    MultiReducedDiags ();

    /** Loop over all ReducedDiags and call their InitData
     */
    void InitData ();

    /** Redistribute parallel data structures during load balance
     */
    void LoadBalance ();

    /** Loop over all ReducedDiags and call their ComputeDiags
     *  @param[in] step current iteration time */
    void ComputeDiags (int step);

    /** Loop over all ReducedDiags and call their ComputeDiagsMidStep
     *  @param[in] step current iteration time */
    void ComputeDiagsMidStep (int step);

    /** Loop over all ReducedDiags and call their WriteToFile
     *  @param[in] step current iteration time */
    void WriteToFile (int step);

    /** Check if any diagnostics will be done */
    bool DoDiags(int step);

    /** \brief Loop over all ReducedDiags and call their WriteCheckpointData
     *  @param[in] dir checkpoint directory */
    void WriteCheckpointData (std::string const & dir);

    /** \brief Loop over all ReducedDiags and call their ReadCheckpointData
     *  @param[in] dir checkpoint directory */
    void ReadCheckpointData (std::string const & dir);
};

#endif

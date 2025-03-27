/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYDIAGS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYDIAGS_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief Reduced diagnostic for tracking memory usage from AMReX memory arenas.
 *
 * This diagnostic uses AMReX::Arena::PrintUsageToFiles to output detailed memory
 * usage information including:
 * - Total/free GPU memory when applicable
 * - Main/device/managed/pinned/comms arena memory usage
 * - Per-rank memory statistics
 */
class MemoryPerRank : public ReducedDiags
{
public:
    /**
     * \brief Constructor.
     *
     * \param[in] rd_name name of the reduced diagnostic
     */
    explicit MemoryPerRank (const std::string& rd_name);

    /**
     * \brief This function computes memory diagnostics by calling
     * AMReX::Arena::PrintUsageToFiles at the specified intervals.
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) override;

private:
    // The base filename for memory usage output files
    std::string m_filename;

};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYDIAGS_H_


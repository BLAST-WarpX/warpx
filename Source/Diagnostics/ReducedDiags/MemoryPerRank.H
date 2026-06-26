/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYPERRANK_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYPERRANK_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief Reduced diagnostic for tracking memory usage on each MPI rank.
 *
 * This diagnostic uses amrex::Arena::PrintUsageToFiles to write detailed
 * memory usage information, one text file per MPI rank, and appends extra
 * context that is useful when debugging out-of-memory (OOM) errors, such as:
 * - Total/free GPU memory (when applicable)
 * - Main/device/managed/pinned arena memory usage
 * - Host process memory (VmRSS/VmHWM/VmPeak/VmSize on Linux)
 * - MPI rank, host name and GPU device id
 *
 * The main use case is to inspect the state of a large, multi-GPU run
 * (typically from a checkpoint) right before an OOM crash, without
 * otherwise modifying the input file.
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
     * \brief Compute memory diagnostics at the specified intervals by calling
     * amrex::Arena::PrintUsageToFiles and appending additional per-rank
     * information (process memory, host name, GPU device id).
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) override;

private:
    //! Basename for per-rank files (the final file name is
    //! ``<m_path><m_filename>.<MPI rank>``).
    std::string m_filename;

    //! Write a file only from every ``m_rank_stride``-th MPI rank.
    //! The default ``1`` writes one file per rank; on very large runs
    //! setting this to e.g. ``8`` reduces the number of files by 8x.
    int m_rank_stride = 1;

    //! Whether this rank participates in writing (derived from
    //! ``m_rank_stride`` in the constructor).
    bool m_rank_participates = true;

};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_MEMORYPERRANK_H_

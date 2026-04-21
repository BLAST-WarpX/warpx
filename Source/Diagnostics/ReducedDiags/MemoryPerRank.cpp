/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MemoryPerRank.H"

#include "WarpX.H"

#include <AMReX_Arena.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_MPI
#   include <mpi.h>
#endif

#ifdef __linux__
#   include <unistd.h>
#endif

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>


namespace {

#ifdef __linux__
    /** Read a named field (e.g. "VmRSS") from /proc/self/status.
     *  Returns the trimmed value (including unit) or an empty string on failure.
     */
    std::string ReadProcStatusField (const std::string& key)
    {
        std::ifstream ifs("/proc/self/status");
        if (!ifs.is_open()) { return {}; }
        std::string line;
        const std::string prefix = key + ":";
        while (std::getline(ifs, line)) {
            if (line.rfind(prefix, 0) == 0) {
                const auto pos = line.find_first_not_of(" \t", prefix.size());
                return (pos == std::string::npos) ? std::string{} : line.substr(pos);
            }
        }
        return {};
    }
#endif

    /** Get a human-readable host name (MPI processor name if available,
     *  otherwise gethostname on Linux, otherwise "unknown").
     */
    std::string GetHostName ()
    {
#ifdef AMREX_USE_MPI
        char hostname[MPI_MAX_PROCESSOR_NAME] = {0};
        int length = 0;
        if (MPI_Get_processor_name(hostname, &length) == MPI_SUCCESS) {
            return {hostname, static_cast<std::size_t>(length)};
        }
#elif defined(__linux__)
        char hostname[256] = {0};
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            hostname[sizeof(hostname)-1] = '\0';
            return {hostname};
        }
#endif
        return {"unknown"};
    }

} // namespace

// Constructor
MemoryPerRank::MemoryPerRank (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
    // Default basename for the per-rank files (the MPI rank is appended by
    // amrex::Arena::PrintUsageToFiles, giving "<m_path><m_filename>.<rank>").
    m_filename = m_rd_name;

    // Read per-diagnostic parameters
    const amrex::ParmParse pp_name(m_rd_name);

    // Optional custom basename
    pp_name.query("file_prefix", m_filename);

    // Optional: only every m_rank_stride-th MPI rank writes a file.
    // This is useful on very large runs where one file per rank would
    // create too many files on the shared filesystem.
    pp_name.query("rank_stride", m_rank_stride);
    if (m_rank_stride < 1) { m_rank_stride = 1; }
    m_rank_participates =
        (amrex::ParallelDescriptor::MyProc() % m_rank_stride == 0);

    // We do not use ReducedDiags::WriteToFile: per-rank files are written
    // directly from ComputeDiags via amrex::Arena::PrintUsageToFiles, and
    // additional per-rank information is appended there.
    m_write_header = false;

    // Note: the output directory (m_path) is already created by the
    // ReducedDiags base class constructor on the IOProcessor.
}

// Compute diagnostics - writes one file per participating MPI rank
void MemoryPerRank::ComputeDiags (int step)
{
    // Only proceed on requested intervals
    if (!m_intervals.contains(step+1)) { return; }

    // Optionally thin out the set of ranks that write a file
    if (!m_rank_participates) { return; }

    // Header line printed at the top of each block
    std::stringstream time_ss;
    time_ss << std::scientific << std::setprecision(14)
            << WarpX::GetInstance().gett_new(0);
    const std::string message =
        "Memory usage at step " + std::to_string(step+1) +
        ", time = " + time_ss.str();

    // Let AMReX print arena usage for this rank into <file_basename>.<rank>
    const std::string file_basename = m_path + m_filename;
    amrex::Arena::PrintUsageToFiles(file_basename, message);

    // Append additional per-rank context that AMReX does not emit
    const std::string per_rank_file =
        file_basename + "." +
        std::to_string(amrex::ParallelDescriptor::MyProc());
    std::ofstream ofs(per_rank_file, std::ios_base::app);
    if (!ofs.is_open()) { return; }

    ofs << "    [Rank] MPI rank: " << amrex::ParallelDescriptor::MyProc()
        << " / " << amrex::ParallelDescriptor::NProcs() << "\n";
    ofs << "    [Rank] hostname: " << GetHostName() << "\n";
#ifdef AMREX_USE_GPU
    ofs << "    [Rank] GPU device id: "
        << amrex::Gpu::Device::deviceId() << "\n";
#endif

#ifdef __linux__
    // Host-process memory (in kB, as reported by the kernel). This captures
    // allocations that are not tracked by AMReX arenas (MPI buffers, I/O
    // libraries, Python, plugin libraries, ...) and is often the first
    // thing that matters when chasing OOM errors.
    for (const auto& field : {"VmPeak", "VmSize", "VmHWM", "VmRSS"}) {
        const std::string v = ReadProcStatusField(field);
        if (!v.empty()) {
            ofs << "    [Process] " << field << ": " << v << "\n";
        }
    }
#endif

    ofs << "\n";
}

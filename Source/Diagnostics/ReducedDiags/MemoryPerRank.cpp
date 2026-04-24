/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MemoryPerRank.H"

#include "WarpX.H"

#include <AMReX_Arena.H>
#include <AMReX_CArena.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_MPI
#   include <mpi.h>
#endif

#ifdef __linux__
#   include <unistd.h>
#endif

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <utility>


namespace {

#ifdef __linux__
    /** Read a named field (e.g. "VmRSS") from /proc/self/status and return its
     *  numeric part as an integer kB value, or -1 on failure. All Vm* fields in
     *  /proc/self/status are reported in kB with a trailing " kB" unit.
     */
    long ReadProcStatusKB (const std::string& key)
    {
        std::ifstream ifs("/proc/self/status");
        if (!ifs.is_open()) { return -1; }
        std::string line;
        const std::string prefix = key + ":";
        while (std::getline(ifs, line)) {
            if (line.rfind(prefix, 0) == 0) {
                const auto pos = line.find_first_not_of(" \t", prefix.size());
                if (pos == std::string::npos) { return -1; }
                try {
                    return std::stol(line.substr(pos));
                } catch (...) {
                    return -1;
                }
            }
        }
        return -1;
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

    /** Width (in decimal digits) required to represent any rank in [0, nprocs-1].
     *  Uses a minimum width of 4 so small-rank runs still produce constant-length
     *  filenames like MPR.0000.yaml / MPR.0001.yaml.
     */
    int RankFilenameWidth (int nprocs)
    {
        int width = 1;
        for (int n = std::max(nprocs - 1, 0); n >= 10; n /= 10) { ++width; }
        return std::max(width, 4);
    }

    /** Bytes → MB (float). Used throughout the YAML output so every memory
     *  quantity is reported in the same unit at the same precision.
     */
    constexpr double bytes_per_mib = 1024.0 * 1024.0;
    double BytesToMb (std::size_t bytes)
    {
        return static_cast<double>(bytes) / bytes_per_mib;
    }

    /** Emit one YAML "arena" block for a CArena-derived pool. No-op for non-CArena
     *  (e.g. BArena) since those do not track usage. Numbers are written as
     *  floating-point MB with 3-decimal precision — the caller is expected to
     *  have set `std::fixed << std::setprecision(3)` on the stream already.
     */
    void EmitArenaBlock (std::ofstream& ofs, const std::string& key, amrex::Arena* arena)
    {
        if (arena == nullptr) { return; }
        auto* const carena = dynamic_cast<amrex::CArena*>(arena);
        if (carena == nullptr) { return; }
        ofs << "  " << key << ":\n";
        ofs << "    allocated_mb: "
            << BytesToMb(carena->heap_space_used()) << "\n";
        ofs << "    used_mb: "
            << BytesToMb(carena->heap_space_actually_used()) << "\n";
    }

} // namespace

// Constructor
MemoryPerRank::MemoryPerRank (const std::string& rd_name)
    : ReducedDiags{rd_name},
      // Default basename for the per-rank YAML files (the MPI rank is appended
      // with zero padding, giving "<m_path><m_filename>.<rank>.yaml").
      // m_rd_name is initialized by the base ReducedDiags ctor above and is
      // already usable here.
      m_filename{m_rd_name}
{
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

    // We do not use ReducedDiags::WriteToFile: per-rank YAML files are written
    // directly from ComputeDiags below.
    m_write_header = false;

    // Note: the output directory (m_path) is already created by the
    // ReducedDiags base class constructor on the IOProcessor.
}

// Compute diagnostics - writes/appends one YAML document per participating MPI
// rank into <m_path><m_filename>.<zero-padded-rank>.yaml.
void MemoryPerRank::ComputeDiags (int step)
{
    // Only proceed on requested intervals
    if (!m_intervals.contains(step+1)) { return; }

    // Optionally thin out the set of ranks that write a file
    if (!m_rank_participates) { return; }

    // Build per-rank filename with a zero-padded rank so every file has the
    // same name length in a given run (easier globbing / sorting).
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    const int myproc = amrex::ParallelDescriptor::MyProc();
    const int width = RankFilenameWidth(nprocs);
    std::ostringstream rank_ss;
    rank_ss << std::setw(width) << std::setfill('0') << myproc;

    const std::string per_rank_file =
        m_path + m_filename + "." + rank_ss.str() + ".yaml";

    std::ofstream ofs(per_rank_file, std::ios_base::app);
    if (!ofs.is_open()) { return; }

    // Start a new YAML document for this snapshot. Every snapshot in a given
    // file is a self-contained dict; readers can use e.g. PyYAML's
    // `yaml.safe_load_all()` to stream them.
    ofs << "---\n";

    ofs << "step: " << (step+1) << "\n";

    // Time in scientific notation (YAML accepts this as a native float).
    ofs << "time: "
        << std::scientific << std::setprecision(14)
        << WarpX::GetInstance().gett_new(0) << "\n";
    ofs.unsetf(std::ios_base::floatfield);

    ofs << "mpi:\n";
    ofs << "  rank: " << myproc << "\n";
    ofs << "  size: " << nprocs << "\n";

    // Host name is single-quoted to be safe against characters (e.g. ':') that
    // would otherwise need YAML escaping.
    ofs << "host:\n";
    ofs << "  name: '" << GetHostName() << "'\n";

    // From here on every memory value is emitted as floating-point MB with
    // 3-decimal precision. Integer fields (rank, step, device_id, ...) are
    // unaffected by `std::fixed`/`setprecision`.
    ofs << std::fixed << std::setprecision(3);

#ifdef AMREX_USE_GPU
    ofs << "gpu:\n";
    ofs << "  device_id: " << amrex::Gpu::Device::deviceId() << "\n";
    ofs << "  total_mb: "
        << BytesToMb(amrex::Gpu::Device::totalGlobalMem()) << "\n";
    ofs << "  free_mb: "
        << BytesToMb(amrex::Gpu::Device::freeMemAvailable()) << "\n";
#endif

    // AMReX arena usage. We walk the arenas ourselves (rather than using
    // amrex::Arena::PrintUsageToFiles) so we can emit structured YAML. We match
    // AMReX's "only emit when distinct" logic so aliased arenas (e.g. Comms
    // aliased to Device/Pinned) do not appear twice.
    ofs << "arenas:\n";
    EmitArenaBlock(ofs, "main", amrex::The_Arena());
    if (amrex::The_Device_Arena() != nullptr &&
        amrex::The_Device_Arena() != amrex::The_Arena())
    {
        EmitArenaBlock(ofs, "device", amrex::The_Device_Arena());
    }
    if (amrex::The_Managed_Arena() != nullptr &&
        amrex::The_Managed_Arena() != amrex::The_Arena())
    {
        EmitArenaBlock(ofs, "managed", amrex::The_Managed_Arena());
    }
    if (amrex::The_Pinned_Arena() != nullptr) {
        EmitArenaBlock(ofs, "pinned", amrex::The_Pinned_Arena());
    }
    if (amrex::The_Comms_Arena() != nullptr &&
        amrex::The_Comms_Arena() != amrex::The_Device_Arena() &&
        amrex::The_Comms_Arena() != amrex::The_Pinned_Arena())
    {
        EmitArenaBlock(ofs, "comms", amrex::The_Comms_Arena());
    }

#ifdef __linux__
    // Host-process memory as reported by the kernel. The kernel emits kB in
    // /proc/self/status; we convert to MB to keep the YAML unit-consistent
    // with the arena and GPU values. This captures allocations that are not
    // tracked by AMReX arenas (MPI buffers, I/O libraries, Python, plugin
    // libraries, ...) and is often the first thing that matters when chasing
    // OOM errors.
    const std::pair<const char*, const char*> vm_fields[] = {
        {"VmPeak", "vm_peak_mb"},
        {"VmSize", "vm_size_mb"},
        {"VmHWM",  "vm_hwm_mb"},
        {"VmRSS",  "vm_rss_mb"},
    };
    ofs << "process:\n";
    for (const auto& [field, yaml_key] : vm_fields) {
        const long kb = ReadProcStatusKB(field);
        if (kb >= 0) {
            ofs << "  " << yaml_key << ": "
                << (static_cast<double>(kb) / 1024.0) << "\n";
        }
    }
#endif

    ofs.flush();
}

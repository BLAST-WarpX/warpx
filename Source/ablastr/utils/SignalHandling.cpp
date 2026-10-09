/* Copyright 2022 Philip Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SignalHandling.H"
#include "TextMsg.H"

#include <AMReX_Arena.H>
#include <AMReX_CArena.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_IParser.H>
#if defined(AMREX_TINY_PROFILING)
#   include <AMReX_TinyProfiler.H>
#endif

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

// For sigaction(), pipe() et al.
#if defined(__linux__) || defined(__APPLE__)
#   include <csignal>
#   include <fcntl.h>
#   include <unistd.h>
#endif
#if defined(__APPLE__)
#   include <mach/mach.h>
#endif

namespace ablastr::utils {

bool SignalHandling::m_any_signal_action_active = false;
std::atomic<bool> SignalHandling::signal_received_flags[NUM_SIGNALS];
bool SignalHandling::signal_conf_requests[SIGNAL_REQUESTS_SIZE][NUM_SIGNALS];
bool SignalHandling::signal_actions_requested[SIGNAL_REQUESTS_SIZE];
#if defined(AMREX_USE_MPI)
MPI_Request SignalHandling::signal_mpi_ibcast_request;
#endif
std::atomic<int> SignalHandling::m_current_step{-1};
std::atomic<int> SignalHandling::m_last_completed_step{-1};
std::atomic<int> SignalHandling::m_status_pipe_write_fd{-1};
std::mutex SignalHandling::m_status_mutex;
bool SignalHandling::m_status_enabled = false;
int SignalHandling::m_rank = 0;
int SignalHandling::m_device_id = 0;

// Only lock-free atomics are async-signal-safe
static_assert(std::atomic<bool>::is_always_lock_free);
static_assert(std::atomic<int>::is_always_lock_free);

namespace {

    /** Innermost active AMReX TinyProfiler section of this process
     *
     * Note: this reads the profiler call stack of the main thread from another
     * thread without synchronization, so the result is a best-effort snapshot.
     */
    std::string current_profiler_section ()
    {
#if defined(AMREX_TINY_PROFILING)
        std::ostringstream os;
        amrex::TinyProfiler::PrintCallStack(os);

        // Output is a header line, followed by one line per section, innermost last.
        // Nothing is printed if the profiler is disabled at runtime.
        std::istringstream is(os.str());
        std::string line;
        if (!std::getline(is, line)) { return "n/a (tiny profiler disabled)"; }

        std::string innermost = "none";
        while (std::getline(is, line)) {
            if (!line.empty()) { innermost = line; }
        }
        return innermost;
#else
        return "n/a (requires AMReX tiny profiler)";
#endif
    }

    //! Resident set size (RSS) of this process in bytes, if available
    std::optional<std::size_t> cpu_resident_memory ()
    {
#if defined(__linux__)
        // Fields in pages: total program size, resident set size, ...
        std::ifstream statm("/proc/self/statm");
        std::size_t size_pages = 0, resident_pages = 0;
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (!(statm >> size_pages >> resident_pages) || page_size <= 0) { return std::nullopt; }
        return resident_pages * static_cast<std::size_t>(page_size);
#elif defined(__APPLE__)
        mach_task_basic_info info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(info.resident_size);
#else
        return std::nullopt;
#endif
    }

#if defined(AMREX_USE_GPU)
    //! Used and total memory of the GPU of this process in bytes, as reported by the driver
    std::optional<std::pair<std::size_t, std::size_t>> gpu_device_memory ()
    {
        // no AMREX_*_SAFE_CALL: a status report must not abort the simulation
#if defined(AMREX_USE_CUDA)
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { return std::nullopt; }
        return std::make_pair(total_bytes - free_bytes, total_bytes);
#elif defined(AMREX_USE_HIP)
        std::size_t free_bytes = 0, total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) { return std::nullopt; }
        return std::make_pair(total_bytes - free_bytes, total_bytes);
#elif defined(AMREX_USE_SYCL) && defined(__INTEL_LLVM_COMPILER)
        auto const& device = amrex::Gpu::Device::syclDevice();
        if (!device.has(sycl::aspect::ext_intel_free_memory)) { return std::nullopt; }
        const std::size_t free_bytes =
            device.get_info<sycl::ext::intel::info::device::free_memory>();
        const std::size_t total_bytes = amrex::Gpu::Device::totalGlobalMem();
        return std::make_pair(total_bytes - free_bytes, total_bytes);
#else
        // SYCL without the Intel free memory extension
        return std::nullopt;
#endif
    }

    /** Bytes in use in The_Arena, if it is an AMReX CArena
     *
     * In GPU builds, The_Arena allocates (by default) a large pool of device memory,
     * so the driver-reported device memory does not reflect the actual usage.
     * Note: this is read without the lock of the arena.
     */
    std::optional<std::size_t> gpu_arena_memory ()
    {
        if (!amrex::Arena::IsInitialized()) { return std::nullopt; }
        auto const* arena = dynamic_cast<amrex::CArena const*>(amrex::The_Arena());
        if (arena == nullptr) { return std::nullopt; }
        return arena->heap_space_actually_used();
    }
#endif

    void print_gib (std::ostream& os, std::optional<std::size_t> bytes)
    {
        if (bytes.has_value()) {
            os << std::fixed << std::setprecision(3)
               << static_cast<double>(*bytes) / (1024.0 * 1024.0 * 1024.0) << "GiB";
        } else {
            os << "n/a";
        }
    }

} // namespace

int
SignalHandling::parseSignalNameToNumber (const std::string &str)
{
    amrex::IParser signals_parser(str);

#if defined(__linux__) || defined(__APPLE__)
    const struct {
        const char* abbrev;
        int value;
    } signals_to_parse[] = {
        {"ABRT", SIGABRT},
        {"ALRM", SIGALRM},
        {"BUS", SIGBUS},
        {"CHLD", SIGCHLD},
        {"CLD", SIGCHLD}, // Synonymous to SIGCHLD on Linux
        {"CONT", SIGCONT},
#if defined(SIGEMT)
        {"EMT", SIGEMT}, // macOS and some Linux architectures
#endif
        // Omitted because AMReX typically handles SIGFPE specially
        // {"FPE", SIGFPE},
        {"HUP", SIGHUP},
        {"ILL", SIGILL},
#if defined(SIGINFO)
        {"INFO", SIGINFO}, // macOS and some Linux architectures
#endif
        {"INT", SIGINT},
        {"IO", SIGIO},
        {"IOT", SIGABRT}, // Synonymous to SIGABRT on Linux
        // {"KILL", SIGKILL}, // Cannot be handled
        {"PIPE", SIGPIPE},
        {"POLL", SIGIO}, // Synonymous to SIGIO on Linux
        {"PROF", SIGPROF},
#if defined(SIGPWR)
        {"PWR", SIGPWR}, // Linux-only
#endif
        {"QUIT", SIGQUIT},
        {"SEGV", SIGSEGV},
#if defined(SIGSTKFLT)
        {"STKFLT", SIGSTKFLT}, // Linux-only
#endif
        // {"STOP", SIGSTOP}, // Cannot be handled
        {"SYS", SIGSYS},
        {"TERM", SIGTERM},
        {"TRAP", SIGTRAP},
        {"TSTP", SIGTSTP},
        {"TTIN", SIGTTIN},
        {"TTOU", SIGTTOU},
        {"URG", SIGURG},
        {"USR1", SIGUSR1},
        {"USR2", SIGUSR2},
        {"VTALRM", SIGVTALRM},
        {"WINCH", SIGWINCH},
        {"XCPU", SIGXCPU},
        {"XFSZ", SIGXFSZ},
    };

    for (const auto& sp : signals_to_parse) {
        const std::string name_upper = sp.abbrev;
        std::string name_lower = name_upper;
        for (char &c : name_lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        signals_parser.setConstant(name_upper, sp.value);
        signals_parser.setConstant(name_lower, sp.value);
        const auto sig_name_upper = "SIG" + name_upper;
        const auto sig_name_lower = "sig" + name_lower;
        signals_parser.setConstant(sig_name_upper, sp.value);
        signals_parser.setConstant(sig_name_lower, sp.value);
    }
#endif // #if defined(__linux__) || defined(__APPLE__)

    auto spf = signals_parser.compileHost<0>();

    const auto sig = int(spf());
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(sig < NUM_SIGNALS,
                                       "Parsed signal value is outside the supported range of [1, 31]");

    return sig;
}

void
SignalHandling::InitSignalHandling ()
{
#if defined(__linux__) || defined(__APPLE__)
    bool any_status_signal = false;
    for (int signal_number = 0; signal_number < NUM_SIGNALS; ++signal_number) {
        any_status_signal |= signal_conf_requests[SIGNAL_REQUESTS_STATUS][signal_number];
    }
    {
        const std::scoped_lock lock(m_status_mutex);
        m_status_enabled = any_status_signal;
        m_rank = amrex::ParallelDescriptor::MyProc();
#if defined(AMREX_USE_GPU)
        m_device_id = amrex::Gpu::Device::deviceId();
#endif
    }
    // The reporter thread must be ready before a signal handler can wake it up
    if (any_status_signal) {
        StartStatusReporter();
    }

    struct sigaction sa{};
    sigemptyset(&sa.sa_mask);
    for (int signal_number = 0; signal_number < NUM_SIGNALS; ++signal_number) {
        signal_received_flags[signal_number] = false;

        // actions that are broadcast from rank 0 and handled at the next timestep
        bool deferred_active = false;
        for (int request = 0; request < SIGNAL_REQUESTS_SIZE; ++request) {
            if (request == SIGNAL_REQUESTS_STATUS) { continue; }
            deferred_active |= signal_conf_requests[request][signal_number];
        }
        const bool status_active = signal_conf_requests[SIGNAL_REQUESTS_STATUS][signal_number];

        if (deferred_active) {
            // at least one signal action is configured that needs communication
            m_any_signal_action_active = true;
        }
        if (deferred_active || status_active) {
            // A handler that returns from a fault would re-execute the faulting instruction
            const bool is_fault_signal = signal_number == SIGSEGV || signal_number == SIGBUS ||
                                         signal_number == SIGILL || signal_number == SIGFPE;
            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!is_fault_signal,
                "Signal handling cannot be configured for SEGV, BUS, ILL or FPE");

            // Restart interrupted system calls (e.g., in I/O or MPI) after a status report.
            // Signals with other actions keep interrupting them, as before.
            sa.sa_flags = deferred_active ? 0 : SA_RESTART;

            // Status reports are handled by every process itself. Otherwise, other
            // processes than rank 0 ignore signals and follow the lead of rank 0.
            // (The flags that SignalSetFlag sets on these processes are never read.)
            if (amrex::ParallelDescriptor::MyProc() == 0 || status_active) {
                sa.sa_handler = &SignalHandling::SignalSetFlag;
            } else {
                sa.sa_handler = SIG_IGN;
            }
            const int result = sigaction(signal_number, &sa, nullptr);
            ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(result == 0,
                                               "Failed to install signal handler for a configured signal");
        }
    }
#endif
}

void
SignalHandling::CheckSignals ()
{
    // Is any signal handling action configured?
    // If not, we can skip all handling and the MPI communication as well.
    if (!m_any_signal_action_active) {
        return;
    }

    // We assume that signals will definitely be delivered to rank 0,
    // and may be delivered to other ranks as well. For coordination,
    // we process them according to when they're received by rank 0.
    if (amrex::ParallelDescriptor::MyProc() == 0) {
        for (int signal_number = 0; signal_number < NUM_SIGNALS; ++signal_number) {
            // Read into a local temporary to ensure the same value is
            // used throughout. Atomically exchange it with false to
            // unset the flag without risking loss of a signal - if a
            // signal arrives after this, it will be handled the next
            // time this function is called.
            const bool signal_received = signal_received_flags[signal_number].exchange(false);

            if (signal_received) {
                for (int request = 0; request < SIGNAL_REQUESTS_SIZE; ++request) {
                    // handled immediately on every process in SignalSetFlag
                    if (request == SIGNAL_REQUESTS_STATUS) { continue; }
                    signal_actions_requested[request] |=
                        signal_conf_requests[request][signal_number];
                }
            }
        }
    }

#if defined(AMREX_USE_MPI)
    // Due to a bug in Cray's MPICH 8.1.13 implementation (CUDA builds on Perlmutter@NERSC in 2022),
    // we cannot use the MPI_CXX_BOOL C++ datatype here. See WarpX PR #3029 and NERSC INC0183281
    static_assert(sizeof(bool) == 1, "We communicate bools as 1 byte-sized type in MPI");
    BL_MPI_REQUIRE(MPI_Ibcast(signal_actions_requested, SIGNAL_REQUESTS_SIZE,
                              MPI_BYTE, 0, amrex::ParallelDescriptor::Communicator(),
                              &signal_mpi_ibcast_request));
#endif
}

void
SignalHandling::WaitSignals ()
{
    // Is any signal handling action configured?
    // If not, we can skip all handling and the MPI communication as well.
    if (!m_any_signal_action_active) {
        return;
    }

#if defined(AMREX_USE_MPI)
    BL_MPI_REQUIRE(MPI_Wait(&signal_mpi_ibcast_request, MPI_STATUS_IGNORE));
#endif
}

bool
SignalHandling::TestAndResetActionRequestFlag (int action_to_test)
{
    const bool retval = signal_actions_requested[action_to_test];
    signal_actions_requested[action_to_test] = false;
    return retval;
}

void
SignalHandling::FinalizeSignalHandling ()
{
    // Waits for a status report in progress. The reporter thread and its pipe are kept,
    // and the signal handlers stay installed, but no reports are printed anymore.
    const std::scoped_lock lock(m_status_mutex);
    m_status_enabled = false;
    m_current_step.store(-1);
    m_last_completed_step.store(-1);
}

void
SignalHandling::SetCurrentStep (int step)
{
    m_current_step.store(step);
}

void
SignalHandling::SetLastCompletedStep (int step)
{
    // order matters for PrintStatusReport, which reads m_current_step first
    m_last_completed_step.store(step);
    m_current_step.store(-1);
}

void
SignalHandling::SignalSetFlag (int signal_number)
{
    signal_received_flags[signal_number] = true;

#if defined(__linux__) || defined(__APPLE__)
    if (signal_conf_requests[SIGNAL_REQUESTS_STATUS][signal_number]) {
        // Only async-signal-safe operations are allowed here,
        // so we just wake up the status reporter thread.
        const int fd = m_status_pipe_write_fd.load();
        if (fd >= 0) {
            const int saved_errno = errno;
            const char wake_up = 'S';
            // Non-blocking: if the pipe is full, a status report is already pending.
            [[maybe_unused]] const auto written = ::write(fd, &wake_up, 1);
            errno = saved_errno;
        }
    }
#endif
}

void
SignalHandling::StartStatusReporter ()
{
#if defined(__linux__) || defined(__APPLE__)
    // Already running, e.g., if a simulation is initialized again from Python
    if (m_status_pipe_write_fd.load() >= 0) { return; }

    int fds[2] = {-1, -1};
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(::pipe(fds) == 0,
                                       "Failed to create a pipe for the status signal reporter");
    for (const int fd : fds) {
        ::fcntl(fd, F_SETFD, ::fcntl(fd, F_GETFD) | FD_CLOEXEC);
    }
    // The signal handler must never block on a full pipe
    ::fcntl(fds[1], F_SETFL, ::fcntl(fds[1], F_GETFL) | O_NONBLOCK);

    // The thread is detached and lives until the process exits: a joinable std::thread
    // would call std::terminate when destroyed, e.g., if the process exits early.
    // FinalizeSignalHandling disables it instead, and a new InitSignalHandling reuses it.
    std::thread(&SignalHandling::StatusReporterLoop, fds[0]).detach();

    m_status_pipe_write_fd.store(fds[1]);
#endif
}

void
SignalHandling::StatusReporterLoop ([[maybe_unused]] int read_fd)
{
#if defined(__linux__) || defined(__APPLE__)
    while (true) {
        // Signals that arrived in quick succession are coalesced into one report
        char buffer[64];
        const auto n = ::read(read_fd, buffer, sizeof(buffer));
        if (n > 0) {
            const std::scoped_lock lock(m_status_mutex);
            if (!m_status_enabled) { continue; }
            // An exception would call std::terminate here: a status report must never
            // abort the simulation. std::fprintf does not throw.
            try {
                PrintStatusReport();
            } catch (std::exception const& e) {
                [[maybe_unused]] const int printed = std::fprintf(
                    stderr, "SIGNAL STATUS: rank=%d report failed: %s\n", m_rank, e.what());
            } catch (...) {
                [[maybe_unused]] const int printed = std::fprintf(
                    stderr, "SIGNAL STATUS: rank=%d report failed\n", m_rank);
            }
        } else if (n == 0 || errno != EINTR) {
            // write end closed or unrecoverable error
            break;
        }
    }
    ::close(read_fd);
#endif
}

void
SignalHandling::PrintStatusReport ()
{
    // The current GPU is a per-thread setting
#if defined(AMREX_USE_CUDA)
    [[maybe_unused]] const auto set_device_error = cudaSetDevice(m_device_id);
#elif defined(AMREX_USE_HIP)
    [[maybe_unused]] const auto set_device_error = hipSetDevice(m_device_id);
#endif

    std::ostringstream ss;
    ss << "SIGNAL STATUS: rank=" << m_rank;

    if (const int step = m_current_step.load(); step >= 0) {
        ss << " step=" << step;
    } else if (const int last_step = m_last_completed_step.load(); last_step >= 0) {
        ss << " last_step=" << last_step;
    } else {
        ss << " step=n/a";
    }

    ss << " section=\"" << current_profiler_section() << "\"";

    ss << " cpu_mem_rss=";
    print_gib(ss, cpu_resident_memory());

#if defined(AMREX_USE_GPU)
    const auto gpu_mem = gpu_device_memory();
    ss << " gpu_mem_used=";
    print_gib(ss, gpu_mem ? std::optional<std::size_t>(gpu_mem->first) : std::nullopt);
    ss << " gpu_mem_total=";
    print_gib(ss, gpu_mem ? std::optional<std::size_t>(gpu_mem->second) : std::nullopt);
    ss << " gpu_arena_used=";
    print_gib(ss, gpu_arena_memory());
#endif

    ss << "\n";

    // Insert the whole line at once, to avoid interleaving with output of other threads
    std::cout << ss.str() << std::flush;
}

} // namespace ablastr::utils

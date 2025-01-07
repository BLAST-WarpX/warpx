/* Copyright 2022 Philip Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SignalHandling.H"
#include "TextMsg.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_IParser.H>

#include <cctype>
#include <stdexcept>

// For sigaction() et al.
#if defined(__linux__) || defined(__APPLE__)
#   include <csignal>
#endif

namespace ablastr::utils {

bool SignalHandling::m_any_signal_action_active = false;
std::atomic<bool> SignalHandling::signal_received_flags[NUM_SIGNALS];
bool SignalHandling::signal_conf_requests[SIGNAL_REQUESTS_SIZE][NUM_SIGNALS];
bool SignalHandling::signal_actions_requested[SIGNAL_REQUESTS_SIZE];
#if defined(AMREX_USE_MPI)
MPI_Request SignalHandling::signal_mpi_ibcast_request;
#endif

int
SignalHandling::parseSignalNameToNumber (const std::string &str)
{
    amrex::IParser signals_parser(str);

#if defined(__linux__) || defined(__APPLE__)
    const struct {
        const char* abbrev;
        const int value;
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
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    for (int signal_number = 0; signal_number < NUM_SIGNALS; ++signal_number) {
        signal_received_flags[signal_number] = false;

        bool signal_active = false;
        for (const auto& request : signal_conf_requests) {
            signal_active |= request[signal_number];
        }
        if (signal_active) {
            // at least one signal action is configured
            m_any_signal_action_active = true;

            if (amrex::ParallelDescriptor::MyProc() == 0) {
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
                int signal_request = 0;
                for (const auto& request : signal_conf_requests) {
                    signal_actions_requested[signal_request++] |= request[signal_number];
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
SignalHandling::SignalSetFlag (int signal_number)
{
    signal_received_flags[signal_number] = true;
}

} // namespace ablastr::utils

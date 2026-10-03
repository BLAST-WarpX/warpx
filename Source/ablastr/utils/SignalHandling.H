/* Copyright 2022 Philip Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_SIGNAL_HANDLING_H_
#define ABLASTR_SIGNAL_HANDLING_H_

#include <AMReX_Config.H>

#if defined(AMREX_USE_MPI)
#   include <mpi.h>
#endif

#include <atomic>
#include <mutex>
#include <string>

namespace ablastr::utils {

/**
 * \brief
 * Signal handling
 *
 * Rank 0 will accept signals and asynchronously broadcast the
 * configured response; other processes will ignore them and
 * follow the lead of rank 0 to avoid potential for deadlocks or
 * timestep-skewed response.
 *
 * The exception is SIGNAL_REQUESTS_STATUS: it is handled immediately and
 * independently by every process that receives the signal, without any
 * MPI communication. The signal handler only wakes up a dedicated
 * reporter thread, which then prints a one-line status report.
 * This works even if the main thread is stuck, e.g., in a hang.
 *
 * Variables and functions are static rather than per-instance
 * because signal handlers are configured at the process level.
 */
class SignalHandling
{
public:
    //! The range of signal values to accept
    static constexpr int NUM_SIGNALS = 32;

    //! Labels for indexed positions in signal_actions_requests
    enum signal_action_requested_labels {
        //! Cleanly stop execution, as if the simulation reached its configured end
        SIGNAL_REQUESTS_BREAK = 0,
        //! Produce a checkpoint
        SIGNAL_REQUESTS_CHECKPOINT = 1,
        //! Immediately print a one-line status report on every process that receives the signal
        SIGNAL_REQUESTS_STATUS = 2,
        SIGNAL_REQUESTS_SIZE = 3 // This should always be 1 greater than the last valid value
    };

    //! Whether configuration requests the code take a particular action on a particular signal
    static bool signal_conf_requests[SIGNAL_REQUESTS_SIZE][NUM_SIGNALS];

    //! Take a string and convert it to a corresponding signal number if possible
    static int parseSignalNameToNumber (const std::string &str);

    //! Set up signal handlers based on input configuration provided in `signal_conf_requests_*`
    static void InitSignalHandling ();

    //! Stop status reports; call before AMReX is finalized
    static void FinalizeSignalHandling ();

    //! Check and clear signal flags and asynchronously broadcast them from process 0
    static void CheckSignals ();
    //! Complete the asynchronous broadcast of signal flags
    static void WaitSignals ();

    //! Check whether a given action has been requested, and reset the associated flag
    static bool TestAndResetActionRequestFlag (int action_to_test);

    /** Set the simulation step that is reported in status reports. Thread-safe.
     *
     * @param step the step that is currently being computed
     */
    static void SetCurrentStep (int step);

    /** Set the last completed simulation step, while no step is computed. Thread-safe.
     *
     * @param step the last step that was completed
     */
    static void SetLastCompletedStep (int step);

    // Don't allow clients to incorrectly try to construct and use an instance of this type
    SignalHandling () = delete;

private:
    //! Is any signal handling action configured in signal_conf_requests that is
    //! broadcast from rank 0 and handled at the next timestep?
    static bool m_any_signal_action_active;

    //! Step that is currently computed (-1: none), as set by SetCurrentStep
    static std::atomic<int> m_current_step;
    //! Last completed step (-1: none), as set by SetLastCompletedStep
    static std::atomic<int> m_last_completed_step;

    //! Write end of the pipe that wakes up the status reporter thread (-1: not started)
    static std::atomic<int> m_status_pipe_write_fd;

    //! Guards m_status_enabled, m_rank and m_device_id, and is held during a status report
    static std::mutex m_status_mutex;
    //! Whether the status reporter may access AMReX, i.e., between Init and Finalize
    static bool m_status_enabled;
    //! MPI rank of this process
    static int m_rank;
    //! GPU device of this process
    static int m_device_id;

    //! Start the detached status reporter thread, if not yet running
    static void StartStatusReporter ();

    //! Main loop of the status reporter thread, reading wake-up bytes from read_fd
    static void StatusReporterLoop (int read_fd);

    //! Assemble and print a one-line status report of this process to stdout (holds m_status_mutex)
    static void PrintStatusReport ();

    //! Whether a given signal has been received since the last check (only read on process 0)
    static std::atomic<bool> signal_received_flags[NUM_SIGNALS];

#if defined(AMREX_USE_MPI)
    //! MPI requests for the asynchronous broadcasts of the signal-requested actions
    static MPI_Request signal_mpi_ibcast_request;
#endif

    /** Signal handler: sets flags and wakes up the status reporter thread
     *
     * Installed on process 0 for all configured signals, and on all processes for status
     * signals; other processes ignore the remaining configured signals.
     * Must only use async-signal-safe operations.
     */
    static void SignalSetFlag (int signal_number);

    //! Boolean flags transmitted between CheckSignals() and
    //! HandleSignals() to indicate actions requested by signals
    static bool signal_actions_requested[SIGNAL_REQUESTS_SIZE];
};

} // namespace ablastr::utils

#endif // ABLASTR_SIGNAL_HANDLING_H_

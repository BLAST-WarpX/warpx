/* Copyright 2022 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_WARN_MANAGER_H_
#define ABLASTR_WARN_MANAGER_H_

#include "ablastr/utils/msg_logger/MsgLogger_fwd.H"

#include <AMReX_BaseFwd.H>

#include <memory>
#include <optional>
#include <string>

namespace ablastr::warn_manager
{

    /** WarnPriority is recorded together with warning messages. It influences
    * the display order and the appearance of a warning message.
    * This enum class mirrors ablastr::utils::msg_logger::Priority.
    */
    enum class WarnPriority
    {
        /** Low priority warning:
        * essentially an informative message
        */
        low,
        /** Medium priority warning:
         * a bug or a performance issue may affect the simulation
         */
        medium,
        /** High priority warning:
         * a very serious bug or performance issue
         * almost certainly affects the simulation
         */
        high
    };

    /**
    * The singleton class WarnManager manages warning messages,
    * providing methods to record warnings, and print warning
    * lists.
    */
    class WarnManager
    {
    public:

        /**
        * A singleton class should not be cloneable.
        */
        WarnManager( const WarnManager& ) = delete;

        /**
        * A singleton class should not be cloneable.
        */
        WarnManager( WarnManager&& ) = delete;

        /**
        * A singleton class should not be assignable.
        */
        WarnManager& operator=( const WarnManager & ) = delete;

        /**
        * A singleton class should not be assignable.
        */
        WarnManager& operator=( const WarnManager&& ) = delete;

        /**
        * Default destructor
        */
        ~WarnManager() = default;

        static WarnManager& GetInstance();

        /**
        * \brief This function records a warning message (recording a warning message is thread-safe)
        *
        * @param[in] topic a string to identify the topic of the warning (e.g., "parallelization", "pbc", "particles"...)
        * @param[in] text the text of the warning message
        * @param[in] priority priority of the warning message ("medium" by default)
         */
        void RecordWarning(
            const std::string& topic,
            const std::string& text,
            const WarnPriority& priority = WarnPriority::medium);

        /**
        * \brief This function prints all the warning messages collected on the present MPI rank
        * (i.e., this is not a collective call). This function is mainly intended for debug purposes.
        *
        * @param[in] when a string to mark when the warnings are printed out (it appears in the warning list)
        * @return a string containing the "local" warning list
        */
        [[nodiscard]] std::string PrintLocalWarnings(
            const std::string& when) const;

        /**
        * \brief This function prints all the warning messages collected by all the MPI ranks
        * (i.e., this is a collective call). Only the I/O rank prints the message.
        *
        * @param[in] when a string to mark when the warnings are printed out (it appears in the warning list)
        * @return a string containing the "global" warning list
        */
        [[nodiscard]] std::string PrintGlobalWarnings(
            const std::string& when) const;

        /**
        * \brief Setter for the m_always_warn_immediately
        *
        * @param[in] always_warn_immediately new value of the m_always_warn_immediately flag
        */
        void SetAlwaysWarnImmediately(bool always_warn_immediately);

        /**
        * \brief Getter for the m_always_warn_immediately
        *
        * @return the m_always_warn_immediately flag
        */
        [[nodiscard]] bool GetAlwaysWarnImmediatelyFlag() const;

        /**
        * \brief Setter for the m_abort_on_warning_threshold flag
        * (pass std::nullopt in order to never abort)
        *
        * @param[in] abort_threshold new value of the m_abort_on_warning_threshold flag
        */
        void SetAbortThreshold(std::optional<WarnPriority> abort_threshold);

        /**
        * \brief Getter for the m_abort_on_warning_threshold flag
        *
        * @return the m_abort_on_warning_threshold flag
        */
        [[nodiscard]] std::optional<WarnPriority> GetAbortThreshold() const;

        /**
        * \brief This function reads warning messages from the inputfile. It is intended for
        * debug&testing purposes
        *
        * @param[in, out] params the inputfile parser
        */
        void debug_read_warnings_from_input(const amrex::ParmParse& params);

        static const int warn_line_size = 80 /*! Maximum line length to be used in formatting warning list*/;
        static const int warn_tab_size = 5 /*! Tabulation size to be used in formatting warning list*/;

    private:

        /**
        * The constructor.
        */
        WarnManager();

        /**
        * \brief This function generates a string for a single entry of the warning list
        * for a MessageWithCounter struct (i.e., a warning message paired with a counter storing
        * how many times the warning has been raised)
        *
        * @param[in] msg_with_counter a MessageWithCounter
        * @return a string containing the warning message
        */
        [[nodiscard]] std::string PrintWarnMsg(
            const ablastr::utils::msg_logger::MsgWithCounter& msg_with_counter) const;

        /**
        * \brief This function generates a string for a single entry of the warning list
        * for a MsgWithCounterAndRanks struct (i.e., a warning message paired with a counter storing
        * how many times the warning has been raised and info on which ranks have raised the warning)
        *
        * @param[in] msg_with_counter_and_ranks a MsgWithCounterAndRanks
        * @return a string containing the warning message
        */
        [[nodiscard]] std::string PrintWarnMsg(
            const ablastr::utils::msg_logger::MsgWithCounterAndRanks& msg_with_counter_and_ranks) const;

        /**
        * \brief This function generates the header of the warning messages list
        *
        * @param[in] when a string to mark when the warnings are printed out (it appears in the warning list)
        * @param[in] line_size maximum line length to be used in formatting warning list
        * @param[in] is_global flag: true if the header is for a global warning list, false otherwise
        * @return a string containing the header of the warning list
        */
        static std::string GetHeader(
            const std::string& when,
            int line_size,
            bool is_global);

        /**
        * \brief This function formats each line of a warning message text
        *
        * @param[in] msg the warning message text
        * @param[in] line_size maximum line length to be used in formatting warning list
        * @param[in] tab_size tabulation size to be used in formatting warning list
        * @return a string containing the formatted warning message text
        */
        static std::string MsgFormatter(
            const std::string& msg,
            int line_size,
            int tab_size);

        int m_rank = 0 /*! MPI rank (appears in the warning list)*/;
        std::unique_ptr<ablastr::utils::msg_logger::Logger> m_p_logger /*! The Logger stores all the warning messages*/;
        bool m_always_warn_immediately = false /*! Flag to control if the warning logger has to emit a warning message as soon as a warning is recorded*/;
        std::optional<WarnPriority> m_abort_on_warning_threshold = std::nullopt /*! Threshold to abort immediately on a warning message*/;
    };

    /**
    * \brief Helper function to abbreviate the call to get a WarnManager instance
    *
    * @return the instance of the WarnManager class
    */
    WarnManager& GetWMInstance();

    /**
    * \brief Helper function to abbreviate the call to WarnManager::GetInstance().RecordWarning
    * (recording a warning message is thread-safe)
    *
    * @param[in] topic a string to identify the topic of the warning (e.g., "parallelization", "pbc", "particles"...)
    * @param[in] text the text of the warning message
    * @param[in] priority priority of the warning message ("medium" by default)
    */
    void WMRecordWarning(
        const std::string& topic,
        const std::string& text,
        const WarnPriority& priority = WarnPriority::medium);
}

#endif //ABLASTR_WARN_MANAGER_H_

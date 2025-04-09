/* Copyright 2022 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarnManager.H"

#include "ablastr/utils/msg_logger/MsgLogger.H"
#include "ablastr/utils/text/StringUtils.H"
#include "ablastr/utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Config.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <sstream>
#include <vector>

namespace abl_msg_logger = ablastr::utils::msg_logger;
using namespace ablastr::warn_manager;

namespace
{
    WarnPriority MapPriorityToWarnPriority (
        const abl_msg_logger::Priority& priority)
    {
        using namespace abl_msg_logger;
        if (priority == Priority::low) {
            return WarnPriority::low;
        } else if (priority == Priority::medium) {
            return WarnPriority::medium;
        } else if (priority == Priority::high) {
            return WarnPriority::high;
        } else {
            ABLASTR_ABORT_WITH_MESSAGE(
                "Parsing Priority to WarnPriority has failed");
        }

        return WarnPriority::high;
    }
}

WarnManager& WarnManager::GetInstance() {
    static auto warn_manager = WarnManager{};
    return warn_manager;
}

WarnManager::WarnManager():
    m_rank{amrex::ParallelDescriptor::MyProc()},
    m_p_logger{std::make_unique<abl_msg_logger::Logger>()}
{}

void WarnManager::RecordWarning(
            const std::string& topic,
            const std::string& text,
            const WarnPriority& priority)
{
    auto msg_priority = abl_msg_logger::Priority::high;
    if(priority == WarnPriority::low) {
        msg_priority = abl_msg_logger::Priority::low;
    } else if(priority == WarnPriority::medium) {
        msg_priority = abl_msg_logger::Priority::medium;
    }

    if(m_always_warn_immediately){

        amrex::Warning(
            ablastr::utils::TextMsg::Warn(
                "["
                + std::string(abl_msg_logger::PriorityToString(msg_priority))
                + "]["
                + topic
                + "] "
                + text));
    }

#ifdef AMREX_USE_OMP
    #pragma omp critical
#endif
    {
        m_p_logger->record_msg(abl_msg_logger::Msg{topic, text, msg_priority});
    }

    if(m_abort_on_warning_threshold){

        auto abort_priority = abl_msg_logger::Priority::high;
        if(m_abort_on_warning_threshold == WarnPriority::low) {
            abort_priority = abl_msg_logger::Priority::low;
        } else if(m_abort_on_warning_threshold == WarnPriority::medium) {
            abort_priority = abl_msg_logger::Priority::medium;
        }

        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
            msg_priority < abort_priority,
            "A warning with priority '"
            + abl_msg_logger::PriorityToString(msg_priority)
            + "' has been raised."
        );
    }
}

std::string WarnManager::PrintLocalWarnings(const std::string& when) const
{
    auto all_warnings = m_p_logger->get_msgs_with_counter();
    std::sort(all_warnings.begin(), all_warnings.end(),
        [](const auto& a, const auto& b){return a.msg < b.msg;});

    std::stringstream ss;

    ss << "\n" << WarnManager::GetHeader(when, warn_line_size, false);

    if(all_warnings.empty()){
        ss << "* No recorded warnings.\n";
    }
    else{
        for(const auto& warn_msg : all_warnings){
            ss << PrintWarnMsg(warn_msg);
            ss << "*\n";
        }
    }

    ss << std::string(warn_line_size, '*') << "\n\n" ;

    return ss.str();
}

std::string WarnManager::PrintGlobalWarnings(const std::string& when) const
{
    auto all_warnings =
        m_p_logger->collective_gather_msgs_with_counter_and_ranks();

    if(m_rank != amrex::ParallelDescriptor::IOProcessorNumber()) {
        return "[see I/O rank message]";
    }

    std::sort(all_warnings.begin(), all_warnings.end(),
        [](const auto& a, const auto& b){
            return a.msg_with_counter.msg < b.msg_with_counter.msg;});

    std::stringstream ss;

    ss << "\n" << WarnManager::GetHeader(when, warn_line_size, true);

    if(all_warnings.empty()){
        ss << "* No recorded warnings.\n";
    }
    else{
        for(const auto& warn_msg : all_warnings){
            ss << PrintWarnMsg(warn_msg);
            ss << "*\n";
        }
    }

    ss << std::string(warn_line_size, '*') << "\n\n" ;

    return ss.str();
}

void WarnManager::SetAlwaysWarnImmediately(bool always_warn_immediately)
{
    m_always_warn_immediately = always_warn_immediately;
}

bool WarnManager::GetAlwaysWarnImmediatelyFlag() const
{
    return m_always_warn_immediately;
}

void WarnManager::SetAbortThreshold(std::optional<WarnPriority> abort_threshold)
{
    m_abort_on_warning_threshold = abort_threshold;
}

std::optional<WarnPriority> WarnManager::GetAbortThreshold() const
{
    return m_abort_on_warning_threshold;
}

void WarnManager::debug_read_warnings_from_input(const amrex::ParmParse& params)
{
    std::vector<std::string> warnings;
    params.queryarr("test_warnings", warnings);

    for (const auto& warn : warnings){
        const amrex::ParmParse pp_warn(warn);

        std::string topic;
        pp_warn.query("topic", topic);

        std::string msg;
        pp_warn.query("msg", msg);

        std::string spriority;
        pp_warn.query("priority", spriority);
        const auto wpriority = MapPriorityToWarnPriority(
            abl_msg_logger::StringToPriority(spriority));

        int all_involved = 0;
        pp_warn.query("all_involved", all_involved);
        if(all_involved != 0){
            this->RecordWarning(topic, msg, wpriority);
        }
        else{
            std::vector<int> who_involved;
            pp_warn.queryarr("who_involved", who_involved);
            if(std::find (who_involved.begin(), who_involved.end(), m_rank)
                != who_involved.end()){
                this->RecordWarning(topic, msg, wpriority);
            }
        }
    }

}

std::string WarnManager::PrintWarnMsg(
    const abl_msg_logger::MsgWithCounter& msg_with_counter) const
{
    std::stringstream ss;
    ss << "* --> ";
    if (msg_with_counter.msg.priority == abl_msg_logger::Priority::high) {
        ss << "[!!!]";
    } else if (msg_with_counter.msg.priority == abl_msg_logger::Priority::medium) {
        ss << "[!! ]";
    } else if (msg_with_counter.msg.priority == abl_msg_logger::Priority::low) {
        ss << "[!  ]";
    } else {
        ss << "[???]";
    }

    ss << " [" + msg_with_counter.msg.topic << "] ";

    if(msg_with_counter.counter == 2) {
        ss << "[raised twice]\n";
    } else if(msg_with_counter.counter == 1) {
        ss << "[raised once]\n";
    } else {
        ss << "[raised " << msg_with_counter.counter << " times]\n";
    }

    ss << MsgFormatter(msg_with_counter.msg.text, warn_line_size, warn_tab_size);

    return ss.str();
}

std::string WarnManager::PrintWarnMsg(
    const abl_msg_logger::MsgWithCounterAndRanks& msg_with_counter_and_ranks) const
{
    std::stringstream ss;
    ss << this->PrintWarnMsg(msg_with_counter_and_ranks.msg_with_counter);

    std::string raised_by = "@ Raised by: ";
    if (!msg_with_counter_and_ranks.all_ranks){
        for (const auto rr : msg_with_counter_and_ranks.ranks) {
            raised_by += " " + std::to_string(rr);
        }
    }
    else{
        raised_by += "ALL\n";
    }
    ss << WarnManager::MsgFormatter(raised_by, warn_line_size, warn_tab_size);

    return ss.str();
}

std::string WarnManager::GetHeader(
    const std::string& when,
    const int line_size,
    const bool is_global)
{
    const std::string warn_header{"**** WARNINGS "};

    std::stringstream ss;

    ss << warn_header <<
        std::string(line_size - static_cast<int>(warn_header.length()), '*') << "\n" ;

    if(is_global){
        ss << "* GLOBAL warning list  after " << " [ " <<  when << " ]\n*\n";
    }
    else{
        auto const mpi_rank = amrex::ParallelDescriptor::MyProc();
        ss << "* LOCAL" << " ( rank # " << mpi_rank << " ) "
            << " warning list  after " <<  when << "\n*\n";
    }

    return ss.str();
}

std::string
WarnManager::MsgFormatter(
        const std::string& msg,
        const int line_size,
        const int tab_size)
{
    const auto prefix = "*" + std::string(tab_size, ' ');
    const auto prefix_length = static_cast<int>(prefix.length());

    const auto wrapped_text = ablastr::utils::text::automatic_text_wrap(
        msg, line_size-prefix_length);

    std::stringstream ss_out;
    for (const auto& line : wrapped_text) {
        ss_out << prefix << line << "\n";
    }

    return ss_out.str();
}

WarnManager& ablastr::warn_manager::GetWMInstance()
{
    return WarnManager::GetInstance();
}

void ablastr::warn_manager::WMRecordWarning(
    const std::string& topic,
    const std::string& text,
    const WarnPriority& priority)
{
    WarnManager::GetInstance().RecordWarning(
        topic, text, priority);
}

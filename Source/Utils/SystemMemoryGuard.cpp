/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SystemMemoryGuard.H"

#include "Utils/TextMsg.H"

#include <ablastr/utils/SystemMemory.H>

#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace
{
    warpx::utils::system_memory_guard::Config& guard_config ()
    {
        static auto config = warpx::utils::system_memory_guard::Config{};
        return config;
    }

    std::string format_memory_guard_message (
        warpx::utils::system_memory_guard::Config const& config,
        ablastr::utils::SystemMemoryInfo const& info,
        int step,
        bool local_over_threshold,
        double max_observed_fraction)
    {
        auto os = std::ostringstream{};
        os << std::setprecision(6);

        if (!info.supported) {
            os << "warpx.max_system_memory_fraction is enabled, but system "
               << "memory query is unsupported on this platform at step "
               << step << ". ";
        } else {
            auto const local_fraction =
                ablastr::utils::UsedSystemMemoryFraction(info);
            os << "System memory guard triggered by "
                << "warpx.max_system_memory_fraction at step " << step
                << " on one or more MPI ranks/nodes. This rank "
                << (local_over_threshold ? "exceeded" : "did not exceed")
                << " the threshold. Local rank memory: "
                << "used=" << info.used_bytes << " bytes, "
                << "total=" << info.total_bytes << " bytes, "
                << "available=" << info.available_bytes << " bytes, "
                << "fraction=" << local_fraction
                << ", max_fraction_across_ranks=" << max_observed_fraction
                << ", threshold=" << config.max_fraction << ". ";
        }

        os << "lower problem size or raise warpx.max_system_memory_fraction";
        return os.str();
    }
} // namespace

namespace warpx::utils::system_memory_guard
{

Config ReadConfig (amrex::ParmParse const& pp_warpx)
{
    auto config = Config{};
    pp_warpx.query("max_system_memory_fraction", config.max_fraction);
    pp_warpx.query("system_memory_check_interval", config.check_interval);
    return config;
}

std::string ValidateConfig (
    Config const& config,
    bool memory_query_supported)
{
    if (config.check_interval < 1) {
        return "warpx.system_memory_check_interval must be >= 1";
    }

    if (!std::isfinite(config.max_fraction)) {
        return "warpx.max_system_memory_fraction must be finite and either "
               "0.0 to disable or in the valid range (0, 1)";
    }

    if (config.max_fraction == 0.0) {
        return {};
    }

    // The guard trips when the used fraction strictly exceeds max_fraction, so a
    // value of 1.0 could never trip (used memory cannot exceed total). Require a
    // fraction strictly below 1.0 so the guard can fire before exhaustion.
    if (config.max_fraction < 0.0 || config.max_fraction >= 1.0) {
        return "warpx.max_system_memory_fraction must be 0.0 to disable or "
               "in the valid range (0, 1)";
    }

    if (!memory_query_supported) {
        return "warpx.max_system_memory_fraction is enabled, but system memory "
               "query is unsupported on this platform";
    }

    return {};
}

bool ShouldCheck (
    Config const& config,
    int step,
    int step_begin) noexcept
{
    if (!config.enabled()) {
        return false;
    }

    if (step <= step_begin) {
        return true;
    }

    return ((step - step_begin) % config.check_interval) == 0;
}

void ReadParameters ()
{
    auto const pp_warpx = amrex::ParmParse{"warpx"};
    auto config = ReadConfig(pp_warpx);

    auto const memory_info = config.enabled()
        ? ablastr::utils::QuerySystemMemory()
        : ablastr::utils::SystemMemoryInfo{};

    auto const error = ValidateConfig(
        config,
        !config.enabled() || memory_info.supported);
    if (!error.empty()) {
        WARPX_ABORT_WITH_MESSAGE(error);
    }

    guard_config() = config;
}

void Check (int step)
{
    auto const config = guard_config();
    if (!config.enabled()) {
        return;
    }

    auto const memory_info = ablastr::utils::QuerySystemMemory();
    auto max_observed_fraction =
        ablastr::utils::UsedSystemMemoryFraction(memory_info);
    amrex::ParallelDescriptor::ReduceRealMax(max_observed_fraction);

    auto const local_over_threshold =
        ablastr::utils::ExceedsSystemMemoryFraction(
            memory_info, config.max_fraction);
    auto const local_guard_failure =
        !memory_info.supported || local_over_threshold;

    auto any_rank_guard_failure = local_guard_failure;
    amrex::ParallelDescriptor::ReduceBoolOr(any_rank_guard_failure);

    if (any_rank_guard_failure) {
        WARPX_ABORT_WITH_MESSAGE(
            format_memory_guard_message(
                config,
                memory_info,
                step,
                local_over_threshold,
                max_observed_fraction));
    }
}

void CheckIfDue (int step, int step_begin)
{
    if (ShouldCheck(guard_config(), step, step_begin)) {
        Check(step);
    }
}

} // namespace warpx::utils::system_memory_guard

/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SYSTEM_MEMORY_GUARD_H_
#define WARPX_SYSTEM_MEMORY_GUARD_H_

#include <AMReX_ParmParse.H>

#include <string>

namespace warpx::utils::system_memory_guard
{

struct Config
{
    double max_fraction = 0.0;
    int check_interval = 1;

    [[nodiscard]] bool enabled () const noexcept { return max_fraction > 0.0; }
};

Config ReadConfig (amrex::ParmParse const& pp_warpx);

std::string ValidateConfig (
    Config const& config,
    bool memory_query_supported);

bool ShouldCheck (
    Config const& config,
    int step,
    int step_begin) noexcept;

void ReadParameters ();

void Check (int step);

void CheckIfDue (int step, int step_begin);

} // namespace warpx::utils::system_memory_guard

#endif // WARPX_SYSTEM_MEMORY_GUARD_H_

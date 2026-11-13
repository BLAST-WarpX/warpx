/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_UTILS_SYSTEM_MEMORY_H_
#define ABLASTR_UTILS_SYSTEM_MEMORY_H_

#include <cstdint>

namespace ablastr::utils
{

struct SystemMemoryInfo
{
    std::uint64_t total_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::uint64_t used_bytes = 0;
    bool supported = false;
};

SystemMemoryInfo MakeSystemMemoryInfo (
    std::uint64_t total_bytes,
    std::uint64_t available_bytes,
    bool supported) noexcept;

SystemMemoryInfo QuerySystemMemory () noexcept;

double UsedSystemMemoryFraction (SystemMemoryInfo const& info) noexcept;

bool ExceedsSystemMemoryFraction (
    SystemMemoryInfo const& info,
    double max_fraction) noexcept;

} // namespace ablastr::utils

#endif // ABLASTR_UTILS_SYSTEM_MEMORY_H_

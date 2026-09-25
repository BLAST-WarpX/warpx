/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SystemMemory.H"

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>

namespace
{
    std::uint64_t saturating_add (std::uint64_t lhs, std::uint64_t rhs) noexcept
    {
        constexpr auto max_value = std::numeric_limits<std::uint64_t>::max();
        if (max_value - lhs < rhs) {
            return max_value;
        }
        return lhs + rhs;
    }

    std::uint64_t saturating_multiply (std::uint64_t lhs, std::uint64_t rhs) noexcept
    {
        constexpr auto max_value = std::numeric_limits<std::uint64_t>::max();
        if (lhs != 0 && max_value / lhs < rhs) {
            return max_value;
        }
        return lhs * rhs;
    }

#if defined(__linux__)
    std::unordered_map<std::string, std::uint64_t> read_meminfo_kb ()
    {
        auto values = std::unordered_map<std::string, std::uint64_t>{};

        auto file = std::ifstream{"/proc/meminfo"};
        auto key = std::string{};
        auto unit = std::string{};
        auto value = std::uint64_t{};

        while (file >> key >> value >> unit) {
            if (!key.empty() && key.back() == ':') {
                key.pop_back();
            }
            values[key] = value;
        }

        return values;
    }
#endif
} // namespace

namespace ablastr::utils
{

SystemMemoryInfo MakeSystemMemoryInfo (
    std::uint64_t total_bytes,
    std::uint64_t available_bytes,
    bool supported) noexcept
{
    if (!supported || total_bytes == 0) {
        return {};
    }

    available_bytes = std::min(available_bytes, total_bytes);

    return SystemMemoryInfo{
        total_bytes,
        available_bytes,
        total_bytes - available_bytes,
        true};
}

SystemMemoryInfo QuerySystemMemory () noexcept
{
#if defined(__linux__)
    auto const meminfo = read_meminfo_kb();
    auto const total_it = meminfo.find("MemTotal");
    if (total_it != meminfo.end()) {
        auto const available_it = meminfo.find("MemAvailable");
        std::uint64_t available_kb = 0;

        if (available_it != meminfo.end()) {
            available_kb = available_it->second;
        } else {
            for (auto const* key : {"MemFree", "Buffers", "Cached", "SReclaimable"}) {
                auto const it = meminfo.find(key);
                if (it != meminfo.end()) {
                    available_kb = saturating_add(available_kb, it->second);
                }
            }
            auto const shmem_it = meminfo.find("Shmem");
            if (shmem_it != meminfo.end()) {
                // Subtract shared memory, saturating at zero: when Shmem is at
                // least the accumulated total, treat available as zero rather
                // than leaving it overstated.
                available_kb = (available_kb > shmem_it->second)
                    ? available_kb - shmem_it->second
                    : std::uint64_t{0};
            }
        }

        return MakeSystemMemoryInfo(
            saturating_multiply(total_it->second, 1024),
            saturating_multiply(available_kb, 1024),
            true);
    }

    struct sysinfo info{};
    if (sysinfo(&info) == 0) {
        auto const mem_unit = static_cast<std::uint64_t>(info.mem_unit);
        auto const total_bytes = saturating_multiply(
            static_cast<std::uint64_t>(info.totalram), mem_unit);
        auto const available_pages = saturating_add(
            static_cast<std::uint64_t>(info.freeram),
            static_cast<std::uint64_t>(info.bufferram));
        return MakeSystemMemoryInfo(
            total_bytes,
            saturating_multiply(available_pages, mem_unit),
            true);
    }
#elif defined(__APPLE__) && defined(__MACH__)
    std::uint64_t total_bytes = 0;
    auto total_size = sizeof(total_bytes);
    if (sysctlbyname("hw.memsize", &total_bytes, &total_size, nullptr, 0) != 0) {
        return {};
    }

    auto page_size = vm_size_t{};
    if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS) {
        return {};
    }

    auto vm_stats = vm_statistics64_data_t{};
    auto count = static_cast<mach_msg_type_number_t>(HOST_VM_INFO64_COUNT);
    if (host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&vm_stats),
            &count) != KERN_SUCCESS)
    {
        return {};
    }

    // On macOS, "available" is defined here as pages the kernel can reuse
    // without swapping: free, inactive, and speculative pages.
    auto const available_pages = saturating_add(
        saturating_add(
            static_cast<std::uint64_t>(vm_stats.free_count),
            static_cast<std::uint64_t>(vm_stats.inactive_count)),
        static_cast<std::uint64_t>(vm_stats.speculative_count));

    return MakeSystemMemoryInfo(
        total_bytes,
        saturating_multiply(available_pages, static_cast<std::uint64_t>(page_size)),
        true);
#endif

    return {};
}

double UsedSystemMemoryFraction (SystemMemoryInfo const& info) noexcept
{
    if (!info.supported || info.total_bytes == 0) {
        return 0.0;
    }

    return static_cast<double>(info.used_bytes) / static_cast<double>(info.total_bytes);
}

bool ExceedsSystemMemoryFraction (
    SystemMemoryInfo const& info,
    double max_fraction) noexcept
{
    return info.supported &&
        info.total_bytes != 0 &&
        UsedSystemMemoryFraction(info) > max_fraction;
}

} // namespace ablastr::utils

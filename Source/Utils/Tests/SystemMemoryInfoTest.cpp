/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include <ablastr/utils/SystemMemory.H>

#include <AMReX_BLassert.H>

int main ()
{
    using ablastr::utils::ExceedsSystemMemoryFraction;
    using ablastr::utils::MakeSystemMemoryInfo;
    using ablastr::utils::QuerySystemMemory;
    using ablastr::utils::UsedSystemMemoryFraction;

    auto const mocked = MakeSystemMemoryInfo(100, 25, true);
    AMREX_ALWAYS_ASSERT(mocked.supported);
    AMREX_ALWAYS_ASSERT(mocked.total_bytes == 100);
    AMREX_ALWAYS_ASSERT(mocked.available_bytes == 25);
    AMREX_ALWAYS_ASSERT(mocked.used_bytes == 75);
    AMREX_ALWAYS_ASSERT(UsedSystemMemoryFraction(mocked) == 0.75);
    AMREX_ALWAYS_ASSERT(ExceedsSystemMemoryFraction(mocked, 0.70));
    AMREX_ALWAYS_ASSERT(!ExceedsSystemMemoryFraction(mocked, 0.75));

    auto const capped = MakeSystemMemoryInfo(100, 125, true);
    AMREX_ALWAYS_ASSERT(capped.available_bytes == 100);
    AMREX_ALWAYS_ASSERT(capped.used_bytes == 0);

    auto const unsupported = MakeSystemMemoryInfo(100, 25, false);
    AMREX_ALWAYS_ASSERT(!unsupported.supported);
    AMREX_ALWAYS_ASSERT(unsupported.total_bytes == 0);
    AMREX_ALWAYS_ASSERT(UsedSystemMemoryFraction(unsupported) == 0.0);
    AMREX_ALWAYS_ASSERT(!ExceedsSystemMemoryFraction(unsupported, 0.70));

    auto const live = QuerySystemMemory();
    if (live.supported) {
        AMREX_ALWAYS_ASSERT(live.total_bytes > 0);
        AMREX_ALWAYS_ASSERT(live.available_bytes <= live.total_bytes);
        AMREX_ALWAYS_ASSERT(live.used_bytes <= live.total_bytes);
    }

    return 0;
}

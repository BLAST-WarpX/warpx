/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/SystemMemoryGuard.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <AMReX_BLassert.H>
#include <limits>
#include <string>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        auto pp_warpx = amrex::ParmParse{"warpx"};
        pp_warpx.remove("max_system_memory_fraction");
        pp_warpx.remove("system_memory_check_interval");

        auto config = warpx::utils::system_memory_guard::ReadConfig(pp_warpx);
        AMREX_ALWAYS_ASSERT(!config.enabled());
        AMREX_ALWAYS_ASSERT(config.max_fraction == 0.0);
        AMREX_ALWAYS_ASSERT(config.check_interval == 1);
        AMREX_ALWAYS_ASSERT(warpx::utils::system_memory_guard::ValidateConfig(config, false).empty());
        AMREX_ALWAYS_ASSERT(!warpx::utils::system_memory_guard::ShouldCheck(config, 0, 0));

        pp_warpx.add("max_system_memory_fraction", 0.5);
        pp_warpx.add("system_memory_check_interval", 3);

        config = warpx::utils::system_memory_guard::ReadConfig(pp_warpx);
        AMREX_ALWAYS_ASSERT(config.enabled());
        AMREX_ALWAYS_ASSERT(config.max_fraction == 0.5);
        AMREX_ALWAYS_ASSERT(config.check_interval == 3);
        AMREX_ALWAYS_ASSERT(warpx::utils::system_memory_guard::ValidateConfig(config, true).empty());
        AMREX_ALWAYS_ASSERT(warpx::utils::system_memory_guard::ShouldCheck(config, 10, 10));
        AMREX_ALWAYS_ASSERT(!warpx::utils::system_memory_guard::ShouldCheck(config, 11, 10));
        AMREX_ALWAYS_ASSERT(warpx::utils::system_memory_guard::ShouldCheck(config, 13, 10));

        config.max_fraction = -0.1;
        auto error = warpx::utils::system_memory_guard::ValidateConfig(config, true);
        AMREX_ALWAYS_ASSERT(error.find("warpx.max_system_memory_fraction") != std::string::npos);
        AMREX_ALWAYS_ASSERT(error.find("(0, 1)") != std::string::npos);

        config.max_fraction = 1.01;
        error = warpx::utils::system_memory_guard::ValidateConfig(config, true);
        AMREX_ALWAYS_ASSERT(error.find("warpx.max_system_memory_fraction") != std::string::npos);
        AMREX_ALWAYS_ASSERT(error.find("(0, 1)") != std::string::npos);

        // A fraction of exactly 1.0 can never trip (used cannot exceed total),
        // so it must be rejected rather than silently accepted as a dead config.
        config.max_fraction = 1.0;
        error = warpx::utils::system_memory_guard::ValidateConfig(config, true);
        AMREX_ALWAYS_ASSERT(error.find("warpx.max_system_memory_fraction") != std::string::npos);
        AMREX_ALWAYS_ASSERT(error.find("(0, 1)") != std::string::npos);

        config.max_fraction = std::numeric_limits<double>::quiet_NaN();
        error = warpx::utils::system_memory_guard::ValidateConfig(config, true);
        AMREX_ALWAYS_ASSERT(error.find("warpx.max_system_memory_fraction") != std::string::npos);
        AMREX_ALWAYS_ASSERT(error.find("finite") != std::string::npos);

        config.max_fraction = 0.5;
        config.check_interval = 0;
        error = warpx::utils::system_memory_guard::ValidateConfig(config, true);
        AMREX_ALWAYS_ASSERT(error.find("warpx.system_memory_check_interval") != std::string::npos);

        config.check_interval = 1;
        error = warpx::utils::system_memory_guard::ValidateConfig(config, false);
        AMREX_ALWAYS_ASSERT(error.find("unsupported") != std::string::npos);

        pp_warpx.remove("max_system_memory_fraction");
        pp_warpx.remove("system_memory_check_interval");
    }

    amrex::Finalize();
    return 0;
}

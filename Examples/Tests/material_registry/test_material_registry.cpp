/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/MaterialRegistry.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using warpx::materials::MaterialRegistry;
using namespace amrex::literals;

namespace
{
    using Selector = MaterialRegistry::ResolvedCellSelector;
    using DensityArray = amrex::GpuArray<
        amrex::Real, MaterialRegistry::max_materials>;

    constexpr int invalid =
        static_cast<int>(MaterialRegistry::ResolvedCell::Invalid);
    constexpr int mixed =
        static_cast<int>(MaterialRegistry::ResolvedCell::Mixed);
    constexpr int vacuum =
        static_cast<int>(MaterialRegistry::ResolvedCell::Vacuum);

    static_assert(MaterialRegistry::max_materials == 8);
    static_assert(invalid == -3);
    static_assert(mixed == -2);
    static_assert(vacuum == -1);
    static_assert(std::is_trivially_copyable_v<Selector>);

    void test_selector_contract ()
    {
        DensityArray density{};

        // The vacuum threshold is inclusive and never queries a material.
        Selector selector{2, 0.5_rt, 0.0_rt, 0.0_rt};
        density = {0.5_rt, 0.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == vacuum);
        density = {0.5001_rt, 0.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == 0);

        // Equality with either tolerance is resolved; a value just above it
        // is a mixed cell.  One eighth is exact in binary, so the relative
        // equality exercises the contract rather than decimal roundoff.
        selector = {2, 0.0_rt, 0.0_rt, 0.125_rt};
        density = {7.0_rt, 1.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == 0);
        density = {7.0_rt, 1.0001_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == mixed);

        selector = {2, 0.0_rt, 0.25_rt, 0.0_rt};
        density = {1.0_rt, 0.25_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == 0);
        density = {1.0_rt, 0.2501_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == mixed);

        // Contaminants are summed. Each secondary component is below the
        // relative limit here, but their aggregate is not.
        selector = {3, 0.0_rt, 0.0_rt, 0.1_rt};
        density = {8.1_rt, 0.5_rt, 0.5_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == mixed);

        // A relative tolerance below one half cannot resolve a tied maximum.
        selector = {2, 0.0_rt, 0.0_rt, 0.499_rt};
        density = {1.0_rt, 1.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == mixed);

        // Absolute tolerance must not override the unique-dominance
        // requirement. Without explicit tie detection, this would resolve
        // to the first material because its full contaminant is tolerated.
        selector = {2, 0.0_rt, 1.0_rt, 0.0_rt};
        density = {1.0_rt, 1.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == mixed);

        selector = {2, 0.0_rt, 0.0_rt, 1.0e-12_rt};
        density = {-1.0_rt, 2.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);
        density = {
            std::numeric_limits<amrex::Real>::infinity(), 2.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);
        density = {
            std::numeric_limits<amrex::Real>::quiet_NaN(), 2.0_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);
        density = {
            std::numeric_limits<amrex::Real>::max(),
            std::numeric_limits<amrex::Real>::max()};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);

        selector.num_materials = MaterialRegistry::max_materials + 1;
        density = {};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);
        selector = {2, 0.0_rt, 0.0_rt, 0.5_rt};
        AMREX_ALWAYS_ASSERT(selector(density) == invalid);
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        test_selector_contract();

        MaterialRegistry registry;
        registry.ReadParameters();

        amrex::ParmParse const pp_test("test");
        std::vector<std::string> expected_materials;
        pp_test.getarr("expected_materials", expected_materials);
        AMREX_ALWAYS_ASSERT(
            registry.size() == static_cast<int>(expected_materials.size()));
        for (int material = 0; material < registry.size(); ++material) {
            AMREX_ALWAYS_ASSERT(
                registry.material(material).name
                    == expected_materials[static_cast<std::size_t>(material)]);
        }

        if (registry.size() == 2) {
            auto const selector = registry.selector();
            AMREX_ALWAYS_ASSERT(selector.num_materials == registry.size());
            std::array<amrex::Real, 2> density{0.0_rt, 0.0_rt};
            AMREX_ALWAYS_ASSERT(
                registry.classifyResolvedCell(density) == vacuum);
            density = {2.0_rt, 0.0_rt};
            AMREX_ALWAYS_ASSERT(
                registry.classifyResolvedCell(density) == 0);
            density = {0.0_rt, 3.0_rt};
            AMREX_ALWAYS_ASSERT(
                registry.classifyResolvedCell(density) == 1);
            density = {2.0_rt, 3.0_rt};
            AMREX_ALWAYS_ASSERT(
                registry.classifyResolvedCell(density) == mixed);
            density = {2.0_rt, 1.0e-14_rt};
            AMREX_ALWAYS_ASSERT(
                registry.classifyResolvedCell(density) == 0);

            // Both input-order tests must resolve the same physical carrier
            // to the same lexicographic material ID.
            AMREX_ALWAYS_ASSERT(registry.material(0).name == "foam");
            AMREX_ALWAYS_ASSERT(registry.material(1).name == "tungsten");

            auto const* tungsten = registry.findMaterial("tungsten");
            if (tungsten != nullptr) {
                AMREX_ALWAYS_ASSERT(tungsten->electron_eos.has_value());
                AMREX_ALWAYS_ASSERT(tungsten->opacity.has_value());
                AMREX_ALWAYS_ASSERT(tungsten->electron_eos->material_id == 74);
                AMREX_ALWAYS_ASSERT(tungsten->opacity->material_key == "W");
            }
            AMREX_ALWAYS_ASSERT(tungsten != nullptr);
        }

        std::cout << "material registry:";
        for (int material = 0; material < registry.size(); ++material) {
            std::cout << " " << registry.material(material).name;
        }
        std::cout << " PASS\n";
    }
    amrex::Finalize();
    return 0;
}

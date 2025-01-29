/* Copyright 2019-2020 Axel Huebl, Ligia Diana Amorim, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorDensity.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cctype>
#include <vector>

using namespace amrex;

void InjectorDensity::clear ()
{
    switch (type)
    {
    case Type::parser:
    {
        break;
    }
    case Type::predefined:
    {
        object.predefined.clear();
        break;
    }
    default:
        return;
    }
}

InjectorDensityPredefined::InjectorDensityPredefined (
    std::string const& a_species_name) noexcept
{
    const ParmParse pp_species_name(a_species_name);

    std::vector<amrex::Real> v;
    // Read parameters for the predefined plasma profile.
    utils::parser::getArrWithParser(
        pp_species_name, "predefined_profile_params", v);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(v.size() <= 6,
                                     "Too many parameters for InjectorDensityPredefined");
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        p[i] = v[i];
    }

    // Parse predefined profile name, and update member variable profile.
    std::string which_profile_s;
    pp_species_name.query("predefined_profile_name", which_profile_s);
    std::transform(which_profile_s.begin(), which_profile_s.end(),
                   which_profile_s.begin(), ::tolower);
    if (which_profile_s == "parabolic_channel"){
        profile = Profile::parabolic_channel;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(v.size() > 5,
            "InjectorDensityPredefined::parabolic_channel: not enough parameters");
    }
}

// Note that we are not allowed to have non-trivial destructor.
// So we rely on clear() to free memory if needed.
void InjectorDensityPredefined::clear ()
{
}

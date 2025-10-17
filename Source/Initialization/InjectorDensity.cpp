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
    case Type::fromfile:
    {
        object.fromfile.clear();
        break;
    }
    default:
        return;
    }
}

void InjectorDensity::prepare (amrex::BoxArray const& grids,
                               amrex::DistributionMapping const& dmap,
                               amrex::IntVect const& ngrow,
                               std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (type == Type::fromfile) {
        object.fromfile.prepare(grids,dmap,ngrow,get_zlab);
    }
}

void InjectorDensity::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                               std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (type == Type::fromfile) {
        object.fromfile.prepare(pbox, moving_dir, moving_sign, get_zlab);
    }
}

void InjectorDensity::prepare (int li)
{
    if (type == Type::fromfile) {
        object.fromfile.prepare(li);
    }
}

bool InjectorDensity::needPreparation () const
{
    if (type == Type::fromfile) {
        return true;
    } else {
        return false;
    }
}

bool InjectorDensity::distributed () const
{
    if (type == Type::fromfile) {
        return object.fromfile.distributed();
    } else {
        return false;
    }
}

InjectorDensityPredefined::InjectorDensityPredefined (
    std::string const& a_species_name)
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

InjectorDensityFromFile::InjectorDensityFromFile (std::string const& a_file_name,
                                                  amrex::Geometry const& a_geom,
                                                  bool a_distributed)
{
    m_external_field_reader = new ExternalFieldReader
        (a_file_name, "density", "", a_geom.ProbLoArray(), a_geom.CellSizeArray(),
         amrex::convert(a_geom.Domain(),amrex::IntVect(1)), a_distributed);
}

void InjectorDensityFromFile::clear ()
{
    delete m_external_field_reader;
    m_external_field_reader = nullptr;
}

void InjectorDensityFromFile::prepare (amrex::BoxArray const& grids,
                                       amrex::DistributionMapping const& dmap,
                                       amrex::IntVect const& ngrow,
                                       std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (m_external_field_reader) {
        m_external_field_reader->prepare(grids,dmap,ngrow,get_zlab);
        m_external_field_view = m_external_field_reader->getView();
    }
}

void InjectorDensityFromFile::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                       std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (m_external_field_reader) {
        m_external_field_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_external_field_view = m_external_field_reader->getView();
    }
}

void InjectorDensityFromFile::prepare (int li)
{
    if (m_external_field_reader) {
        m_external_field_view = m_external_field_reader->getView(li);
    }
}

bool InjectorDensityFromFile::distributed () const
{
    if (m_external_field_reader) {
        return m_external_field_reader->distributed();
    } else {
        return false;
    }
}

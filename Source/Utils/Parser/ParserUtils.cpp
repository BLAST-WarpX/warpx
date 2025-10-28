/* Copyright 2022 Andrew Myers, Burlen Loring, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParserUtils.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Parser.H>
#include <AMReX_ParmParse.H>

#include <limits>
#include <map>
#include <set>

void utils::parser::Store_parserString(
    amrex::ParmParse const& pp,
    std::string const& query_string,
    std::string& stored_string)
{
    std::vector<std::string> f;
    pp.getarr(query_string.c_str(), f);
    stored_string.clear();
    for (auto const& s : f) {
        stored_string += s;
    }
    f.clear();
}

void utils::parser::Store_parserString(
    amrex::ParmParse const& a_pp,
    std::string const& group,
    std::string const& query_string,
    std::string& stored_string)
{
    const bool is_specified_without_group = a_pp.contains(query_string.c_str());
    const std::string grp_str = group + "." + query_string;
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

    if (is_specified_without_group && !is_specified_with_group) {
        // If found without the group but not with the group, then use the one without the group.
        utils::parser::Store_parserString(a_pp, query_string, stored_string);
    } else {
        // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
        utils::parser::Store_parserString(a_pp, grp_str, stored_string);
    }
}

bool utils::parser::Query_parserString(
    amrex::ParmParse const& pp,
    std::string const& query_string,
    std::string& stored_string)
{
    bool const input_specified = pp.contains(query_string.c_str());
    if (input_specified) {
        stored_string.clear();
        utils::parser::Store_parserString(pp, query_string, stored_string);
    }
    return input_specified;
}

int utils::parser::query (const amrex::ParmParse& a_pp, std::string const& group, char const * str, std::string& val)
{
    const bool is_specified_without_group = a_pp.contains(str);
    const std::string grp_str = group + "." + std::string(str);
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

    if (is_specified_without_group && !is_specified_with_group) {
        // If found without the group but not with the group, then use the one without the group.
        return a_pp.query(str, val);
    } else {
        // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
        return a_pp.query(grp_str.c_str(), val);
    }
}

void utils::parser::get (const amrex::ParmParse& a_pp, std::string const& group, char const * str, std::string& val)
{
    const bool is_specified_without_group = a_pp.contains(str);
    const std::string grp_str = group + "." + std::string(str);
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

    if (is_specified_without_group && !is_specified_with_group) {
        // If found without the group but not with the group, then use the one without the group.
        a_pp.get(str, val);
    } else {
        // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
        a_pp.get(grp_str.c_str(), val);
    }
}

amrex::Parser utils::parser::makeParser (
    std::string const& parse_function, amrex::Vector<std::string> const& varnames)
{
    const amrex::ParmParse pp;
    return pp.makeParser(parse_function, varnames);
}

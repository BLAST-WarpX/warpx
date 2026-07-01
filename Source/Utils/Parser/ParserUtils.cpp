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

void utils::parser::getWithParser (const amrex::ParmParse& a_pp, char const * const str, std::string& val)
{
    // Get the value of the input parameter
    a_pp.get(str, val);

    while (val.find('{') != std::string::npos) {
        const auto i1 = val.rfind('{');
        const auto i2 = val.find('}', i1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(i2 != std::string::npos,
                                         "Bad format for input paramter " + std::string(str) + ", unclosed brace");
        const std::string var = val.substr(i1+1, i2-i1-1);
        const amrex::ParmParse pp_my_constants("my_constants");
        std::string replacer;
        pp_my_constants.get(var, replacer);
        val.replace(i1, i2-i1+1, replacer);
    }
}

int utils::parser::queryWithParser (const amrex::ParmParse& a_pp, char const * const str, std::string& val)
{
    const bool is_specified = a_pp.query(str, val);
    if (is_specified) {
        getWithParser (a_pp, str, val);
    }
    return is_specified;
}

void utils::parser::Store_parserString(
    amrex::ParmParse const& pp,
    std::string const& query_string,
    std::string& stored_string)
{
    std::vector<std::string> f;
    pp.getarr(query_string, f);
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
    const bool is_specified_without_group = a_pp.contains(query_string);
    const std::string grp_str = group + "." + query_string;
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str));

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
    bool const input_specified = pp.contains(query_string);
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
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str));

    if (is_specified_without_group && !is_specified_with_group) {
        // If found without the group but not with the group, then use the one without the group.
        return a_pp.query(str, val);
    } else {
        // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
        return a_pp.query(grp_str, val);
    }
}

void utils::parser::get (const amrex::ParmParse& a_pp, std::string const& group, char const * str, std::string& val)
{
    const bool is_specified_without_group = a_pp.contains(str);
    const std::string grp_str = group + "." + std::string(str);
    const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str));

    if (is_specified_without_group && !is_specified_with_group) {
        // If found without the group but not with the group, then use the one without the group.
        a_pp.get(str, val);
    } else {
        // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
        a_pp.get(grp_str, val);
    }
}

amrex::Parser utils::parser::makeParser (
    std::string const& parse_function, amrex::Vector<std::string> const& varnames)
{
    const amrex::ParmParse pp;
    return pp.makeParser(parse_function, varnames);
}

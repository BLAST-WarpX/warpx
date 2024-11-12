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

namespace {
    template< typename int_type >
    AMREX_FORCE_INLINE
    int_type safeCastTo(const amrex::Real x, const std::string& real_name) {
        auto result = int_type(0);
        bool error_detected = false;
        std::string assert_msg;
        // (2.0*(numeric_limits<int>::max()/2+1)) converts numeric_limits<int>::max()+1 to a real ensuring accuracy to all digits
        // This accepts x = 2**31-1 but rejects 2**31.
        using namespace amrex::literals;
        constexpr int_type half_max_plus_one = std::numeric_limits<int_type>::max()/2+1;
        constexpr amrex::Real max_range = (2.0_rt*static_cast<amrex::Real>(half_max_plus_one));
        if (x < max_range) {
            if (std::ceil(x) >= std::numeric_limits<int_type>::min()) {
                result = static_cast<int_type>(x);
            } else {
                error_detected = true;
                assert_msg = "Negative overflow detected when casting " + real_name + " = " +
                             std::to_string(x) + " to integer type";
            }
        } else if (x > 0) {
            error_detected = true;
            assert_msg =  "Overflow detected when casting " + real_name + " = " + std::to_string(x) + " to integer type";
        } else {
            error_detected = true;
            assert_msg =  "NaN detected when casting " + real_name + " to integer type";
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!error_detected, assert_msg);
        return result;
   }
}


int utils::parser::safeCastToInt(const amrex::Real x, const std::string& real_name) {
    return ::safeCastTo<int> (x, real_name);
}


long utils::parser::safeCastToLong(const amrex::Real x, const std::string& real_name) {
    return ::safeCastTo<long> (x, real_name);
}


amrex::Parser utils::parser::makeParser (
    std::string const& parse_function, amrex::Vector<std::string> const& varnames)
{
    // Since queryWithParser recursively calls this routine, keep track of symbols
    // in case an infinite recursion is found (a symbol's value depending on itself).
    static std::set<std::string> recursive_symbols;

    amrex::Parser parser(parse_function);
    parser.registerVariables(varnames);

    std::set<std::string> symbols = parser.symbols();
    for (auto const& v : varnames) { symbols.erase(v); }

    // User can provide inputs under this name, through which expressions
    // can be provided for arbitrary variables. PICMI inputs are aware of
    // this convention and use the same prefix as well. This potentially
    // includes variable names that match physical or mathematical
    // constants, in case the user wishes to enforce a different
    // system of units or some form of quasi-physical behavior in the
    // simulation. Thus, this needs to override any built-in
    // constants.
    const amrex::ParmParse pp_my_constants("my_constants");

    // Physical / Numerical Constants available to parsed expressions
    static std::map<std::string, amrex::Real> warpx_constants =
      {
       {"clight", PhysConst::c},
       {"epsilon0", PhysConst::ep0},
       {"mu0", PhysConst::mu0},
       {"q_e", PhysConst::q_e},
       {"m_e", PhysConst::m_e},
       {"m_p", PhysConst::m_p},
       {"m_u", PhysConst::m_u},
       {"kb", PhysConst::kb},
       {"pi", MathConst::pi},
      };

    for (auto it = symbols.begin(); it != symbols.end(); ) {
        // Always parsing in double precision avoids potential overflows that may occur when parsing
        // user's expressions because of the limited range of exponentials in single precision
        double v = 0.0;

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            recursive_symbols.count(*it)==0,
            "Expressions contains recursive symbol "+*it);
        recursive_symbols.insert(*it);
        const bool is_input = queryWithParser(pp_my_constants, it->c_str(), v);
        recursive_symbols.erase(*it);

        if (is_input) {
            parser.setConstant(*it, v);
            it = symbols.erase(it);
            continue;
        }

        const auto constant = warpx_constants.find(*it);
        if (constant != warpx_constants.end()) {
            parser.setConstant(*it, constant->second);
            it = symbols.erase(it);
            continue;
        }

        ++it;
    }
    for (auto const& s : symbols) {
        WARPX_ABORT_WITH_MESSAGE("makeParser::Unknown symbol "+s);
    }
    return parser;
}


double
utils::parser::parseStringtoDouble(const std::string& str)
{
    const auto parser = makeParser(str, {});
    const auto exe = parser.compileHost<0>();
    const auto result = exe();
    return result;
}


int
utils::parser::parseStringtoInt(const std::string& str, const std::string& name)
{
    const auto rval = static_cast<amrex::Real>(parseStringtoDouble(str));
    const auto ival = safeCastToInt(std::round(rval), name);
    return ival;
}

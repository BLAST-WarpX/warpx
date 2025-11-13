/* Copyright 2022 Andrew Myers, Burlen Loring, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_UTILS_PARSER_PARSERUTILS_H_
#define WARPX_UTILS_PARSER_PARSERUTILS_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>
#include <type_traits>

namespace utils::parser
{
    /**
    * \brief Initialize an amrex::Parser object from a string containing a math expression
    *
    * \param parse_function String to read to initialize the parser.
    * \param varnames A list of predefined independent variables
    */
    amrex::Parser makeParser (
        std::string const& parse_function,
        amrex::Vector<std::string> const& varnames);


    /**
    * \brief Parse a string (typically a mathematical expression) from the
    * input file and store it into a variable.
    *
    * \param pp used to read the query_string `pp.<function>=string`
    * \param query_string ParmParse.query will look for this string
    * \param stored_string variable in which the string to parse is stored
    */
    void Store_parserString(
        amrex::ParmParse const& pp,
        std::string const& query_string,
        std::string& stored_string);

    /**
     * \brief Parse a string (typically a mathematical expression) from the
     * input file and store it into a variable.
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * get(pp, "group", "name", val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] pp used to read the query_string `pp.<function>=string`
     * \param[in] group name of the optional group
     * \param[in] query_string ParmParse.query will look for this string
     * \param[out] stored_string variable in which the string to parse is stored
     */
    void Store_parserString(
        const amrex::ParmParse &pp,
        std::string const& group,
        std::string const& query_string,
        std::string& stored_string);

    /**
    * \brief If the input is provided, parse the string (typically a mathematical expression) from the
    * input file and store it into a variable, replacing its contents.
    *
    * \param pp used to read the query_string `pp.<function>=string`
    * \param query_string ParmParse.query will look for this string
    * \param stored_string variable in which the string to parse is stored
    */
    bool Query_parserString(
        amrex::ParmParse const& pp,
        std::string const& query_string,
        std::string& stored_string);

    template <int N>
    amrex::ParserExecutor<N> compileParser (amrex::Parser const* parser)
    {
        if (parser) {
            return parser->compile<N>();
        } else {
            return amrex::ParserExecutor<N>{};
        }
    }


    /** Similar to amrex::ParmParse::queryAsDouble
     *
     * amrex::ParmParse::query reads a name and a value from the input file. This function does the
     * same, and applies the amrex::Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored, either a scalar or vector
     */
    template <typename T>
    int queryWithParser (const amrex::ParmParse& a_pp, char const * const str, T& val)
    {
        return a_pp.queryAsDouble(str, val);
    }


    template <typename T>
    int queryArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<T>& val)
    {
        auto nvals = a_pp.countval(str);
        if (nvals > 0) {
            val.resize(nvals);
            return a_pp.queryarrAsDouble(str, nvals, val.data());
        } else {
            return 0;
        }
    }


    /** Similar to amrex::ParmParse::queryarrWithParser
     *
     * amrex::ParmParse::query reads a name and a value from the input file. This function does the
     * same, and applies the amrex::Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored, either a scalar or vector
     * \param[in] start_ix start index in the list of inputs values (optional with arrays, default is
     * amrex::ParmParse::FIRST for starting with the first input value)
     * \param[in] num_val number of input values to use (optional with arrays, default is
     * amrex::ParmParse::LAST for reading until the last input value)
     */
    template <typename T>
    int queryArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<T>& val,
                            const int start_ix, const int num_val)
    {
        const int is_specified = queryArrWithParser(a_pp, str, val);
        if (is_specified)
        {
            val.erase(val.begin(), val.begin()+start_ix);
            if (num_val != amrex::ParmParse::LAST) {
                if (num_val > int(val.size())) { return 0; }
                val.resize(num_val);
            }
        }
        // return the same output as amrex::ParmParse::query
        return is_specified;
    }


    /** Similar to amrex::ParmParse::getWithParser
     *
     * amrex::ParmParse::get reads a name and a value from the input file. This function does the
     * same, and applies the Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     */
    template <typename T>
    void getWithParser (const amrex::ParmParse& a_pp, char const * const str, T& val)
    {
        a_pp.getAsDouble(str, val);
    }

    template <typename T>
    void getArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<T>& val)
    {
        int const is_specified = queryArrWithParser(a_pp, str, val);
        if (is_specified == 0) {
            throw std::runtime_error("utils::parser::getArrWithParser failed");
        }
    }


    /** Similar to amrex::ParmParse::getarrWithParser
     *
     * amrex::ParmParse::get reads a name and a value from the input file. This function does the
     * same, and applies the Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     * \param[in] start_ix start index in the list of inputs values (optional with arrays, default is
     * amrex::ParmParse::FIRST for starting with the first input value)
     * \param[in] num_val number of input values to use (optional with arrays, default is
     * amrex::ParmParse::LAST for reading until the last input value)
     */
    template <typename T>
    void getArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<T>& val,
                        const int start_ix, const int num_val)
    {
        int const is_specified = queryArrWithParser(a_pp, str, val, start_ix, num_val);
        if (is_specified == 0) {
            throw std::runtime_error("utils::parser::getArrWithParser failed");
        }
    }


    /** Similar to amrex::ParmParse::queryWithParser
     *
     * amrex::ParmParse::query reads a name and a value from the input file. This function does the
     * same, and applies the amrex::Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * queryWithParser(pp, "group", "name", val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored, either a scalar or vector
     */
    template <typename T>
    int queryWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, T& val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            return queryWithParser(a_pp, str, val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            return queryWithParser(a_pp, grp_str.c_str(), val);
        }
    }


    template <typename T>
    int queryArrWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, std::vector<T>& val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            return queryArrWithParser(a_pp, str, val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            return queryArrWithParser(a_pp, grp_str.c_str(), val);
        }
    }


    /** Similar to amrex::ParmParse::queryarrWithParser
     *
     * amrex::ParmParse::query reads a name and a value from the input file. This function does the
     * same, and applies the amrex::Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * queryArrWithParser(pp, "group", "name", val, start_ix, num_val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored, either a scalar or vector
     * \param[in] start_ix start index in the list of inputs values (optional with arrays, default is
     * amrex::ParmParse::FIRST for starting with the first input value)
     * \param[in] num_val number of input values to use (optional with arrays, default is
     * amrex::ParmParse::LAST for reading until the last input value)
     */
    template <typename T>
    int queryArrWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, std::vector<T>& val,
                            const int start_ix, const int num_val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            return queryArrWithParser(a_pp, str, val, start_ix, num_val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            return queryArrWithParser(a_pp, grp_str.c_str(), val, start_ix, num_val);
        }
    }

    /** Wraps around amrex::ParmParse::query, but also supports an option group name
     *
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * get(pp, "group", "name", val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     */
    int query (const amrex::ParmParse& a_pp, std::string const& group, char const * str, std::string& val);

    /** Similar to amrex::ParmParse::getWithParser
     *
     * amrex::ParmParse::get reads a name and a value from the input file. This function does the
     * same, and applies the Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * getWithParser(pp, "group", "name", val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     */
    template <typename T>
    void getWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, T& val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            getWithParser(a_pp, str, val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            getWithParser(a_pp, grp_str.c_str(), val);
        }
    }

    template <typename T>
    void getArrWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, std::vector<T>& val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            getArrWithParser(a_pp, str, val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            getArrWithParser(a_pp, grp_str.c_str(), val);
        }
    }


    /** Similar to amrex::ParmParse::getarrWithParser
     *
     * amrex::ParmParse::get reads a name and a value from the input file. This function does the
     * same, and applies the Parser to the value, so the user has the choice to specify a value or
     * a math expression (including user-defined constants).
     * Works for amrex::Real numbers and integers.
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * getArrWithParser(pp, "group", "name", val, start_ix, num_val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     * \param[in] start_ix start index in the list of inputs values (optional with arrays, default is
     * amrex::ParmParse::FIRST for starting with the first input value)
     * \param[in] num_val number of input values to use (optional with arrays, default is
     * amrex::ParmParse::LAST for reading until the last input value)
     */
    template <typename T>
    void getArrWithParser (const amrex::ParmParse& a_pp, std::string const& group, char const * const str, std::vector<T>& val,
                        const int start_ix, const int num_val)
    {
        const bool is_specified_without_group = a_pp.contains(str);
        const std::string grp_str = group + "." + std::string(str);
        const bool is_specified_with_group = (group.empty() ? false : a_pp.contains(grp_str.c_str()));

        if (is_specified_without_group && !is_specified_with_group) {
            // If found without the group but not with the group, then use the one without the group.
            getArrWithParser(a_pp, str, val, start_ix, num_val);
        } else {
            // Otherwise, use the one with the group even if not found, in which case an exception may be raised.
            getArrWithParser(a_pp, grp_str.c_str(), val, start_ix, num_val);
        }
    }

    /** Wraps around amrex::ParmParse::get, but also supports an option group name
     *
     * The group name specified is optional part of the parameter name. A parameter that includes the group
     * name takes precedence over one without the group name. If this routine is called like
     * get(pp, "group", "name", val), it will query both "group.name" or "name" and return
     * the value of "group.name" if found, otherwise the value of "name".
     *
     * \param[in] a_pp amrex::ParmParse object
     * \param[in] group name of the optional group
     * \param[in] str name of the parameter to read
     * \param[out] val where the value queried and parsed is stored
     */
    void get (amrex::ParmParse const& a_pp, std::string const& group, char const * str, std::string& val);

}

#endif // WARPX_UTILS_PARSER_PARSERUTILS_H_

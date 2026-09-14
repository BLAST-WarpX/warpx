/* Copyright 2022 The ABLASTR Community
 *
 * This file is part of ABLASTR.
 *
 * Authors: Neil Zaim, Luca Fedeli, Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */

#include "IntervalsParser.H"

#include "ablastr/utils/TextMsg.H"
#include "StringUtils.H"

#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>


namespace ablastr::utils::text
{

SliceParser::SliceParser (const std::string& instr, const bool require_stop):
    m_require_stop{require_stop}
{
    const amrex::ParmParse pp;

    // split string and trim whitespaces
    auto insplit = split_string<std::vector<std::string>>(instr, m_separator, true);

    if(insplit.size() == 1){ // no colon in input string. The input is the period.
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!m_require_stop,
            "interval stop required but not specified in '" + instr + "'");
        m_period = int(std::round(pp.eval<double>(insplit[0])));}
    else if(insplit.size() == 2) // 1 colon in input string. The input is start:stop
    {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!m_require_stop || !insplit[1].empty(),
            "interval stop required but not specified in '" + instr + "'");
        if (!insplit[0].empty()){
            m_start = int(std::round(pp.eval<double>(insplit[0])));}
        if (!insplit[1].empty()){
            m_stop = int(std::round(pp.eval<double>(insplit[1])));}
    }
    else // 2 colons in input string. The input is start:stop:period
    {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!m_require_stop || !insplit[1].empty(),
            "interval stop required but not specified in '" + instr + "'");
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
            insplit.size() == 3,
            instr + "' is not a valid syntax for a slice.");
        if (!insplit[0].empty()){
            m_start = int(std::round(pp.eval<double>(insplit[0])));}
        if (!insplit[1].empty()){
            m_stop = int(std::round(pp.eval<double>(insplit[1])));}
        if (!insplit[2].empty()){
            m_period = int(std::round(pp.eval<double>(insplit[2])));}
    }
}


bool SliceParser::contains (const int n) const
{
    if (m_period <= 0) {return false;}
    return (n - m_start) % m_period == 0 && n >= m_start && n <= m_stop;
}


int SliceParser::nextContains (const int n) const
{
    if (m_period <= 0) {return std::numeric_limits<int>::max();}
    int next = m_start;
    if (n >= m_start) {next = ((n-m_start)/m_period + 1)*m_period+m_start;}
    if (next > m_stop) {next = std::numeric_limits<int>::max();}
    return next;
}


int SliceParser::previousContains (const int n) const
{
    if (m_period <= 0) {return 0;}
    int previous = ((std::min(n-1,m_stop)-m_start)/m_period)*m_period+m_start;
    if ((n < m_start) || (previous < 0)) {previous = 0;}
    return previous;
}


int SliceParser::getPeriod () const {return m_period;}


int SliceParser::getStart () const {return m_start;}


int SliceParser::getStop () const {return m_stop;}


int SliceParser::numContained () const {
    return (m_stop - m_start) / m_period + 1;}


IntervalsParser::IntervalsParser (const std::vector<std::string>& instr_vec)
{
    std::string inconcatenated;
    for (const auto& instr_element : instr_vec) { inconcatenated +=instr_element; }

    auto insplit = split_string<std::vector<std::string>>(inconcatenated, m_separator);

    for(const auto& inslc : insplit)
    {
        const SliceParser temp_slice(inslc);
        m_slices.push_back(temp_slice);
        if ((temp_slice.getPeriod() > 0) &&
               (temp_slice.getStop() >= temp_slice.getStart())) { m_activated = true; }
    }
}


bool IntervalsParser::contains (const int n) const
{
    return std::any_of(m_slices.begin(), m_slices.end(),
        [&](const auto& slice){return slice.contains(n);});
}


int IntervalsParser::nextContains (const int n) const
{
    int next = std::numeric_limits<int>::max();
    for(const auto& slice: m_slices){
        next = std::min(slice.nextContains(n),next);
    }
    return next;
}


int IntervalsParser::previousContains (const int n) const
{
    int previous = 0;
    for(const auto& slice: m_slices){
        previous = std::max(slice.previousContains(n),previous);
    }
    return previous;
}


int IntervalsParser::previousContainsInclusive (const int n) const
{
    if (contains(n)){return n;}
    else {return previousContains(n);}
}


int IntervalsParser::localPeriod (const int n) const
{
    return nextContains(n) - previousContainsInclusive(n);
}


bool IntervalsParser::isActivated () const {return m_activated;}

} // namespace ablastr::utils::text

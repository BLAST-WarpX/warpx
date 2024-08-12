/* Copyright 2022 Andrew Myers, Burlen Loring, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "IntervalsParser.H"

#include "ablastr/utils/text/StringUtils.H"
#include "ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_Utility.H>

#include <algorithm>

utils::parser::SliceParser::SliceParser (const std::string& instr, const bool isBTD):
    m_isBTD{isBTD}
{
    const amrex::ParmParse pp;

    // split string and trim whitespaces
    auto insplit = ablastr::utils::text::split_string<std::vector<std::string>>(
        instr, m_separator, true);

    if(insplit.size() == 1){ // no colon in input string. The input is the period.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_isBTD, "must specify interval stop for BTD");
        m_period = int(std::round(pp.eval<double>(insplit[0])));}
    else if(insplit.size() == 2) // 1 colon in input string. The input is start:stop
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_isBTD || !insplit[1].empty(), "must specify interval stop for BTD");
        if (!insplit[0].empty()){
            m_start = int(std::round(pp.eval<double>(insplit[0])));}
        if (!insplit[1].empty()){
            m_stop = int(std::round(pp.eval<double>(insplit[1])));}
    }
    else // 2 colons in input string. The input is start:stop:period
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_isBTD || !insplit[1].empty(), "must specify interval stop for BTD");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
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


bool utils::parser::SliceParser::contains (const int n) const
{
    if (m_period <= 0) {return false;}
    return (n - m_start) % m_period == 0 && n >= m_start && n <= m_stop;
}


int utils::parser::SliceParser::nextContains (const int n) const
{
    if (m_period <= 0) {return std::numeric_limits<int>::max();}
    int next = m_start;
    if (n >= m_start) {next = ((n-m_start)/m_period + 1)*m_period+m_start;}
    if (next > m_stop) {next = std::numeric_limits<int>::max();}
    return next;
}


int utils::parser::SliceParser::previousContains (const int n) const
{
    if (m_period <= 0) {return false;}
    int previous = ((std::min(n-1,m_stop)-m_start)/m_period)*m_period+m_start;
    if ((n < m_start) || (previous < 0)) {previous = 0;}
    return previous;
}


int utils::parser::SliceParser::getPeriod () const {return m_period;}


int utils::parser::SliceParser::getStart () const {return m_start;}


int utils::parser::SliceParser::getStop () const {return m_stop;}


int utils::parser::SliceParser::numContained () const {
    return (m_stop - m_start) / m_period + 1;}

utils::parser::IntervalsParser::IntervalsParser (
    const std::vector<std::string>& instr_vec)
{
    std::string inconcatenated;
    for (const auto& instr_element : instr_vec) { inconcatenated +=instr_element; }

    auto insplit = ablastr::utils::text::split_string<std::vector<std::string>>(
        inconcatenated, m_separator);

    for(const auto& inslc : insplit)
    {
        const SliceParser temp_slice(inslc);
        m_slices.push_back(temp_slice);
        if ((temp_slice.getPeriod() > 0) &&
               (temp_slice.getStop() >= temp_slice.getStart())) { m_activated = true; }
    }
}


bool utils::parser::IntervalsParser::contains (const int n) const
{
    return std::any_of(m_slices.begin(), m_slices.end(),
        [&](const auto& slice){return slice.contains(n);});
}


int utils::parser::IntervalsParser::nextContains (const int n) const
{
    int next = std::numeric_limits<int>::max();
    for(const auto& slice: m_slices){
        next = std::min(slice.nextContains(n),next);
    }
    return next;
}


int utils::parser::IntervalsParser::previousContains (const int n) const
{
    int previous = 0;
    for(const auto& slice: m_slices){
        previous = std::max(slice.previousContains(n),previous);
    }
    return previous;
}


int utils::parser::IntervalsParser::previousContainsInclusive (
    const int n) const
{
    if (contains(n)){return n;}
    else {return previousContains(n);}
}


int utils::parser::IntervalsParser::localPeriod (const int n) const
{
    return nextContains(n) - previousContainsInclusive(n);
}


bool utils::parser::IntervalsParser::isActivated () const {return m_activated;}


utils::parser::BTDIntervalsParser::BTDIntervalsParser (
    const std::vector<std::string>& instr_vec)
{
    std::string inconcatenated;
    for (const auto& instr_element : instr_vec) { inconcatenated +=instr_element; }

    auto const insplit = ablastr::utils::text::split_string<std::vector<std::string>>(
        inconcatenated, std::string(1,m_separator));

    // parse the Intervals string into Slices and store each slice in m_slices,
    // in order of increasing Slice start value
    for(const auto& inslc : insplit)
    {
        const bool isBTD = true;
        const SliceParser temp_slice(inslc, isBTD);
        if (!m_slices.empty())
        {
            // find the last index i_slice where
            // the start value of m_slices[i_slice] is greater than temp_slices' start_value
            int i_slice = 0;
            while (temp_slice.getStart() > m_slices[i_slice].getStart() && i_slice < static_cast<int>(m_slices.size()))
            {
                i_slice++;
            }
            m_slices.insert(m_slices.begin() + i_slice, temp_slice);
        }
        else
        {
            m_slices.push_back(temp_slice);
        }
    }
    // from the vector of slices, m_slices,
    // create a vector of integers, m_btd_iterations, containing
    // the iteration of every back-transformed snapshot that will be saved
    // the iteration values in m_btd_iterations are
    // 1. saved in increasing order
    // 2. unique, i.e. no duplicate iterations are saved
    for (const auto& temp_slice : m_slices)
    {
        const int start = temp_slice.getStart();
        const int period = temp_slice.getPeriod();
        int btd_iter_ind;
        // Implementation note:
        // start at the end of m_btd_iterations and search backward
        // with the thinking that the user mostly lists slices in ascending order
        //
        // for Slice temp_slice in m_slices,
        // determine the largest index in m_btd_iterations smaller than start
        if (m_btd_iterations.empty())
        {
            btd_iter_ind = 0;
        }
        else
        {
            btd_iter_ind = static_cast<int>(m_btd_iterations.size() - 1);
            while (start < m_btd_iterations.at(btd_iter_ind) and btd_iter_ind>0)
            {
                btd_iter_ind--;
            }
        }
        // insert each iteration contained in temp_slice into m_btd_iterations
        // adding them in increasing sorted order and not adding any iterations
        // already contained in m_btd_iterations
        for (int slice_iter = start; slice_iter <= temp_slice.getStop(); slice_iter += period)
        {
            // number of iterations currently in this slice
            auto const num_btd_iterations = static_cast<int>(m_btd_iterations.size());

            if (num_btd_iterations > 0)
            {
                // increment btd_iter_ind for each existing iteration,
                // if slice_iter is larger than an existing one
                while (btd_iter_ind < num_btd_iterations &&
                       slice_iter > m_btd_iterations.at(btd_iter_ind))
                {
                    btd_iter_ind++;
                }
                // this is the place to insert slice_iter if it is not in m_btd_iterations
                // if slice_iter > all entries in m_btd_iterations, append slice_iter
                if (btd_iter_ind == num_btd_iterations)
                {
                    m_btd_iterations.insert(m_btd_iterations.begin() + btd_iter_ind, slice_iter);
                } else
                {
                    if (slice_iter != m_btd_iterations.at(btd_iter_ind))
                    {
                        m_btd_iterations.insert(m_btd_iterations.begin() + btd_iter_ind, slice_iter);
                    }
                }
            } else
            {
                m_btd_iterations.push_back(slice_iter);
            }
        }
        if (temp_slice.getPeriod() > 0 &&
            temp_slice.getStop() >= start)
        {
            m_activated = true;
        }
    }
}


int utils::parser::BTDIntervalsParser::NumSnapshots () const
{
    return static_cast<int>(m_btd_iterations.size());
}


int utils::parser::BTDIntervalsParser::GetBTDIteration (int i_buffer) const
{
    return m_btd_iterations[i_buffer];
}


int utils::parser::BTDIntervalsParser::GetFinalIteration () const
{
    return m_btd_iterations.back();
}


bool utils::parser::BTDIntervalsParser::isActivated () const {return m_activated;}

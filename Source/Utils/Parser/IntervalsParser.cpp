/* Copyright 2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Neil Zaim, Luca Fedeli, Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */

#include "IntervalsParser.H"

#include <ablastr/utils/text/StringUtils.H>

#include <string>


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
        constexpr bool require_stop = true;
        const SliceParser temp_slice(inslc, require_stop);
        if (!m_slices.empty())
        {
            // find the last index i_slice where
            // the start value of m_slices[i_slice] is greater than temp_slices' start_value
            int i_slice = 0;
            while (i_slice < static_cast<int>(m_slices.size()) && temp_slice.getStart() > m_slices[i_slice].getStart())
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

/* Copyright 2022 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_UTILS_TEXT_STRINGUTILS_H_
#define ABLASTR_UTILS_TEXT_STRINGUTILS_H_

#include <AMReX_Utility.H>

#include <cstddef>
#include <string>
#include <vector>

namespace ablastr::utils::text
{
    /** \brief Splits a string using a string separator. This is somewhat similar to
     *  amrex::Tokenize. The main difference is that, if the separator ":" is used,
     *  amrex::Tokenize will split ":3::2" into ["3","2"] while this functio will
     *  split ":3::2" into ["","3","","2"]. This function can also perform a trimming to
     *  remove whitespaces (or any other arbitrary string) from the split string.
     *
     * @tparam Container the type of the split string.
     *
     * @param[in] instr the input string
     * @param[in] separator the separator string
     * @param[in] trim true to trim the split string, false otherwise.
     * @param[in] trim_space the string to trim if trim is true.
     * @return cont the split string
     */
    template <typename Container>
    auto split_string (std::string const& instr, std::string const& separator,
                  bool const trim = false, std::string const& trim_space = " \t")
    {
        Container cont;
        std::size_t current = instr.find(separator);
        std::size_t previous = 0;
        while (current != std::string::npos) {
            if (trim){
                cont.push_back(amrex::trim(instr.substr(previous, current - previous),trim_space));}
            else{
                cont.push_back(instr.substr(previous, current - previous));}
            previous = current + separator.size();
            current = instr.find(separator, previous);
        }
        if (trim){
            cont.push_back(amrex::trim(instr.substr(previous, current - previous),trim_space));}
        else{
            cont.push_back(instr.substr(previous, current - previous));}
        return cont;
    }

    /** \brief This function performs automatic text wrapping on a string,
     *  returning an array of strings each not exceeding the maximum line length
     *  (unless the text contains a word exceeding the maximum line length).
     *
     * @param[in] text the string containing the text to be wrapped
     * @param[in] max_line_length the maximum line length
     * @return an std::vector containing the lines of the wrapped text
     */
    std::vector<std::string> automatic_text_wrap(
        const std::string& text, int max_line_length);

}

#endif // ABLASTR_UTILS_TEXT_STRINGUTILS_H_

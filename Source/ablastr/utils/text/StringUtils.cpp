/* Copyright 2022 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "StringUtils.H"

#include <sstream>

std::vector<std::string>
ablastr::utils::text::automatic_text_wrap(
    const std::string& text, const int max_line_length){

    auto ss_text = std::stringstream{text};
    auto wrapped_text_lines = std::vector<std::string>{};

    std::string line;
    while(std::getline(ss_text, line,'\n')){

        auto ss_line = std::stringstream{line};
        int counter = 0;
        std::stringstream ss_line_out;
        std::string word;

        while (ss_line >> word){
            const auto wlen = static_cast<int>(word.length());

            if(counter == 0){
                ss_line_out << word;
                counter += wlen;
            }
            else{
                if (counter + wlen < max_line_length){
                    ss_line_out << " " << word;
                    counter += (wlen+1);
                }
                else{
                    wrapped_text_lines.push_back(ss_line_out.str());
                    ss_line_out.str("");
                    ss_line_out << word;
                    counter = wlen;
                }
            }
        }

        wrapped_text_lines.push_back(ss_line_out.str());
    }

    return wrapped_text_lines;
}

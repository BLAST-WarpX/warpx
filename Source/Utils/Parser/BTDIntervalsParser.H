/* Copyright 2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Neil Zaim, Luca Fedeli, Weiqun Zhang, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_UTILS_PARSER_BTDINTERVALSPARSER_H_
#define WARPX_UTILS_PARSER_BTDINTERVALSPARSER_H_

#include <ablastr/utils/text/IntervalsParser.H>

#include <string>
#include <vector>

namespace utils::parser
{
    /**
     * \brief This class is a parser for multiple slices of the form x,y,z,... where x, y and z are
     * slices of the form i:j:k, as defined in the SliceParser class. This class contains a vector of
     * SliceParsers. The supported function set differs from the IntervalsParser
     */
    class BTDIntervalsParser
    {
    public:
        /**
        * \brief Default constructor of the BTDIntervalsParser class.
        */
        BTDIntervalsParser () = default;

        /**
        * \brief Constructor of the BTDIntervalsParser class.
        *
        * @param[in] instr_vec an input vector string, which when concatenated is of the form
        * "x,y,z,...". This will call the constructor of SliceParser using x, y and z as input
        * arguments.
        */
        explicit BTDIntervalsParser (const std::vector<std::string>& instr_vec);

        /**
         * @brief Return the total number of unique labframe snapshots
         */
        [[nodiscard]] int NumSnapshots () const;

        /**
         * @brief Return the iteration number stored at index i_buffer
         *
         * @param i_buffer buffer or iteration index, between 0 and NumSnapshots
         */
        [[nodiscard]] int GetBTDIteration(int i_buffer) const;

        /**
         * @brief Return the final BTD iteration
         *
         */
        [[nodiscard]] int GetFinalIteration() const;

        /**
        * \brief A method that returns true if any of the slices contained by the IntervalsParser
        * has a strictly positive period.
        */
        [[nodiscard]] bool isActivated () const;

    private:
        std::vector<int> m_btd_iterations;
        std::vector<ablastr::utils::text::SliceParser> m_slices;
        std::vector<int> m_slice_starting_i_buffer;
        static constexpr char m_separator = ',';
        bool m_activated = false;
    };
}

#endif // WARPX_UTILS_PARSER_BTDINTERVALSPARSER_H_

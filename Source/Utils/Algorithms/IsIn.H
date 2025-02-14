/* Copyright 2022 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_UTILS_ALGORITHMS_ISIN_H_
#define WARPX_UTILS_ALGORITHMS_ISIN_H_

#include <algorithm>
#include <vector>

namespace utils::algorithms
{
    /** \brief Returns true if an item of type TE is in a vector
     * of TV objects (provided that TE can be converted into TV), false otherwise
     *
     * @tparam TV the typename of the vector elements
     * @tparam TE the typename of the item
     *
     * @param vect a vector of TV objects
     * @param elem an object of type TE
     *
     * @return true if elem is in vect, false otherwise
     */
    template <typename TV, typename TE,
        class = typename std::enable_if_t<std::is_convertible_v<TE,TV>>>
    bool is_in(const std::vector<TV>& vect,
               const TE& elem)
    {
        return (std::find(vect.begin(), vect.end(), elem) != vect.end());
    }


    /** \brief Returns true if any of the items of a vector<TE> is contained
     * in another vector<TV> (provided that TE can be converted into TV)
     *
     * @tparam TV the typename of the first vector elements
     * @tparam TV the typename of the second vector elements
     *
     * @param vect a vector of TV objects
     * @param elems a vector of TE objects
     *
     * @return true if any element of elems is in vect, false otherwise
     */
    template <typename TV, typename TE,
        class = typename std::enable_if_t<std::is_convertible_v<TE,TV>>>
    bool any_of_is_in(const std::vector<TV>& vect,
               const std::vector<TE>& elems)
    {
        return std::any_of(elems.begin(), elems.end(),
            [&](const auto elem){return is_in(vect, elem);});
    }
}

#endif //WARPX_UTILS_ALGORITHMS_ISIN_H_

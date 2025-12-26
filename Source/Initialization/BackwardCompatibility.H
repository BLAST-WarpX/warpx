/* Copyright 2025 Luca Fedeli, Arianna Formenti, Remi Lehe,  Olga Shapoval
 * Davide Terzani, Maxence Thevenet, Axel Huebl, Edoardo Zoni
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

namespace warpx::initialization
{
    /** This function queries deprecated input parameters and abort
     *  the run or generates a warning if one of them is specified. */
    void backward_compatibility ();
}

/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PUSHTYPE_H_
#define WARPX_PUSHTYPE_H_

// Specify which scheme to use for the particle advance
enum struct PushType : int
{
    Explicit = 0, // Use the standard leap-frog scheme
    Implicit      // Use the Crank-Nicolson scheme.
                  // See for example Eqs. 15-18 in Chen, JCP 407 (2020) 109228
};

#endif // WARPX_PUSHTYPE_H_

# Copyright 2020 Edoardo Zoni
#
# This file is part of WarpX
#
# License: BSD-3-Clause-LBNL

# -------------------------------------------------------------------------------
# Compute interpolation points and weights for coarsening and refinement in IO
# and MR applications in 1D (extensions to 2D and 3D are trivial). Weights are
# computed in order to guarantee total charge conservation for both cell-centered
# data (equal weights) and nodal data (weights depend on distance between points
# on fine and coarse grids).
#
# Notation:
# - index i  refers to points on coarse grid
# - index ii refers to points on fine   grid
# - sc denotes the staggering of the coarse data
# - sf denotes the staggering of the fine   data
# - cr denotes the coarsening ratio (must be cr=1,2,4 only)
#
# For MR applications only the cases sc=sf=0 and sc=sf=1 are considered. Terms
# multiplied by (1-sf)*(1-sc) are ON for cell-centered data and OFF for nodal data,
# while terms multiplied by sf*sc are ON for nodal data and OFF for cell-centered
# data. C++ implementation in Source/ablastr/coarsen/average.(H/.cpp) and
# Source/ablastr/coarsen/sample.(H/.cpp)
# -------------------------------------------------------------------------------

import sys

import numpy as np


# Fine grid limits (without ghost cells)
def coarsening_fine_grid_limits(sc, sf, cr):
    if sf == 0:  # cell-centered
        iimin = 0
        iimax = 4 * cr - 1
    elif sf == 1:  # nodal
        iimin = 0
        iimax = 4 * cr
    return [iimin, iimax]


# Coarse grid limits (without ghost cells)
def coarsening_coarse_grid_limits(sc, sf, cr, ii_min, ii_max):
    i_range_start = (ii_min // cr) - 100
    i_range_end = (ii_max // cr) + 100

    i_min = i_range_end
    i_max = i_range_start

    for i in range(i_range_start, i_range_end + 1):
        num_ii_pts, ii_start, weights = coarsening_points_and_weights(i, sc, sf, cr)
        ii_end = ii_start + num_ii_pts - 1
        if (ii_min <= ii_end):
            i_min = min(i_min, i)
        if (ii_max >= ii_start):
            i_max = max(i_max, i)
    return [i_min, i_max]


# Coarsening for MR: interpolation points and weights
def coarsening_points_and_weights(i, sc, sf, cr):
    two_ii_start = -cr * sc + sf - 1
    if two_ii_start % 2 == 0:
        ii_start = i * cr + two_ii_start // 2
        num_ii_pts = cr + 1
        weights = np.zeros(num_ii_pts)
        weights[0] = 1.0 / (2 * cr)
        weights[num_ii_pts - 1] = 1.0 / (2 * cr)
        for ir in range(1, num_ii_pts-1):
            weights[ir] = 1.0 / cr
    else:
        ii_start = i * cr + (two_ii_start + 1) // 2
        num_ii_pts = cr
        weights = np.zeros(num_ii_pts)
        for ir in range(0, num_ii_pts):
            weights[ir] = 1.0 / cr

    return [num_ii_pts, ii_start, weights]


# Refinement functions

def refinement_coarse_grid_limits(sc, sf, cr):
    i_min = 0
    i_max = 3
    return [i_min, i_max]

def refinement_fine_grid_limits(sc, sf, cr, i_min, i_max):
    ii_range_start = i_min*cr - 100 * cr
    ii_range_end = i_max*cr + 100 * cr

    # print("ii_range_start={} and ii_range_end={}".format(ii_range_start,ii_range_end))

    ii_min = ii_range_end
    ii_max = ii_range_start

    print("Before ii_min={} and ii_max={}".format(ii_min,ii_max))

    for ii in range(ii_range_start,ii_range_end+1):
        num_i_pts, i_start, weights = refinement_points_and_weights(ii, sc, sf, cr)
        i_end = i_start + num_i_pts - 1
        if i_min <= i_end:
            ii_min = min(ii_min, ii)
        if i_max >= i_start:
            ii_max = max(ii_max, ii)

    print("After ii_min={} and ii_max={}".format(ii_min,ii_max))

    return [ii_min, ii_max]

# Refinement for MR: interpolation points and weights
def refinement_points_and_weights(ii, sc, sf, cr):
    i_start = 0
    num_i_pts = 0
    if cr == 1:
        num_i_pts = 1
        i_start = ii
    elif cr >= 2:
        if ii % cr == 0:
            num_i_pts = (1 - sf) * (1 - sc) + sf * sc
        elif ii % cr != 0:
            num_i_pts = (1 - sf) * (1 - sc) + 2 * sf * sc
        i_start = (ii // cr) * (1 - sf) * (1 - sc) + (ii // cr) * sf * sc
    weights = np.zeros(num_i_pts)
    for ir in range(num_i_pts):
        i = i_start + ir
        if ii == iimin or ii == iimax:
            weights[ir] = (1 - sf) * (1 - sc) + (
                (abs(cr - abs(ii - i * cr))) / (cr) + (cr / 2 - 0.5)
            ) * sf * sc
        else:
            weights[ir] = (1 - sf) * (1 - sc) + (
                (abs(cr - abs(ii - i * cr))) / (cr)
            ) * sf * sc
    return [num_i_pts, i_start, weights]

## TODO Coarsening for IO: interpolation points and weights
# def coarsening_points_and_weights_for_IO( i, sf, sc, cr ):
#    if   ( cr==1 ):
#        numpts = 1+abs(sf-sc)
#        idxmin = i-sc*(1-sf)
#    elif ( cr>=2 ):
#        numpts = 2-sf
#        idxmin = i*cr+cr//2*(1-sc)-(1-sf)
#    weights = np.zeros( numpts )
#    for ir in range( numpts ):
#        weights[ir] = (1/numpts)*(1-sf)*(1-sc)+(1/numpts)*sf*sc
#    return [ numpts, idxmin, weights ]

# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------

# Input coarsening ratio
cr = int(input("\n Select coarsening ratio (cr>=1): cr="))

# Loop over possible staggering of coarse and fine grid (cell-centered or nodal)
for sc in [0, 1]:
    for sf in [0, 1]:
        print("\n **************************************************")
        print(" * Staggering of coarse grid: sc={}".format(sc), end="")
        if sc == 0:
            print(" cell-centered  *")
        elif sc == 1:
            print(" nodal          *")
        print(" * Staggering of fine   grid: sf={}".format(sf), end="")
        if sf == 0:
            print(" cell-centered  *")
        elif sf == 1:
            print(" nodal          *")
        print(" **************************************************")

        iimin, iimax = coarsening_fine_grid_limits(sc, sf, cr)
        imin, imax  = coarsening_coarse_grid_limits(sc, sf, cr, iimin, iimax)

        print(
            "\n Min and max index on coarse grid:  imin={}  imax={}".format(imin, imax)
        )
        print(
            " Min and max index on fine   grid: iimin={} iimax={}".format(iimin, iimax)
        )

        print("\n Coarsening for MR: check interpolation points and weights")
        print(" ---------------------------------------------------------")

        # Coarsening for MR: interpolation points and weights
        for i in range(imin, imax+1): # index on coarse grid
            numpts, idxmin, weights = coarsening_points_and_weights(i, sc, sf, cr)
            print(
                "\n Find value at i={} by interpolating over the following points and weights:".format(
                    i
                )
            )
            wtotal = 0.0
            for ir in range(numpts):  # interpolation points and weights
                ii = idxmin + ir
                wtotal += weights[ir]
                print(" (ii={},w={:.3f})".format(ii, weights[ir]), end="")
                if not (ir == numpts - 1):
                    print(" ", end="")
            print()
            if abs(wtotal - 1.0) > 1E-9:
                print ("\n ERROR: total weight wtotal={} should be 1 for coarse index i={}".format(wtotal,i))

        # Coarsening for MR: check conservation properties
        for ii in range(iimin, iimax+1): # index on fine grid
            ws = 0.0
            for i in range(imin, imax+1): # index on coarse grid
                num_ii_pts, ii_start, weights = coarsening_points_and_weights(i, sc, sf, cr)
                for ir in range(num_ii_pts): # interpolation points and weights
                    jj = ii_start + ir
                    if jj == ii:  # interpolation point matches point on fine grid
                        ws += weights[ir]
            if abs(ws - 1.0 / cr) > 1E-9:
                print("\n ERROR: sum of weights ws={} should be 1/cr={} for ii={}".format(ws, 1.0 / cr, ii))

        print("\n Refinement for MR: check interpolation points and weights")
        print(" ---------------------------------------------------------")

        if sf != sc:
            print("\n WARNING: sc={} not equal to sf={}, not implemented for Refinement for MR, continue ...".format(sc, sf))
            continue

        imin, imax  = refinement_coarse_grid_limits(sc, sf, cr)
        iimin, iimax = refinement_fine_grid_limits(sc, sf, cr, imin, imax)

        # Number of grid points
        print( '\n Min and max index on coarse grid:  imin={}  imax={}'.format( imin, imax ) )
        print(   ' Min and max index on fine   grid: iimin={} iimax={}'.format( iimin, iimax ) )

        # Refinement for MR: interpolation points and weights
        for ii in range (iimin, iimax+1): # index on fine grid
            num_i_pts,i_start,weights = refinement_points_and_weights( ii, sc, sf, cr )
            print(
                "\n Find value at ii={} by interpolating over the following points and weights:".format(
                    ii
                )
            )
            for ir in range(num_i_pts):  # interpolation points and weights
                i = i_start + ir
                print( ' (i={},w={:.3f})'.format( i, weights[ir] ), end="")
                if not ( ir == num_i_pts-1 ):
                    print(" ", end="")
            print()

        # Refinement for MR: check conservation properties
        for i in range(imin, imax + 1): # index on coarse grid
            ws = 0.0
            for ii in range(iimin, iimax + 1): # index on fine grid
                num_i_pts, idxmin, weights = refinement_points_and_weights(ii, sc, sf, cr)
                for ir in range(num_i_pts):  # interpolation points and weights
                    j = idxmin + ir
                    if j == i: # interpolation point matches point on coarse grid
                        ws += weights[ir]
            if abs(ws - cr) > 1E-9:
                print("\n ERROR: sum of weights ws={:.3f} should be cr={} for i={}".format(ws, cr, i))

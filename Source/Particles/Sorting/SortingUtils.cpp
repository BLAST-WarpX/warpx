/* Copyright 2019-2020 Andrew Myers, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SortingUtils.H"

void fillWithConsecutiveIntegers( amrex::Gpu::DeviceVector<int>& v )
{
#ifdef AMREX_USE_GPU
    // On GPU: Use amrex
    auto data = v.data();
    auto N = v.size();
    AMREX_FOR_1D( N, i, {data[i] = i;});
#else
    // On CPU: Use std library
    std::iota( v.begin(), v.end(), 0L );
#endif
}

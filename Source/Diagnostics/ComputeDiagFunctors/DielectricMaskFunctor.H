/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIELECTRICMASKFUNCTOR_H_
#define WARPX_DIELECTRICMASKFUNCTOR_H_

#include "ComputeDiagFunctor.H"

/**
 * \brief Functor to write the dielectric material-id mask to diagnostics.
 */
class DielectricMaskFunctor final
    : public ComputeDiagFunctor
{
public:
    DielectricMaskFunctor (int lev, amrex::IntVect crse_ratio);

    void operator() (amrex::MultiFab& mf_dst, int dcomp, int i_buffer = 0) const override;

private:
    int m_lev;
};

#endif // WARPX_DIELECTRICMASKFUNCTOR_H_

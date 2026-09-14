/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWINEFIELDFUNCTOR_H_
#define WARPX_DARWINEFIELDFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to compute the full E-field (electrostatic + inductive)
 * of the semi-implicit Darwin solver into mf_out.
 */
class DarwinEfieldFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     * \param[in] mf_Efield Efield_aux component for the given direction, which at
     *                      this point in the step holds only the electrostatic
     *                      E-field (see SemiImplicitDarwin::OneStep), plus any
     *                      external field
     * \param[in] mf_dA dA_fp component for the same direction, from which the
     *                  inductive E-field is recovered as E = -dA_fp/dt
     * \param[in] lev level of the given MultiFabs
     * \param[in] crse_ratio coarsening ratio for interpolation of field values
     *                       from simulation MultiFabs to the output MultiFab mf_dst
     * \param[in] ncomp Number of component of the source MultiFabs to cell-center
     *                  in dst multifab.
     */
    DarwinEfieldFunctor (
        const amrex::MultiFab* mf_Efield,
        const amrex::MultiFab* mf_dA,
        int lev,
        amrex::IntVect crse_ratio,
        int ncomp=1
    );

    /** \brief Compute Efield_aux + ( -dA_fp/dt ) directly into mf_dst.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    /** Efield_aux component holding the electrostatic (+ external) E-field */
    const amrex::MultiFab* m_mf_Efield = nullptr;
    /** dA_fp component from which the inductive E-field is recovered */
    const amrex::MultiFab* m_mf_dA = nullptr;
    int m_lev; /**< level on which the source MultiFabs are defined */
};

#endif // WARPX_DARWINEFIELDFUNCTOR_H_

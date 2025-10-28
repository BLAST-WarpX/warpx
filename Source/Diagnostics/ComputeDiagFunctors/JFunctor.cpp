/* This file is part of WarpX.
 *
 * Authors: Roelof Groenewald
 * License: BSD-3-Clause-LBNL
 */

#include "JFunctor.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

using warpx::fields::FieldType;

JFunctor::JFunctor (const int dir, int lev,
                   amrex::IntVect crse_ratio,
                   bool convertRZmodes2cartesian,
                   bool deposit_current, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_dir(dir), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian),
      m_deposit_current(deposit_current)
{ }

void
JFunctor::operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    using ablastr::fields::Direction;

    auto& warpx = WarpX::GetInstance();
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab* m_mf_src = warpx.m_fields.get(FieldType::current_fp,Direction{m_dir},m_lev);

    // Deposit current if no solver or the electrostatic solver is being used
    if (m_deposit_current)
    {
        // allocate temporary multifab to deposit current density into
        ablastr::fields::MultiLevelVectorField current_fp_temp {
            warpx.m_fields.get_alldirs(FieldType::current_fp, m_lev)
        };

        auto& mypc = warpx.GetPartContainer();
        mypc.DepositCurrent(current_fp_temp, warpx.getdt(m_lev), 0.0);

        // sum values in guard cells - note that this does not filter the
        // current density.
        for (int idim = 0; idim < 3; ++idim) {
            current_fp_temp[0][idim]->FillBoundary(warpx.Geom(m_lev).periodicity());
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mf_src != nullptr, "m_mf_src can't be a nullptr.");
    AMREX_ASSUME(m_mf_src != nullptr);

    InterpolateMFForDiag(
        mf_dst, *m_mf_src, dcomp,
        warpx.DistributionMap(m_lev), m_convertRZmodes2cartesian);
}

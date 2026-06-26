#ifndef WARPX_COMPUTEDIAGFUNCTOR_H_
#define WARPX_COMPUTEDIAGFUNCTOR_H_

#include "ComputeDiagFunctor_fwd.H"
#include "Utils/TextMsg.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Functor to compute a diagnostic and store the result in existing
 * MultiFab
 */
class ComputeDiagFunctor
{
public:
    ComputeDiagFunctor( int ncomp, amrex::IntVect crse_ratio) :
                        m_ncomp(ncomp), m_crse_ratio(crse_ratio) {}
    //** Virtual Destructor to handle clean destruction of derived classes */
    virtual ~ComputeDiagFunctor() = default;

    // Default move and copy operations
    ComputeDiagFunctor(const ComputeDiagFunctor&) = default;
    ComputeDiagFunctor& operator=(const ComputeDiagFunctor&) = default;
    ComputeDiagFunctor(ComputeDiagFunctor&&) = default;
    ComputeDiagFunctor& operator=(ComputeDiagFunctor&&) = default;

    /** Compute a field and store the result in mf_dst
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which the result is written
     *            to the output diagnostic MultiFab, mf_dst.
     * \param[in] i_buffer index of a back-transformed snapshot
     */
    virtual void operator() (amrex::MultiFab& mf_dst, int dcomp, int i_buffer = 0) const = 0;
    /** Number of component from the input multifab to write to the output
     * multifab */
    [[nodiscard]] int nComp () const { return m_ncomp; }

    /** \brief Prepare data required to process fields in the operator()
     *         Note that this function has parameters that are specific to
     *         back-transformed diagnostics, that are unused for regular diagnostics.
     *
     * \param[in] i_buffer       index of the back-transform snapshot
     * \param[in] z_slice_in_domain if the z-slice at current_z_boost is within
     *            the boosted-frame and lab-frame domain.
     *            The fields are sliced and back-transformed only if this value is true.
     * \param[in] current_z_boost current z coordinate in the boosted-frame
     * \param[in] buffer_box     Box with index-space in lab-frame for the ith buffer
     * \param[in] k_index_zlab   k-index in the lab-frame corresponding to the
     *                           current z co-ordinate in the lab-frame for the ith buffer.
     * \param[in] snapshot_full   if the current snapshot, with index, i_buffer, is full (1)
                                 or not (0). If it is full, then Lorentz-transform
                                 is not performed by setting m_perform_backtransform to 0;
     */
    virtual void PrepareFunctorData ( int i_buffer, bool z_slice_in_domain,
                                      amrex::Real current_z_boost,
                                      amrex::Box buffer_box, const int k_index_zlab,
                                      const int snapshot_full) {
                                          amrex::ignore_unused(i_buffer,
                                          z_slice_in_domain,
                                          current_z_boost, buffer_box,
                                          k_index_zlab, snapshot_full);
                                      }
    virtual void InitData() {}

    void InterpolateMFForDiag (
        amrex::MultiFab& mf_dst, const amrex::MultiFab& mf_src, int dcomp,
        const amrex::DistributionMapping& dm, bool convertRZmodes2cartesian ) const
    {
#ifdef WARPX_DIM_RZ
        if (convertRZmodes2cartesian) {
            // In cylindrical geometry, sum real part of all modes of mf_src in
            // temporary MultiFab mf_dst_stag, and cell-center it to mf_dst
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                nComp()==1,
                "The RZ averaging over modes must write into one single component");
            amrex::MultiFab mf_dst_stag( mf_src.boxArray(), dm, 1, mf_src.nGrowVect() );
            // Mode 0
            amrex::MultiFab::Copy( mf_dst_stag, mf_src, 0, 0, 1, mf_src.nGrowVect() );
            for (int ic=1 ; ic < mf_src.nComp() ; ic += 2) {
                // Real part of all modes > 0
                amrex::MultiFab::Add( mf_dst_stag, mf_src, ic, 0, 1, mf_src.nGrowVect() );
            }
            ablastr::coarsen::sample::Coarsen( mf_dst, mf_dst_stag, dcomp, 0, nComp(), 0, m_crse_ratio );
        } else {
            ablastr::coarsen::sample::Coarsen( mf_dst, mf_src, dcomp, 0, nComp(), 0, m_crse_ratio );
        }
#else
        // In Cartesian geometry, coarsen and interpolate from temporary MultiFab mf_src
        // to output diagnostic MultiFab mf_dst
        ablastr::coarsen::sample::Coarsen(mf_dst, mf_src, dcomp, 0, nComp(), mf_dst.nGrowVect(), m_crse_ratio );
        amrex::ignore_unused(convertRZmodes2cartesian, dm);
#endif
    }

private:
    /** Number of components of mf_dst that this functor updates. */
    int m_ncomp;

protected:
    /** Coarsening ratio used to interpolate fields from simulation MultiFabs to output MultiFab. */
    amrex::IntVect m_crse_ratio;
};

#endif // WARPX_COMPUTEDIAGFUNCTOR_H_

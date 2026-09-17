#ifndef WARPX_DIVBFUNCTOR_H_
#define WARPX_DIVBFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to compute divB into mf_out.
 */
class DivBFunctor final : public ComputeDiagFunctor
{
public:
    DivBFunctor (
        int lev,
        amrex::IntVect crse_ratio,
        bool convertRZmodes2cartesian=true,
        int ncomp=1
    );

    /** \brief Compute DivB directly into mf_dst.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer*/) const override;
private:
    int m_lev; /**< level on which mf_src is defined (used in cylindrical) */
    /**< (for cylindrical) whether to average all modes into 1 comp */
    bool m_convertRZmodes2cartesian;
    /**< whether to compute divB from the fine-patch (solver) fields instead of the aux fields */
};

#endif // WARPX_DIVBFUNCTOR_H_

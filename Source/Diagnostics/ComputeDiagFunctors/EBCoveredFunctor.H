#ifndef WARPX_EBCOVEREDFUNCTOR_H_
#define WARPX_EBCOVEREDFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to indicate the fraction of the cell that is covered by the embedded boundary.
 */
class EBCoveredFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     * \param[in] lev level of multifab.
     * \param[in] crse_ratio coarsening ratio for interpolation of field values
     *                       from simulation MultiFabs to the output MultiFab mf_dst
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    EBCoveredFunctor (
        int lev,
        amrex::IntVect crse_ratio,
        int ncomp=1
    );

    /** \brief Save the fraction of the cell that is covered by the embedded boundary.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    int m_lev; /**< level on which mf_src is defined */
};

#endif // WARPX_EBCOVEREDFUNCTOR_H_

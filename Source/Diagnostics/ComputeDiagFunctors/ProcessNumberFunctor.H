#ifndef WARPX_PROCESSNUMBERFUNCTOR_H_
#define WARPX_PROCESSNUMBERFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to store the process each Box is assigned to in mf_out.
 */
class ProcessNumberFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     * \param[in] mf_src source multifab. Must be nullptr as a source is not needed
                  to compute the box-to-processes mapping.
     * \param[in] lev level of multifab. Used for averaging in rz.
     * \param[in] crse_ratio for interpolating field values from simulation MultiFabs
                  to diags MultiFab mf_dst
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    ProcessNumberFunctor(const amrex::MultiFab * mf_src, int lev,
                         amrex::IntVect crse_ratio, int ncomp=1);

    /** \brief Store the process assignment of each box in mf_dst.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    int m_lev; /**< level on which mf_src is defined */
};

#endif // WARPX_PROCESSNUMBERFUNCTOR_H_

#ifndef WARPX_DIVEFUNCTOR_H_
#define WARPX_DIVEFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BaseFwd.H>

#include <array>

/**
 * \brief Functor to compute divE into mf_out.
 */
class DivEFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     * \param[in] arr_mf_src source multifabs (3 elements for x y z).
     * \param[in] lev level of multifab.
     * \param[in] crse_ratio coarsening ratio for interpolation of field values
     *                       from simulation MultiFabs to the output MultiFab mf_dst
     * \param[in] convertRZmodes2cartesian if true, all RZ modes are averaged into one component
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    DivEFunctor (
        ablastr::fields::VectorField const & arr_mf_src,
        int lev,
        amrex::IntVect crse_ratio,
        bool convertRZmodes2cartesian=true,
        int ncomp=1
    );

    /** \brief Compute DivE directly into mf_dst.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    /** Vector of pointer to source multifab Bx, By, Bz */
    ablastr::fields::VectorField m_arr_mf_src;
    int const m_lev; /**< level on which mf_src is defined (used in cylindrical) */
    /**< (for cylindrical) whether to average all modes into 1 comp */
    bool m_convertRZmodes2cartesian;
};

#endif // WARPX_DIVEFUNCTOR_H_

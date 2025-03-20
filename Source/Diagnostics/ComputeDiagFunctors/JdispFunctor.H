#ifndef WARPX_JDISPFUNCTOR_H_
#define WARPX_JDISPFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF for displacement current density and store
 * result in mf_out.
 */
class
JdispFunctor final: public ComputeDiagFunctor
{
public:
    /**
     *
     * \param[in] dir direction of vector field to operate on
     * \param[in] lev level of multifab. Used for averaging in rz.
     * \param[in] crse_ratio for interpolating field values from the simulation Multifab, src_mf,
     *            to the output diagnostic MultiFab, mf_dst.
     * \param[in] convertRZmodes2cartesian (in cylindrical) whether to
     *            sum all modes in mf_src before cell-centering into dst multifab.
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
    */
    JdispFunctor (int dir, int lev,
                amrex::IntVect crse_ratio,
                bool convertRZmodes2cartesian=true,
                int ncomp=1);
    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
    */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    /** direction of the electron current density to save */
    int m_dir;
    /** level on which mf_src is defined */
    int m_lev;
    /** (for cylindrical) whether to average all modes into 1 comp */
    bool m_convertRZmodes2cartesian;
};

#endif // WARPX_JDISPFUNCTOR_H_

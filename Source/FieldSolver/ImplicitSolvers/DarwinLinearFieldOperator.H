/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_LINEAR_FIELD_OPERATOR_H_
#define WARPX_DARWIN_LINEAR_FIELD_OPERATOR_H_

#include "WarpXSolverVec.H"

#include "NonlinearSolvers/LinearFunction.H"

#include <AMReX_MultiFab.H>

class SemiImplicitDarwin;

/**
 * \brief Matrix-free linear operator of the semi-implicit Darwin solver.
 *
 *  This is the operator `A` of the linear system `A Z = b` that the Darwin
 *  solver hands to GMRES, i.e. the left-hand side of the magnetoinductive
 *  field equation
 *
 *      bilaplacian(Z) + curl(chi curl(Z)) = 2 * laplacian(B) + 2 * mu_0 curl(J)
 *
 *  where `chi` is the deposited mass matrix scaled by `2 mu_0 / dt`. The
 *  action of the operator on a vector is evaluated by apply(), which is
 *  called once per GMRES iteration.
 *
 *  This class is deliberately specific to the Darwin solver rather than a
 *  generic matrix-free wrapper: it owns the scratch space the operator needs,
 *  and it holds a pointer back to the solver for the fields, geometry and
 *  mass matrices that the evaluation reads.
 *
 *  No preconditioner is currently implemented, so precond() applies the
 *  identity and define() rejects any preconditioner type other than `none`.
 */
class DarwinLinearFieldOperator final : public LinearFunction<WarpXSolverVec,SemiImplicitDarwin>
{
public:

    DarwinLinearFieldOperator () = default;
    ~DarwinLinearFieldOperator () override = default;

    // Prohibit Move and Copy operations (this class owns MultiFab scratch space)
    DarwinLinearFieldOperator (const DarwinLinearFieldOperator&) = delete;
    DarwinLinearFieldOperator& operator= (const DarwinLinearFieldOperator&) = delete;
    DarwinLinearFieldOperator (DarwinLinearFieldOperator&&) = delete;
    DarwinLinearFieldOperator& operator= (DarwinLinearFieldOperator&&) = delete;

    /**
     * \brief Evaluate the action of the Darwin field operator, i.e. compute
     *  `a_Ax = bilaplacian(a_x) + curl(chi curl(a_x))`.
     *
     * \param[out] a_Ax result of applying the operator
     * \param[in] a_x vector the operator is applied to (a GMRES iterate of Z)
     */
    void apply ( WarpXSolverVec& a_Ax, const WarpXSolverVec& a_x ) override;

    /** \brief Apply the preconditioner. No preconditioner is implemented for
     *  the Darwin solver, so this applies the identity. */
    inline
    void precond ( WarpXSolverVec& a_U, const WarpXSolverVec& a_X ) override
    {
        a_U.Copy(a_X);
    }

    inline
    void updatePreCondMat ( const WarpXSolverVec& a_X ) override
    {
        amrex::ignore_unused(a_X);
    }

    inline
    void getPCMatrix ( amrex::Gpu::DeviceVector<int>& a_ridx_g,
                       amrex::Gpu::DeviceVector<int>& a_nnz,
                       amrex::Gpu::DeviceVector<int>& a_cidx_g,
                       amrex::Gpu::DeviceVector<RT>&  a_aij,
                       int& a_n, int& a_ncols_max ) override
    {
        amrex::ignore_unused(a_ridx_g, a_nnz, a_cidx_g, a_aij, a_n, a_ncols_max);
    }

    [[nodiscard]] WarpXSolverVec makeVecLHS () const override;
    [[nodiscard]] WarpXSolverVec makeVecRHS () const override;

    [[nodiscard]] inline
    bool isDefined () const { return m_is_defined; }

    /**
     * \brief Define the operator, including the scratch space used by apply().
     *
     * \param[in] a_U a defined solver vector with the layout of Z, used both
     *            to make new vectors and to size the scratch space
     * \param[in] a_ops pointer back to the Darwin solver
     * \param[in] a_pc_type preconditioner type; must be `none`
     */
    void define ( const WarpXSolverVec& a_U,
                  SemiImplicitDarwin* a_ops,
                  const PreconditionerType& a_pc_type ) override;

    [[nodiscard]] inline
    PreconditionerType pcType () const override { return PreconditionerType::none; }

private:

    bool m_is_defined = false;

    /** \brief Prototype vector used by makeVecLHS()/makeVecRHS() */
    WarpXSolverVec m_R;

    /** \brief Pointer back to the Darwin solver */
    SemiImplicitDarwin* m_ops = nullptr;

    /**
     * \brief Scratch space used by apply(), allocated once in define() rather
     *  than on every GMRES iteration since every iterate of Z shares the same
     *  layout.
     */
    amrex::MultiFab m_lapZ_x, m_lapZ_y, m_lapZ_z;
    amrex::MultiFab m_Zscratch_x, m_Zscratch_y, m_Zscratch_z;
};

#endif // WARPX_DARWIN_LINEAR_FIELD_OPERATOR_H_

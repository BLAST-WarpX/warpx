/* Copyright 2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CHEBYSHEV_PC_H_
#define CHEBYSHEV_PC_H_

#include "MatrixPrintControl.H"
#include "Preconditioner.H"
#include "ChebyshevImpl.H"

#include "MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>
#include <sstream>
#include <string>

/**
 * \brief Chebyshev polynomial preconditioner.
 *
 *  Approximately solves A x = b by running Chebyshev semi-iterative
 *  steps, delegating the iteration logic to ChebyshevImpl<T>.
 */
template <class T, class Ops>
class ChebyshevPC : public Preconditioner<T,Ops>
{
public:

    using RT = typename T::value_type;

    ChebyshevPC () = default;

    ~ChebyshevPC () override = default;

    ChebyshevPC (const ChebyshevPC&) = delete;
    ChebyshevPC& operator= (const ChebyshevPC&) = delete;
    ChebyshevPC (ChebyshevPC&&) noexcept = delete;
    ChebyshevPC& operator= (ChebyshevPC&&) noexcept = delete;

    void Define (const T&, Ops*) override;

    void Update () override;

    void Apply (T&, const T&) override;

    void printParameters () const override;

    [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

protected:

    bool m_is_defined = false;

    Ops* m_ops = nullptr;

    int m_num_amr_levels = 0;

    std::unique_ptr<MatrixSolverOps<T>> m_matrix_ops;

    ChebyshevImpl<T> m_solver;

    MatrixPrintControl m_print;
    bool m_print_pending = false;

private:

};

template <class T, class Ops>
void ChebyshevPC<T,Ops>::printParameters () const
{
    auto pc_name = amrex::getEnumNameString(PreconditionerType::pc_chebyshev);
    m_solver.printParameters(pc_name);
    m_print.printParameters();
    if (m_matrix_ops) {
        amrex::Print() << pc_name << " matrix ops:           " << m_matrix_ops->Name() << "\n";
    }
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Define (const T& a_U,
                                  Ops* const a_ops)
{
    BL_PROFILE("ChebyshevPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "ChebyshevPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "ChebyshevPC::Define(): a_ops is nullptr");
    amrex::ignore_unused(a_U);

    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();

    m_matrix_ops = Ops::MakePCMatrixOps(m_ops);

    m_solver.Define(m_matrix_ops.get(),
                    getEnumNameString(PreconditionerType::pc_chebyshev));
    m_print.Define(getEnumNameString(PreconditionerType::pc_chebyshev));

    m_is_defined = true;
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Update ()
{
    BL_PROFILE("ChebyshevPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ChebyshevPC::Update() called on undefined object");

    m_solver.Update();

    // Written after the next Apply(), which estimates the eigenvalue bounds
    m_print_pending = m_print.ShouldPrint(m_ops->GetCurrentStep());
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("ChebyshevPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ChebyshevPC::Apply() called on undefined object");

    m_solver.Solve(a_x, a_b);

    if (m_print_pending) {
        auto header = m_print.HeaderLines();
        std::ostringstream ss;
        ss.precision(10);
        ss << "chebyshev_lambda_DinvA: " << m_solver.getLambdaMin() << " "
           << m_solver.getLambdaMax();
        header.emplace_back(ss.str());
        m_matrix_ops->PrintMatrix(m_print.FileName(), header);
        m_print_pending = false;
    }
}

#endif

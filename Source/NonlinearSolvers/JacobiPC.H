/* Copyright 2024-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JACOBI_PC_H_
#define JACOBI_PC_H_

#include "Preconditioner.H"
#include "WeightedJacobiImpl.H"

#include "MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Approximately solves A x = b via damped Jacobi iterations,
 *  delegating the iteration logic to WeightedJacobiImpl<T>.
 */

template <class T, class Ops>
class JacobiPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        JacobiPC () = default;

        ~JacobiPC () override = default;

        JacobiPC(const JacobiPC&) = delete;
        JacobiPC& operator=(const JacobiPC&) = delete;
        JacobiPC(JacobiPC&&) noexcept = delete;
        JacobiPC& operator=(JacobiPC&&) noexcept = delete;

        void Define (const T&, Ops*) override;

        void Update () override;

        /**
         * \brief Solve A x = b via damped Jacobi iteration.
         */
        void Apply (T&, const T&) override;

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        std::unique_ptr<MatrixSolverOps<T>> m_matrix_ops;

        WeightedJacobiImpl<T> m_solver;

    private:

};

template <class T, class Ops>
void JacobiPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_jacobi);
    m_solver.printParameters(pc_name);
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Define ( const T& a_U,
                               Ops* const a_ops )
{
    BL_PROFILE("JacobiPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "JacobiPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "JacobiPC::Define(): a_ops is nullptr" );
    amrex::ignore_unused(a_U);

    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();

    m_matrix_ops = Ops::MakePCMatrixOps(m_ops);

    m_solver.Define(m_matrix_ops.get(),
                    getEnumNameString(PreconditionerType::pc_jacobi));

    m_is_defined = true;
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Update ()
{
    BL_PROFILE("JacobiPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Update() called on undefined object" );

    m_solver.Update();

    if (m_solver.isVerbose()) {
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_jacobi)
                       << ": dt = " << this->m_dt
                       << ", omega = " << m_solver.getOmega() << "\n";
    }
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("JacobiPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Apply() called on undefined object" );

    m_solver.Solve(a_x, a_b);
}

#endif

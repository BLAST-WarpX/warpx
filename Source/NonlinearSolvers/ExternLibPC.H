/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef EXTERNLIB_PC_H_
#define EXTERNLIB_PC_H_

#include "Fields.H"
#include "Preconditioner.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Print.H>

#include <string>

/**
 * \brief External Library Preconditioner
 *
 *  This preconditioner delegates the sparse matrix assembly to the operator
 *  (ImplicitSolver) and provides the assembled matrix to external solver
 *  libraries (e.g., PETSc) via getPCMatrix(). The native Apply() is not
 *  implemented — it requires an external library.
 *
 *  This class is templated on a solution-type class T and an operator class Ops.
 */

template <class T, class Ops>
class ExternLibPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        ExternLibPC () = default;

        ~ExternLibPC () override = default;

        ExternLibPC(const ExternLibPC&) = delete;
        ExternLibPC& operator=(const ExternLibPC&) = delete;
        ExternLibPC(ExternLibPC&&) noexcept = delete;
        ExternLibPC& operator=(ExternLibPC&&) noexcept = delete;

        void Define (const T&, Ops* const) override;

        void Update (const T& a_U) override;

        /**
         * \brief Apply (solve) the preconditioner given a RHS.
         *        Not implemented natively — requires an external library (e.g., PETSc).
         */
        void Apply (T&, const T&) override;

        inline void getPCMatrix (amrex::Gpu::DeviceVector<int>& a_r_indices_g,
                                 amrex::Gpu::DeviceVector<int>& a_num_nz,
                                 amrex::Gpu::DeviceVector<int>& a_c_indices_g,
                                 amrex::Gpu::DeviceVector<RT>& a_a_ij,
                                 int& a_n, int& a_ncols_max) override
        {
            m_ops->GetPCMatrix(a_r_indices_g, a_num_nz, a_c_indices_g,
                               a_a_ij, a_n, a_ncols_max);
        }

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        inline void setName (const std::string& a_name) override { m_name = a_name; }

    protected:

        bool m_is_defined = false;
        bool m_verbose = true;

        Ops* m_ops = nullptr;

        std::string m_name = "noname";

    private:

};

template <class T, class Ops>
void ExternLibPC<T,Ops>::printParameters () const
{
    amrex::Print() << m_name << " (ExternLibPC): delegates to SparseJacobianMatrix\n";
}

template <class T, class Ops>
void ExternLibPC<T,Ops>::Define ( const T& a_U,
                                  Ops* const a_ops )
{
    BL_PROFILE("ExternLibPC::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "ExternLibPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "ExternLibPC::Define(): a_ops is nullptr");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ExternLibPC::Define() must be called with Efield_fp type");

    amrex::ignore_unused(a_U);
    m_ops = a_ops;
    m_is_defined = true;
}

template <class T, class Ops>
void ExternLibPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("ExternLibPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ExternLibPC::Update() called on undefined object");

    m_ops->AssemblePCMatrix(a_U);

    if (m_verbose) {
        amrex::Print() << "Updated " << m_name
                       << " (ExternLibPC): theta*dt = " << this->m_dt << "\n";
    }
}

template <class T, class Ops>
void ExternLibPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("ExternLibPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ExternLibPC::Apply() called on undefined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ExternLibPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ExternLibPC::Apply() - a_b must be Efield_fp type");

    WARPX_ABORT_WITH_MESSAGE(
        "ExternLibPC::Apply() - native matrix solvers not implemented."
        " Use with external library, e.g., PETSc.");
}

#endif

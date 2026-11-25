/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ALGMG_PC_H_
#define ALGMG_PC_H_

#include "Fields.H"
#include "MatrixPCAlgMG.H"
#include "Preconditioner.H"
#include "PreconditionerTypes.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <memory>
#include <string>

/**
 * \brief AMReX algebraic multigrid (AlgMG) preconditioner
 *
 *  The operator (ImplicitSolver) assembles the sparse matrix, as for pc_petsc;
 *  AlgMG V-cycles on this matrix apply the preconditioner.
 *
 *  This class is templated on a solution-type class T and an operator class Ops.
 *  The T class must return the first global DOF of each process.
 */
template <class T, class Ops>
class AlgMGPC : public Preconditioner<T,Ops>
{
public:

    using RT = typename T::value_type;

    AlgMGPC () = default;

    ~AlgMGPC () override = default;

    AlgMGPC (const AlgMGPC&) = delete;
    AlgMGPC& operator= (const AlgMGPC&) = delete;
    AlgMGPC (AlgMGPC&&) noexcept = delete;
    AlgMGPC& operator= (AlgMGPC&&) noexcept = delete;

    void Define (const T&, Ops*) override;

    void Update () override;

    void Apply (T&, const T&) override;

    void printParameters () const override;

    [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

protected:

    bool m_is_defined = false;
    bool m_verbose = true;

    Ops* m_ops = nullptr;

    std::string m_name = amrex::getEnumNameString(PreconditionerType::pc_algmg);

    std::unique_ptr<MatrixPCAlgMG> m_algmg;
    bool m_algmg_is_set_up = false;
    RT m_algmg_theta_dt = RT(0); // theta*dt of the last AlgMG setup

private:

};

template <class T, class Ops>
void AlgMGPC<T,Ops>::printParameters () const
{
    amrex::Print() << m_name << " verbose:                " << (m_verbose?"true":"false") << "\n";
    if (m_algmg) { m_algmg->printParameters(); }
}

template <class T, class Ops>
void AlgMGPC<T,Ops>::Define (const T& a_U,
                             Ops* const a_ops)
{
    BL_PROFILE("AlgMGPC::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "AlgMGPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "AlgMGPC::Define(): a_ops is nullptr");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "AlgMGPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;

    const amrex::ParmParse pp(m_name);
    pp.query("verbose", m_verbose);

    m_algmg = std::make_unique<MatrixPCAlgMG>(m_name, a_U.dofRowOffsets());

    m_is_defined = true;
}

template <class T, class Ops>
void AlgMGPC<T,Ops>::Update ()
{
    BL_PROFILE("AlgMGPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "AlgMGPC::Update() called on undefined object");

    m_ops->AssemblePCMatrix();

    // Without mass matrices, the matrix only depends on theta*dt.
    // Load balancing is not supported by the implicit solvers. If it is
    // supported in the future, the AlgMG partition needs to be updated when
    // the DistributionMapping changes.
    const RT theta_dt = m_ops->GetThetaForPC()*this->m_dt;
    const bool include_mass_matrices = (m_ops->GetMassMatricesCoeff() != nullptr);
    if (include_mass_matrices || !m_algmg_is_set_up || theta_dt != m_algmg_theta_dt) {
        amrex::Gpu::DeviceVector<int> r_indices_g, num_nz, c_indices_g;
        amrex::Gpu::DeviceVector<RT> a_ij;
        int n = 0;
        int ncols_max = 0;
        m_ops->GetPCMatrix(r_indices_g, num_nz, c_indices_g, a_ij, n, ncols_max);
        m_algmg->Setup(n, ncols_max, num_nz.data(), c_indices_g.data(), a_ij.data());
        m_algmg_is_set_up = true;
        m_algmg_theta_dt = theta_dt;
        if (m_verbose) {
            amrex::Print() << "Updated " << m_name << ": theta*dt = " << theta_dt << "\n";
        }
    }
}

template <class T, class Ops>
void AlgMGPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("AlgMGPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "AlgMGPC::Apply() called on undefined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "AlgMGPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "AlgMGPC::Apply() - a_b must be Efield_fp type");

    a_b.copyTo(m_algmg->rhs());
    m_algmg->Apply();
    a_x.copyFrom(m_algmg->sol());
}

#endif

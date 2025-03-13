/* Copyright 2024 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JACOBI_PC_H_
#define JACOBI_PC_H_

#include "Fields.H"
#include "Utils/WarpXConst.H"
#include "Preconditioner.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Solve a given system using the Point-Jaocbi method
 *
 *  The equation solves for Eg in:
 *  A * Eg = b
 *  where
 *    + A is an operator
 *    + Eg is the electric field.
 *    + b is a specified RHS with the same layout as Eg
 *
 *  This class is templated on a solution-type class T and an operator class Ops.
 *
 *  The Ops class must have the following function:
 *      + Return number of AMR levels
 *      + Return the amrex::Geometry object given an AMR level
 *      + Return hi and lo linear operator boundaries
 *      + Return the time step factor (theta) for the time integration scheme
 *
 *  The T class must have the following functions:
 *      + Return underlying vector of amrex::MultiFab arrays
 */

template <class T, class Ops>
class JacobiPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        JacobiPC () = default;

        /**
         * \brief Default destructor
         */
        ~JacobiPC () override = default;

        // Prohibit move and copy operations
        JacobiPC(const JacobiPC&) = delete;
        JacobiPC& operator=(const JacobiPC&) = delete;
        JacobiPC(JacobiPC&&) noexcept = delete;
        JacobiPC& operator=(JacobiPC&&) noexcept = delete;

        /**
         * \brief Define the preconditioner
         */
        void Define (const T&, Ops* const) override;

        /**
         * \brief Update the preconditioner
         */
        void Update (const T& a_U) override;

        /**
         * \brief Apply (solve) the preconditioner given a RHS
         *
         *  Given a right-hand-side b, solve:
         *      A x = b
         *  where A is the linear operator.
         */
        void Apply (T&, const T&) override;

        /**
         * \brief Print parameters
         */
        void printParameters() const override;

        /**
         * \brief Check if the nonlinear solver has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        using MFArr = amrex::Array<amrex::MultiFab,3>;

        bool m_is_defined = false;

        bool m_verbose = true;
        int m_max_iter = 10;

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;
        amrex::Vector<amrex::Geometry> m_geom;
        amrex::Vector<amrex::BoxArray> m_grids;
        amrex::Vector<amrex::DistributionMapping> m_dmap;
        amrex::IntVect m_gv;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        /**
         * \brief Read parameters
         */
        void readParameters();

    private:

};

template <class T, class Ops>
void JacobiPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_jacobi);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " max iter:             " << m_max_iter << "\n";
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T, class Ops>
void JacobiPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_jacobi));
    pp.query("verbose", m_verbose);
    pp.query("max_iter", m_max_iter);
    pp.query("absolute_tolerance",  m_atol);
    pp.query("relative_tolerance",  m_rtol);
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Define ( const T& a_U,
                               Ops* const a_ops )
{
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "JacobiPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "JacobiPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Define() must be called with Efield_fp type");

    amrex::ignore_unused(a_U);
    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    m_bcoefs = m_ops->GetSigmaCoeff();

    // read preconditioner parameters
    readParameters();

    m_is_defined = true;
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Update (const T& a_U)
{
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Update() called on undefined object" );

    // a_U is not needed for a linear operator
    amrex::ignore_unused(a_U);

    if (m_verbose) {
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_jacobi)
                       << ": theta*dt = " << this->m_dt << "\n";
    }
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    //  Given a right-hand-side b, solve:
    //      A x = b
    //  where A is the linear operator.

    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Apply() - a_b must be Efield_fp type");

    if (m_bcoefs == nullptr) {

        a_x.Copy(a_b);

    } else {

        // Get the data vectors
        auto& b_mfarrvec = a_b.getArrayVec();
        auto& x_mfarrvec = a_x.getArrayVec();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ((b_mfarrvec.size() == m_num_amr_levels) && (x_mfarrvec.size() == m_num_amr_levels)),
            "Error in JacobiPC::Apply() - mismatch in number of levels." );

        // setting initial guess x = b/diag(A)
        for (int n = 0; n < m_num_amr_levels; n++) {
            for (int dim = 0; dim < 3; dim++) {
                const auto& b_mf = *(b_mfarrvec[n][dim]);
                const auto& a_mf = *((*m_bcoefs)[n][dim]);
                auto& x_mf = *(x_mfarrvec[n][dim]);

                for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box bx = mfi.tilebox();
                    auto x_arr = x_mf.array(mfi);
                    auto b_arr = b_mf.const_array(mfi);
                    auto a_arr = a_mf.const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        x_arr(i,j,k) = b_arr(i,j,k) / a_arr(i,j,k);
                    });
                }
            }
        }

        // Jacobi iterations
        // not yet implemented; will do after mass matrix has nondiagonal elements

    }
}

#endif

/* Copyright 2024 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef PETSC_PC_H_
#define PETSC_PC_H_

#include "Fields.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"
#include "Preconditioner.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

/**
 * \brief PETSc Preconditioner Wrapper
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
class PetscPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        PetscPC () = default;

        /**
         * \brief Default destructor
         */
        ~PetscPC () override = default;

        // Prohibit move and copy operations
        PetscPC(const PetscPC&) = delete;
        PetscPC& operator=(const PetscPC&) = delete;
        PetscPC(PetscPC&&) noexcept = delete;
        PetscPC& operator=(PetscPC&&) noexcept = delete;

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
         *  where A is the linear operator, in this case, the curl-curl operator:
         *      A x = curl (alpha * curl (x) ) + beta * x
         */
        void Apply (T&, const T&) override;

        inline void getPCMatrix (amrex::Gpu::DeviceVector<int>& a_r_indices_g,
                                 amrex::Gpu::DeviceVector<int>& a_num_nz,
                                 amrex::Gpu::DeviceVector<int>& a_c_indices_g,
                                 amrex::Gpu::DeviceVector<RT>& a_a_ij,
                                 int& a_n, int& a_ncols_max )
        {
            a_n = m_ndofs_l;
            a_ncols_max = m_pc_mat_nnz;

            a_r_indices_g.resize(m_r_indices_g.size());
            amrex::Gpu::copy( amrex::Gpu::deviceToDevice,
                              m_r_indices_g.begin(),
                              m_r_indices_g.end(),
                              a_r_indices_g.begin() );

            a_num_nz.resize(m_num_nz.size());
            amrex::Gpu::copy( amrex::Gpu::deviceToDevice,
                              m_num_nz.begin(),
                              m_num_nz.end(),
                              a_num_nz.begin() );

            a_c_indices_g.resize(m_c_indices_g.size());
            amrex::Gpu::copy( amrex::Gpu::deviceToDevice,
                              m_c_indices_g.begin(),
                              m_c_indices_g.end(),
                              a_c_indices_g.begin() );

            a_a_ij.resize(m_a_ij.size());
            amrex::Gpu::copy( amrex::Gpu::deviceToDevice,
                              m_a_ij.begin(),
                              m_a_ij.end(),
                              a_a_ij.begin() );

        }

        /**
         * \brief Print parameters
         */
        void printParameters() const override;

        /**
         * \brief Check if the nonlinear solver has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;
        bool m_verbose = true;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        int m_ndofs_l = 0;
        int m_ndofs_g = 0;
        bool m_pc_diag_only = false;
        int m_pc_mat_nnz = 0;
        int m_pc_mass_matrix_width = 0;
        bool m_pc_mass_matrix_include_ij = false;

        amrex::Gpu::DeviceVector<int> m_r_indices_g;
        amrex::Gpu::DeviceVector<int> m_num_nz;
        amrex::Gpu::DeviceVector<int> m_c_indices_g;
        amrex::Gpu::DeviceVector<amrex::Real> m_a_ij;

        /**
         * \brief Read parameters
         */
        void readParameters();

        /**
         * \brief compute max number of non-zero elements per row in matrix
         */
        void computePCMatrixNNZ();

    private:

};

template <class T, class Ops>
void PetscPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_petsc);
    Print() << pc_name << " verbose:                   " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " pc_diagonal_only:          " << (m_pc_diag_only?"true":"false") << "\n";
    Print() << pc_name << " pc_mass_matrix_width:      " << m_pc_mass_matrix_width << "\n";
    Print() << pc_name << " pc_mass_matrix_include_ij: " << (m_pc_mass_matrix_include_ij?"true":"false") << "\n";
}

template <class T, class Ops>
void PetscPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_petsc));
    pp.query("verbose", m_verbose);
    pp.query("pc_diagonal_only", m_pc_diag_only);
    pp.query("pc_mass_matrix_width", m_pc_mass_matrix_width);
    if(m_pc_mass_matrix_width>0) {
       pp.query("pc_mass_matrix_include_ij", m_pc_mass_matrix_include_ij);
    }
}

template <class T, class Ops>
void PetscPC<T,Ops>::computePCMatrixNNZ()
{
    if (m_pc_diag_only) {
        m_pc_mat_nnz = 1;
    } else {
#if defined(WARPX_DIM_1D_Z)
        static int nbands_J_mass_matrix = 1 + 2*3*(m_pc_mass_matrix_width+1);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        static int nbands_J_mass_matrix =    3
                                           * (1+2*m_pc_mass_matrix_width)
                                           * (1+2*m_pc_mass_matrix_width);
#else
    WARPX_ABORT_WITH_MESSAGE("PetscPC<T,Ops>::computePCMatrixNNZ() not implemented for this dimensionality");
#endif
        // the following overallocates
        static int nbands_EMfields = 1 + 2*amrex::SpaceDim
                                       + 2*(amrex::SpaceDim-1)*(amrex::SpaceDim-1);
        m_pc_mat_nnz = nbands_EMfields + nbands_J_mass_matrix;
    }

}

template <class T, class Ops>
void PetscPC<T,Ops>::Define ( const T& a_U,
                              Ops* const a_ops )
{
    BL_PROFILE("PetscPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "PetscPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "PetscPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "PetscPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    // read preconditioner parameters
    readParameters();

    // a_U is not needed
    amrex::ignore_unused(a_U);

    // Set number of AMR levels and create geometry, grids, and
    // distribution mapping vectors.
    m_num_amr_levels = m_ops->numAMRLevels();
    if (m_num_amr_levels > 1) {
        WARPX_ABORT_WITH_MESSAGE("PetscPC::Define(): m_num_amr_levels > 1");
    }

    computePCMatrixNNZ();

    m_ndofs_l = a_U.nDOF_local();
    m_ndofs_g = a_U.nDOF_global();

    m_r_indices_g.resize(m_ndofs_l);
    m_num_nz.resize(m_ndofs_l);
    m_c_indices_g.resize(m_pc_mat_nnz*m_ndofs_l);
    m_a_ij.resize(m_pc_mat_nnz*m_ndofs_l);

    m_is_defined = true;
}

template <class T, class Ops>
void PetscPC<T,Ops>::Update (const T& a_U)
{
    // Assemble the sparse matrix representation of the preconditioner
    //      A = curl (alpha * curl []) + M
    // where M is the mass matrix. The following data is set in this function:
    // - m_r_indices_g: integer array of size n with the global row indices
    // - m_num_nz:      integer array of size n with the number of non-zero elements
    //                  in each row
    // - m_c_indices:   integer array of size n*ncmax with the global column indices
    //                  of non-zero elements in each row (row-major)
    // - m_a_ij:        real-type array of size n*ncmax with the matrix element values
    //                  (row-major format)
    // where n is the local number of rows, and ncmax is the maximum number of non-zero
    // elements per row.

    BL_PROFILE("PetscPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "PetscPC::Update() called on undefined object" );

    // set the alpha coefficient for the curl-curl op
    const RT thetaDt = m_ops->GetThetaForPC()*this->m_dt;
    const RT alpha = (thetaDt*PhysConst::c) * (thetaDt*PhysConst::c);
    if (m_verbose) {
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_petsc)
                       << ": theta*dt = " << thetaDt << ", "
                       << " coefficients: "
                       << "alpha = " << alpha << "\n";
    }

    // Get DOF object from a_U
    const auto& dofs_obj = a_U.getDOFsObject();
    const auto& dofs_mfarrvec = dofs_obj->m_array;
    AMREX_ALWAYS_ASSERT(m_ndofs_l == dofs_obj->m_nDoFs_l);
    AMREX_ALWAYS_ASSERT(m_ndofs_g == dofs_obj->m_nDoFs_g);

    auto r_indices_g_ptr = m_r_indices_g.data();
    auto num_nz_ptr = m_num_nz.data();
    auto c_indices_g_ptr = m_c_indices_g.data();
    auto a_ij_ptr = m_a_ij.data();

    const auto ndofs_l = m_ndofs_l;
    const auto ndofs_g = m_ndofs_g;
    const auto nnz_max = m_pc_mat_nnz;

    for (int lev = 0; lev < m_num_amr_levels; lev++) {
        for (int n = 0; n < 3; n++) {
            auto ncomp = dofs_mfarrvec[lev][n]->nComp();
            AMREX_ALWAYS_ASSERT(ncomp == 2); // local, global
            for (amrex::MFIter mfi(*dofs_mfarrvec[lev][n]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto dof_arr = dofs_mfarrvec[lev][n]->const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto row_idx_l = (int) dof_arr(i,j,k,0);
                    auto row_idx_g = (int) dof_arr(i,j,k,1);
                    AMREX_ALWAYS_ASSERT((row_idx_l >= 0) && (row_idx_l < ndofs_l));
                    AMREX_ALWAYS_ASSERT((row_idx_g >= 0) && (row_idx_g < ndofs_g));

                    r_indices_g_ptr[row_idx_l] = row_idx_g;
                    num_nz_ptr[row_idx_l] = 1;
                    c_indices_g_ptr[row_idx_l*nnz_max+0] = row_idx_g;
                    a_ij_ptr[row_idx_l*nnz_max+0] = 1.0;
                });
            }
        }
    }

}

template <class T, class Ops>
void PetscPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    //  Given a right-hand-side b, solve:
    //      A x = b
    //  where A is the linear operator, in this case, the curl-curl
    //  operator:
    //      A x = curl (alpha * curl (x) ) + beta * x

    BL_PROFILE("PetscPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "PetscPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "PetscPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "PetscPC::Apply() - a_b must be Efield_fp type");

    WARPX_ABORT_WITH_MESSAGE("PetscPC<T,Ops>::Apply() function should not have been called. PETSc should be using it's own PC-apply function.");

}

#endif

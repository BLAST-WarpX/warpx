/* Copyright 2024 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MATRIX_PC_H_
#define MATRIX_PC_H_

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

namespace MatrixPCUtils
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool insertOrAdd( const int a_cidx, /*!< Column index (global) */
                      const amrex::Real a_val, /*!< Value to insert/add */
                      int* const a_cidxs, /*!< Column index array */
                      amrex::Real* const a_aij, /*!< array of values */
                      const int a_nnz, /*!< max number of non-zero columns */
                      int& a_ncol /*!< number of non-zero columns */)
    {
        if (a_cidx < 0) { return true; /* outside domain */ }
        int loc = -1;
        for (int icol = 0; icol < std::min(a_ncol,a_nnz); icol++) {
            if (a_cidxs[icol] == a_cidx) {
                loc = icol;
                break;
            }
        }
        if (loc < 0) {
            a_ncol++;
            if (a_ncol > a_nnz) { return false; }
            else {
                // column index not found; add new entry
                a_cidxs[a_ncol-1] = a_cidx;
                a_aij[a_ncol-1] = a_val;
            }
        } else {
            // column index already exists; add to value
            a_aij[loc] += a_val;
        }
        return true;
    }
}

/**
 * \brief Matrix Preconditione
 *
 *  This class is templated on a solution-type class T and an operator class Ops. It implements
 *  a preconditioner based on constructing and solving the sparse matrix representation of the
 *  Jacobian. Currently, it implements constructing/assembling the sparse matrix. An external
 *  library with sparse matrix solvers (for example, PETSc) is needed to apply the preconditioner.
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
class MatrixPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        MatrixPC () = default;

        /**
         * \brief Default destructor
         */
        ~MatrixPC () override = default;

        // Prohibit move and copy operations
        MatrixPC(const MatrixPC&) = delete;
        MatrixPC& operator=(const MatrixPC&) = delete;
        MatrixPC(MatrixPC&&) noexcept = delete;
        MatrixPC& operator=(MatrixPC&&) noexcept = delete;

        /**
         * \brief Define the preconditioner
         */
        void Define (const T&, Ops* const) override;

        /**
         * \brief Update the preconditioner
         */
        void Update (const T& a_U) override;

        /**
         * \brief Assemble the matrix
         *
         * Returns 0 if matrix assembly is successful; returns
         * positive integer if allocated arrays are insufficient to hold all
         * non-zero elements (return value is difference in current number
         * of nonzero elements and the desired number).
         */
        int Assemble (const T& a_U);

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
                                 int& a_n, int& a_ncols_max ) override
        {
            a_n = m_ndofs_l;
            a_ncols_max = m_pc_mat_nnz;

            a_r_indices_g.resize(m_r_indices_g.size());
            amrex::Gpu::copyAsync( amrex::Gpu::deviceToDevice,
                                   m_r_indices_g.begin(),
                                   m_r_indices_g.end(),
                                   a_r_indices_g.begin() );

            a_num_nz.resize(m_num_nz.size());
            amrex::Gpu::copyAsync( amrex::Gpu::deviceToDevice,
                                   m_num_nz.begin(),
                                   m_num_nz.end(),
                                   a_num_nz.begin() );

            a_c_indices_g.resize(m_c_indices_g.size());
            amrex::Gpu::copyAsync( amrex::Gpu::deviceToDevice,
                                   m_c_indices_g.begin(),
                                   m_c_indices_g.end(),
                                   a_c_indices_g.begin() );

            a_a_ij.resize(m_a_ij.size());
            amrex::Gpu::copyAsync( amrex::Gpu::deviceToDevice,
                                   m_a_ij.begin(),
                                   m_a_ij.end(),
                                   a_a_ij.begin() );
            amrex::Gpu::streamSynchronize();
        }

        /**
         * \brief Print parameters
         */
        void printParameters() const override;

        /**
         * \brief Check if the nonlinear solver has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        inline void setName (const std::string& a_name) override { m_name = a_name; }

    protected:

        bool m_is_defined = false;
        bool m_verbose = true;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;
        amrex::Vector<amrex::Geometry> m_geom;

        int m_ndofs_l = 0;
        int m_ndofs_g = 0;
        bool m_pc_diag_only = false;
        int m_pc_mat_nnz = 1;
        int m_pc_mass_matrix_width = 0;
        bool m_pc_mass_matrix_include_ij = false;

        std::string m_name = "noname";

        amrex::Gpu::DeviceVector<int> m_r_indices_g;
        amrex::Gpu::DeviceVector<int> m_num_nz;
        amrex::Gpu::DeviceVector<int> m_c_indices_g;
        amrex::Gpu::DeviceVector<amrex::Real> m_a_ij;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        int m_num_realloc = 0;

        /**
         * \brief Read parameters
         */
        void readParameters();

    private:

};

template <class T, class Ops>
void MatrixPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    Print() << m_name << " verbose:                   " << (m_verbose?"true":"false") << "\n";
    Print() << m_name << " pc_diagonal_only:          " << (m_pc_diag_only?"true":"false") << "\n";
    Print() << m_name << " pc_mass_matrix_width:      " << m_pc_mass_matrix_width << "\n";
    Print() << m_name << " pc_mass_matrix_include_ij: " << (m_pc_mass_matrix_include_ij?"true":"false") << "\n";
}

template <class T, class Ops>
void MatrixPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(m_name);
    pp.query("verbose", m_verbose);
    pp.query("pc_diagonal_only", m_pc_diag_only);
    pp.query("pc_mass_matrix_width", m_pc_mass_matrix_width);
    if(m_pc_mass_matrix_width>0) {
       pp.query("pc_mass_matrix_include_ij", m_pc_mass_matrix_include_ij);
    }
}

template <class T, class Ops>
void MatrixPC<T,Ops>::Define ( const T& a_U,
                              Ops* const a_ops )
{
    BL_PROFILE("MatrixPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "MatrixPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "MatrixPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "MatrixPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    // read preconditioner parameters
    readParameters();

    // a_U is not needed
    amrex::ignore_unused(a_U);

    // Set number of AMR levels and create geometry, grids, and
    // distribution mapping vectors.
    m_num_amr_levels = m_ops->numAMRLevels();
    if (m_num_amr_levels > 1) {
        WARPX_ABORT_WITH_MESSAGE("MatrixPC::Define(): m_num_amr_levels > 1");
    }
    m_geom.resize(m_num_amr_levels);
    for (int n = 0; n < m_num_amr_levels; n++) {
        m_geom[n] = m_ops->GetGeometry(n);
    }

    m_ndofs_l = a_U.nDOF_local();
    m_ndofs_g = a_U.nDOF_global();

    auto n_rows = size_t(m_ndofs_l);
    auto n_cols = size_t(m_pc_mat_nnz) * size_t(m_ndofs_l);

    m_r_indices_g.resize(n_rows);
    m_num_nz.resize(n_rows);
    m_c_indices_g.resize(n_cols);
    m_a_ij.resize(n_cols);


    m_bcoefs = m_ops->GetMassMatricesCoeff();
    if (m_bcoefs == nullptr) {
        if (m_pc_mass_matrix_width >= 0) {
            std::stringstream warning_message;
            warning_message << "pc_mass_matrix_width is nonnegative but unable to get mass matrices from operator.\n";
            warning_message << "setting m_pc_mass_matrix_width = -1 (no mass matrix in PC).\n";
            ablastr::warn_manager::WMRecordWarning("MatrixPC", warning_message.str());
        }
        m_pc_mass_matrix_width = -1;
    }

    m_is_defined = true;
}

template <class T, class Ops>
void MatrixPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("MatrixPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "MatrixPC::Update() called on undefined object" );

    while(1) {

        auto nnz_diff = Assemble(a_U);
        AMREX_ALWAYS_ASSERT(nnz_diff >= 0);
        if (nnz_diff) {

            m_pc_mat_nnz += nnz_diff;
            m_num_realloc++;

        } else {
            break;
        }
    }

    if (m_num_realloc > 1) {
        std::stringstream warning_message;
        warning_message << "Number of times arrays were reallocated due to new nonzero elements "
                        << "is greater than 1 (" << m_num_realloc <<"). This is unexpected.\n";
        ablastr::warn_manager::WMRecordWarning("MatrixPC", warning_message.str());
    }

}

template <class T, class Ops>
int MatrixPC<T,Ops>::Assemble (const T& a_U)
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

    BL_PROFILE("MatrixPC::Assemble()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "MatrixPC::Assemble() called on undefined object" );

    // set the alpha coefficient for the curl-curl op
    const RT thetaDt = m_ops->GetThetaForPC()*this->m_dt;
    const RT alpha = (thetaDt*PhysConst::c) * (thetaDt*PhysConst::c);
    if (m_verbose) {
        amrex::Print() << "Updating " << m_name
                       << ": theta*dt = " << thetaDt << ", "
                       << " coefficients: "
                       << "alpha = " << alpha << "\n";
    }

    // Get DOF object from a_U
    const auto& dofs_obj = a_U.getDOFsObject();
    const auto& dofs_mfarrvec = dofs_obj->m_array;
    const auto& dofs_lhs_mfarrvec = dofs_obj->m_array_lhs;
    AMREX_ALWAYS_ASSERT(m_ndofs_l == dofs_obj->m_nDoFs_l);
    AMREX_ALWAYS_ASSERT(m_ndofs_g == dofs_obj->m_nDoFs_g);

    m_r_indices_g.clear();
    m_num_nz.clear();
    m_c_indices_g.clear();
    m_a_ij.clear();

    auto n_rows = size_t(m_ndofs_l);
    auto n_cols = size_t(m_pc_mat_nnz) * size_t(m_ndofs_l);

    m_r_indices_g.resize(n_rows);
    m_num_nz.resize(n_rows);
    m_c_indices_g.resize(n_cols);
    m_a_ij.resize(n_cols);

    auto r_indices_g_ptr = m_r_indices_g.data();
    auto num_nz_ptr = m_num_nz.data();
    auto c_indices_g_ptr = m_c_indices_g.data();
    auto a_ij_ptr = m_a_ij.data();

    const auto ndofs_l = m_ndofs_l;
    const auto ndofs_g = m_ndofs_g;
    const auto nnz_max = m_pc_mat_nnz;
    auto nnz_actual = nnz_max;

    for (int lev = 0; lev < m_num_amr_levels; lev++) {

        auto ncomp = dofs_mfarrvec[lev][0]->nComp();
        AMREX_ALWAYS_ASSERT(ncomp == 2); // local, global

        const auto& geom = m_geom[lev];
        const auto dxi = geom.InvCellSizeArray();

        Gpu::Buffer<int> nnz_actual_d({nnz_max});
        auto* nnz_actual_ptr = nnz_actual_d.data();

        for (int dir = 0; dir < 3; dir++) {

            for (amrex::MFIter mfi(*dofs_mfarrvec[lev][dir]); mfi.isValid(); ++mfi) {

                auto bx = mfi.tilebox();
                auto dof_lhs_arr = dofs_lhs_mfarrvec[lev][dir]->const_array(mfi);

#if defined(WARPX_DIM_1D_Z)
                auto dof_arr = dofs_mfarrvec[lev][dir]->const_array(mfi);
#elif defined(WARPX_DIM_RCYLINDER)
                amrex::ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE("MatrixPC<T,Ops>::Assemble() not yet implemented for WARPX_DIM_RCYLINDER");
#elif defined(WARPX_DIM_RSPHERE)
                amrex::ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE("MatrixPC<T,Ops>::Assemble() not yet implemented for WARPX_DIM_RSPHERE");
#elif defined(WARPX_DIM_RZ)
                amrex::ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE("MatrixPC<T,Ops>::Assemble() not yet implemented for WARPX_DIM_RZ");
#elif defined(WARPX_DIM_XZ)
                auto dof_arr = dofs_mfarrvec[lev][dir]->const_array(mfi);
                int tdir = -1;
                if  (dir == 0) { tdir = 2; }
                else if (dir == 2) { tdir = 0; }
                else { tdir = 1; }
                auto dof_tdir_arr = dofs_mfarrvec[lev][tdir]->const_array(mfi);
#elif defined(WARPX_DIM_3D)
                int tdir1 = (dir + 1) % 3;
                int tdir2 = (dir + 2) % 3;
                GpuArray<Array4<const int>, AMREX_SPACEDIM>
                    const dof_arrays {{ AMREX_D_DECL( dofs_mfarrvec[lev][dir]->const_array(mfi),
                                                      dofs_mfarrvec[lev][tdir1]->const_array(mfi),
                                                      dofs_mfarrvec[lev][tdir2]->const_array(mfi) ) }};
#else
                amrex::ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE("MatrixPC<T,Ops>::Assemble encountered unknown dimensionality");
#endif

                // Add the Maxwell's piece
                if (thetaDt > 0.0) {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int icol = 0;
                        int ridx_l = dof_lhs_arr(i,j,k,0);
                        int ridx_g = dof_lhs_arr(i,j,k,1);
                        AMREX_ALWAYS_ASSERT((ridx_l >= 0) && (ridx_l < ndofs_l));
                        AMREX_ALWAYS_ASSERT((ridx_g >= 0) && (ridx_g < ndofs_g));

                        r_indices_g_ptr[ridx_l] = ridx_g;

                        {
                            int cidx_g_lhs = dof_lhs_arr(i,j,k,1);
                            amrex::Real val = 1.0;
                            auto flag = MatrixPCUtils::insertOrAdd( cidx_g_lhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }

#if defined(WARPX_DIM_1D_Z)
                        if (dir != 2) {
                            {
                                int cidx_g_rhs = dof_arr(i,j,k,1);
                                amrex::Real val = 2.0_rt*alpha * dxi[0]*dxi[0];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_arr(i-1,j,k,1);
                                amrex::Real val = -alpha * dxi[0]*dxi[0];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_arr(i+1,j,k,1);
                                amrex::Real val = -alpha * dxi[0]*dxi[0];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
#elif defined(WARPX_DIM_XZ)
                        {
                            int cidx_g_rhs = dof_arr(i,j,k,1);
                            amrex::Real val = 0.0;
                            if (dir == 0) {
                                val += 2.0_rt*alpha * dxi[1]*dxi[1];
                            } else if (dir == 2) {
                                val += 2.0_rt*alpha * dxi[0]*dxi[0];
                            } else {
                                val += 2.0_rt*alpha *(dxi[0]*dxi[0] + dxi[1]*dxi[1]);
                            }
                            auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        if ((dir == 0) || (dir == 2)) {
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_arr(i,j-1,k,1) : dof_arr(i-1,j,k,1));
                                amrex::Real val = -alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_arr(i,j+1,k,1) : dof_arr(i+1,j,k,1));
                                amrex::Real val = -alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            // The following four blocks are for
                            // dir = 0: d/dx(dEz/dz) at Ex(i,j) with Ex centered in x and nodal in z
                            // dir = 2: d/dz(dEx/dx) at Ez(i,j) with Ez centered in z and nodal in x
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i,j-1,k,1) : dof_tdir_arr(i-1,j,k,1));
                                amrex::Real val = alpha * dxi[0] * dxi[1];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_tdir_arr(i,j,k,1);
                                amrex::Real val = -alpha * dxi[0] * dxi[1];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j-1,k,1) : dof_tdir_arr(i-1,j+1,k,1));
                                amrex::Real val = -alpha * dxi[0] * dxi[1];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j,k,1) : dof_tdir_arr(i,j+1,k,1));
                                amrex::Real val = alpha * dxi[0] * dxi[1];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        } else {
                            for (int jdir = 0; jdir <= 2; jdir+=2) {
                                {
                                    int cidx_g_rhs = (jdir == 0 ? dof_arr(i-1,j,k,1) : dof_arr(i,j-1,k,1));
                                    amrex::Real val = -alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1];
                                    auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                                                            &a_ij_ptr[ridx_l*nnz_max],
                                                                            nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                                {
                                    int cidx_g_rhs = (jdir == 0 ? dof_arr(i+1,j,k,1) : dof_arr(i,j+1,k,1));
                                    amrex::Real val = -alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1];
                                    auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                                                            &a_ij_ptr[ridx_l*nnz_max],
                                                                            nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#elif defined(WARPX_DIM_3D)
                        amrex::IntVect dvec(AMREX_D_DECL(dir,tdir1,tdir2));
                        amrex::IntVect ic(AMREX_D_DECL(i,j,k));
                        {
                            int cidx_g_rhs = dof_arrays[0](ic,1);
                            amrex::Real val = 2.0_rt * alpha * (dxi[dvec[1]]*dxi[dvec[1]] + dxi[dvec[2]]*dxi[dvec[2]]);
                            auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        for (int ctr = -1; ctr <= 1; ctr += 2) {
                            for (int tdir = 1; tdir <= 2; tdir++) {
                                auto iv = ic; iv[dvec[tdir]] += ctr;
                                int cidx_g_rhs = dof_arrays[0](iv,1);
                                amrex::Real val = -alpha * dxi[dvec[tdir]]*dxi[dvec[tdir]];
                                auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
                        for (int ctr_dir = -1; ctr_dir <= 0; ctr_dir++) {
                            for (int ctr_tdir = -1; ctr_tdir <= 0; ctr_tdir++) {
                                for (int tdir = 1; tdir <= 2; tdir++) {
                                    auto iv = ic; iv[dvec[0]] += (ctr_dir+1); iv[dvec[tdir]] += ctr_tdir;
                                    auto sign = std::copysign(1,ctr_dir) * std::copysign(1,ctr_tdir);
                                    int cidx_g_rhs = dof_arrays[tdir](iv,1);
                                    amrex::Real val = amrex::Real(sign) * alpha * dxi[dvec[0]]*dxi[dvec[tdir]];
                                    auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                                                            &a_ij_ptr[ridx_l*nnz_max],
                                                                            nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#endif
                        num_nz_ptr[ridx_l] = icol;
                    });
                }

                // Add the mass matrix piece
                if (m_pc_mass_matrix_width >= 0) {

                    auto mm_width = m_pc_mass_matrix_width;
                    auto mm_incl_ij = m_pc_mass_matrix_include_ij;
                    auto sigma_ii_arr = (*m_bcoefs)[lev][dir]->const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int ridx_l = dof_lhs_arr(i,j,k,0);
                        AMREX_ALWAYS_ASSERT((ridx_l >= 0) && (ridx_l < ndofs_l));
                        int icol = num_nz_ptr[ridx_l];

                        {
                            int cidx_g_lhs = dof_lhs_arr(i,j,k,1);
                            amrex::Real val = sigma_ii_arr(i,j,k,0) - 1.0;
                            auto flag = MatrixPCUtils::insertOrAdd( cidx_g_lhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }

                        amrex::ignore_unused(mm_width); // will be used when adding off-diagonal elements
                        amrex::ignore_unused(mm_incl_ij); // will be used when adding off-diagonal elements

                        num_nz_ptr[ridx_l] = icol;
                    });
                }
                Gpu::synchronize();
            }
        }

        nnz_actual = std::max(nnz_actual, *(nnz_actual_d.copyToHost()));
    }

    amrex::ParallelDescriptor::ReduceIntMax(&nnz_actual, 1);
    return (nnz_actual - nnz_max);
}

template <class T, class Ops>
void MatrixPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    //  Given a right-hand-side b, solve:
    //      A x = b
    //  where A is the linear operator, in this case, the curl-curl
    //  operator:
    //      A x = curl (alpha * curl (x) ) + beta * x

    BL_PROFILE("MatrixPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "MatrixPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "MatrixPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "MatrixPC::Apply() - a_b must be Efield_fp type");

    WARPX_ABORT_WITH_MESSAGE("MatrixPC<T,Ops>::Apply() - native matrix solvers not implemented. Use with external library, eg, PETSc.");

}

#endif

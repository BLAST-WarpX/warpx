/* Copyright 2024 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CURL_CURL_MLMG_PC_H_
#define CURL_CURL_MLMG_PC_H_

#include "Fields.H"
#include "Utils/WarpXConst.H"
#include "Preconditioner.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#ifndef WARPX_DIM_1D_Z // currently not implemented in 1D
#include <AMReX_MLLinOp.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLMG.H>
#include <AMReX_GMRES_MLMG.H>
#endif

/**
 * \brief Curl-curl Preconditioner
 *
 *  Preconditioner that solves the curl-curl equation for the E-field, given
 *  a RHS. Uses AMReX's curl-curl linear operator and multigrid solver.
 *
 *  The equation solves for Eg in:
 *  curl ( alpha * curl ( Eg ) ) + beta * Eg = b
 *  where
 *    + alpha is a scalar
 *    + beta can either be a scalar that is constant in space or a MultiFab
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
class CurlCurlMLMGPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        CurlCurlMLMGPC () = default;

        /**
         * \brief Default destructor
         */
        ~CurlCurlMLMGPC () override = default;

        // Prohibit move and copy operations
        CurlCurlMLMGPC(const CurlCurlMLMGPC&) = delete;
        CurlCurlMLMGPC& operator=(const CurlCurlMLMGPC&) = delete;
        CurlCurlMLMGPC(CurlCurlMLMGPC&&) noexcept = delete;
        CurlCurlMLMGPC& operator=(CurlCurlMLMGPC&&) noexcept = delete;

        /**
         * \brief Define the preconditioner
         */
        void Define (const T&, Ops*) override;

        /**
         * \brief Update the preconditioner
         */
        void Update (const T&) override;

        /**
         * \brief Apply (solve) the preconditioner given a RHS
         *
         *  Given a right-hand-side b, solve:
         *      A x = b
         *  where A is the linear operator, in this case, the curl-curl operator:
         *      A x = curl (alpha * curl (x) ) + beta * x
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
        bool m_bottom_verbose = false;
        bool m_agglomeration = true;
        bool m_consolidation = true;
        bool m_use_gmres = false;
        bool m_use_gmres_pc = true;

        int m_max_iter = 10;
        int m_max_coarsening_level = 30;

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;
        amrex::Vector<amrex::Geometry> m_geom;
        amrex::Vector<amrex::BoxArray> m_grids;
        amrex::Vector<amrex::DistributionMapping> m_dmap;
        amrex::IntVect m_gv;

// currently not implemented in 1D
#ifndef WARPX_DIM_1D_Z
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> m_bc_lo;
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> m_bc_hi;

        std::unique_ptr<amrex::LPInfo> m_info;
        std::unique_ptr<amrex::MLCurlCurl> m_curl_curl;
        std::unique_ptr<amrex::MLMGT<MFArr>> m_solver;
        std::unique_ptr<amrex::GMRESMLMGT<MFArr>> m_gmres_solver;
#endif

        /**
         * \brief Read parameters
         */
        void readParameters();

    private:

};

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_curl_curl_mlmg);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " bottom verbose:       " << (m_bottom_verbose?"true":"false") << "\n";
    Print() << pc_name << " max iter:             " << m_max_iter << "\n";
    Print() << pc_name << " agglomeration:        " << m_agglomeration << "\n";
    Print() << pc_name << " consolidation:        " << m_consolidation << "\n";
    Print() << pc_name << " max_coarsening_level: " << m_max_coarsening_level << "\n";
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
    Print() << pc_name << " use GMRES:            " << (m_use_gmres?"true":"false") << "\n";
    if (m_use_gmres) {
        Print() << pc_name
                << " use PC for GMRES:     "
                << (m_use_gmres_pc?"true":"false") << "\n";
    }
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_curl_curl_mlmg));
    pp.query("verbose", m_verbose);
    pp.query("bottom_verbose", m_bottom_verbose);
    pp.query("max_iter", m_max_iter);
    pp.query("agglomeration", m_agglomeration);
    pp.query("consolidation", m_consolidation);
    pp.query("max_coarsening_level", m_max_coarsening_level);
    pp.query("absolute_tolerance",  m_atol);
    pp.query("relative_tolerance",  m_rtol);
    pp.query("use_gmres",  m_use_gmres);
    pp.query("use_gmres_pc",  m_use_gmres_pc);
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::Define ( const T& a_U,
                                     Ops* const a_ops )
{
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "CurlCurlMLMGPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "CurlCurlMLMGPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "CurlCurlMLMGPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    // read preconditioner parameters
    readParameters();

// currently not implemented in 1D
#ifdef WARPX_DIM_1D_Z
    WARPX_ABORT_WITH_MESSAGE("CurlCurlMLMGPC not yet implemented for 1D");
#else
    // create info object for curl-curl op
    m_info = std::make_unique<LPInfo>();
    m_info->setAgglomeration(m_agglomeration);
    m_info->setConsolidation(m_consolidation);
    m_info->setMaxCoarseningLevel(m_max_coarsening_level);

    // Get data vectors from a_U
    auto& u_mfarrvec = a_U.getArrayVec();

    // Set number of AMR levels and create geometry, grids, and
    // distribution mapping vectors.
    m_num_amr_levels = m_ops->numAMRLevels();
    m_geom.resize(m_num_amr_levels);
    m_grids.resize(m_num_amr_levels);
    m_dmap.resize(m_num_amr_levels);
    for (int n = 0; n < m_num_amr_levels; n++) {
        m_geom[n] = m_ops->GetGeometry(n);
        m_dmap[n] = u_mfarrvec[n][0]->DistributionMap();

        BoxArray ba = u_mfarrvec[n][0]->boxArray();
        m_grids[n] = ba.enclosedCells();
    }

    // Construct the curl-curl linear operator and set its BCs
    m_curl_curl = std::make_unique<MLCurlCurl>(m_geom, m_grids, m_dmap, *m_info);
    m_curl_curl->setDomainBC(m_ops->GetLinOpBCLo(), m_ops->GetLinOpBCHi());

    // Dummy value for alpha and beta to avoid abort due to degenerate matrix by MLMG solver
    m_curl_curl->setScalars(1.0, 1.0);

    // Construct the MLMG solver
    m_solver = std::make_unique<MLMGT<MFArr>>(*m_curl_curl);
    m_solver->setMaxIter(m_max_iter);
    m_solver->setFixedIter(m_max_iter);
    m_solver->setVerbose(static_cast<int>(m_verbose));
    m_solver->setBottomVerbose(static_cast<int>(m_bottom_verbose));

    // If using GMRES solver, construct it
    if (m_use_gmres) {
        m_gmres_solver = std::make_unique<GMRESMLMGT<MFArr>>(*m_solver);
        m_gmres_solver->usePrecond(m_use_gmres_pc);
        m_gmres_solver->setPrecondNumIters(m_max_iter);
        m_gmres_solver->setVerbose(static_cast<int>(m_verbose));
    }
#endif

    m_is_defined = true;
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::Update (const T& a_U)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "CurlCurlMLMGPC::Update() called on undefined object" );

    // a_U is not needed for a linear operator
    amrex::ignore_unused(a_U);

    // set the coefficients alpha and beta for curl-curl op
    const RT alpha = (m_ops->theta()*this->m_dt*PhysConst::c) * (m_ops->theta()*this->m_dt*PhysConst::c);
    const RT beta = RT(1.0);

// currently not implemented in 1D
#ifndef WARPX_DIM_1D_Z
    m_curl_curl->setScalars(alpha, beta);
#endif

    if (m_verbose) {
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_curl_curl_mlmg)
                       << ": dt = " << this->m_dt << ", "
                       << " coefficients: "
                       << "alpha = " << alpha << ", "
                       << "beta = " << beta << "\n";
    }
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    //  Given a right-hand-side b, solve:
    //      A x = b
    //  where A is the linear operator, in this case, the curl-curl
    //  operator:
    //      A x = curl (alpha * curl (x) ) + beta * x

    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "CurlCurlMLMGPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "CurlCurlMLMGPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "CurlCurlMLMGPC::Apply() - a_b must be Efield_fp type");

    // Get the data vectors
    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        ((b_mfarrvec.size() == m_num_amr_levels) && (x_mfarrvec.size() == m_num_amr_levels)),
        "Error in CurlCurlMLMGPC::Apply() - mismatch in number of levels." );

    for (int n = 0; n < m_num_amr_levels; n++) {

        // Copy initial guess to local object
#if defined(WARPX_DIM_1D_Z)
        // Missing dimensions is x,y in WarpX and y,z in AMReX
        WARPX_ABORT_WITH_MESSAGE("CurlCurlMLMGPC not yet implemented for 1D");
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        // Missing dimension is y in WarpX and z in AMReX
        Array<MultiFab,3> solution { MultiFab(*x_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][2], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][1], make_alias, 0, 1) };
        Array<MultiFab,3> rhs      { MultiFab(*b_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][2], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][1], make_alias, 0, 1) };
#elif defined(WARPX_DIM_3D)
        Array<MultiFab,3> solution { MultiFab(*x_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][2], make_alias, 0, 1) };
        Array<MultiFab,3> rhs      { MultiFab(*b_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][2], make_alias, 0, 1) };
#endif

// currently not implemented in 1D
#ifndef WARPX_DIM_1D_Z
        m_curl_curl->prepareRHS({&rhs});
        if (m_use_gmres) {
            m_gmres_solver->solve(solution, rhs, m_rtol, m_atol);
        } else {
            m_solver->solve({&solution}, {&rhs}, m_rtol, m_atol);
        }
#endif
    }
}

#endif

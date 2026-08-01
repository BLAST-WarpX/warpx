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
#include "Utils/TextMsg.H"
#include "Preconditioner.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLMG.H>
#include <AMReX_GMRES_MLMG.H>

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

        /**
         * \brief Print parameters
         */
        void printParameters() const override;

        /**
         * \brief Check if the nonlinear solver has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        using MFArr = amrex::Array<amrex::MultiFab,3>;

        /** Hall-Schur PC application (public: CUDA requires the enclosing
         *  function of extended device lambdas to have public access). */
        void ApplyHallSchur (MFArr& a_sol, MFArr& a_rhs);

    protected:

        bool m_is_defined = false;

        bool m_verbose = true;
        bool m_bottom_verbose = false;
        bool m_agglomeration = true;
        bool m_consolidation = true;
        bool m_use_gmres = false;
        bool m_use_gmres_pc = true;

        int m_max_iter = 10;
        int m_max_coarsening_level = 30;

        bool m_beta_scalar = true;

        /** Hall-Schur variant (input pc_curl_curl_mlmg.hall_schur): precondition the
         *  skew Hall block A = I + S, S v = (curl(alpha curl v)) x b-hat, using the
         *  factorization A^{-1} ~ (I - S)(I - S^2)^{-1} with (I - S^2)^{-1}
         *  approximated by two nested (I + alpha curl curl) MLMG solves, applied to
         *  the perpendicular-to-B projection of the residual (identity on the
         *  parallel part, where the Hall response vanishes). 3D Yee only. */
        bool m_hall_schur = false;
        bool m_ws_defined = false;
        MFArr m_ws_rperp, m_ws_t, m_ws_u, m_ws_w;
        amrex::MultiFab m_ws_na, m_ws_nb;
        const amrex::Vector<amrex::MultiFab*>* m_bunit = nullptr;

        bool m_alpha_scalar = true; // For hybrid solver, we need spatially varying alpha

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;
        amrex::Vector<amrex::Geometry> m_geom;
        amrex::Vector<amrex::BoxArray> m_grids;
        amrex::Vector<amrex::DistributionMapping> m_dmap;
        amrex::IntVect m_gv;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        const amrex::Vector<amrex::MultiFab*>* m_acoefs_pc = nullptr;

        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> m_bc_lo;
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> m_bc_hi;

        std::unique_ptr<amrex::LPInfo> m_info;
        std::unique_ptr<amrex::MLCurlCurl> m_curl_curl;
        std::unique_ptr<amrex::MLMGT<MFArr>> m_solver;
        std::unique_ptr<amrex::GMRESMLMGT<MFArr>> m_gmres_solver;

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
    Print() << pc_name << " beta coeff:           " << (m_beta_scalar?"scalar (1.0)":"MultiFab") << "\n";
    Print() << pc_name << " alpha coeff:          " << (m_alpha_scalar?"scalar":"MultiFab") << "\n";
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
    pp.query("hall_schur", m_hall_schur);
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
    BL_PROFILE("CurlCurlMLMGPC::Define()");
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

    // Get data vectors from a_U
    auto& u_mfarrvec = a_U.getArrayVec();

    // create info object for curl-curl op
    m_info = std::make_unique<LPInfo>();
    m_info->setAgglomeration(m_agglomeration);
    m_info->setConsolidation(m_consolidation);
    m_info->setMaxCoarseningLevel(m_max_coarsening_level);

    // Set number of AMR levels and create geometry, grids, and
    // distribution mapping vectors.
    m_num_amr_levels = m_ops->numAMRLevels();
    if (m_num_amr_levels > 1) {
        WARPX_ABORT_WITH_MESSAGE("CurlCurlMLMGPC::Define(): m_num_amr_levels > 1");
    }
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

    m_bcoefs = m_ops->GetMassMatricesCoeff();
    if (m_bcoefs != nullptr) { m_beta_scalar = false; }

    m_acoefs_pc = m_ops->GetAlphaCoeff();
    if (m_acoefs_pc != nullptr) { m_alpha_scalar = false; }

    m_bunit = m_ops->GetBUnitCoeff();

    m_is_defined = true;
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("CurlCurlMLMGPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "CurlCurlMLMGPC::Update() called on undefined object" );

    // a_U is not needed for a linear operator
    amrex::ignore_unused(a_U);

    // set the alpha coefficient for the curl-curl op
    if (m_alpha_scalar) {
        const RT thetaDt = m_ops->GetThetaForPC()*this->m_dt;
        if (thetaDt==0.) {
            std::stringstream warningMsg;
            warningMsg << "Light waves are not treated implicitly with the chosen time solver.\n";
            warningMsg << "Use jacobian.pc_type = pc_jacobi for better performance.\n";
            ablastr::warn_manager::WMRecordWarning("CurlCurlMLMGPC", warningMsg.str());
        }
        const RT alpha = (thetaDt*PhysConst::c) * (thetaDt*PhysConst::c);
        m_curl_curl->setScalars(alpha, RT(1.0));
        if (m_verbose) {
            amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_curl_curl_mlmg)
                           << ": theta*dt = " << thetaDt << ", "
                           << " coefficients: "
                           << "alpha = " << alpha << "\n";
        }
    } else {
        m_curl_curl->setScalars(RT(1.0), RT(1.0));
        const auto* acoefs_ptr = m_ops->GetAlphaCoeff();
        amrex::Vector<amrex::MultiFab const*> acoefs(m_num_amr_levels);
        for (int n = 0; n < m_num_amr_levels; n++) { acoefs[n] = (*acoefs_ptr)[n]; }
        m_curl_curl->setAlpha(acoefs);
        if (m_verbose) {
            amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_curl_curl_mlmg)
                           << ": alpha = spatially varying\n";
        }
    }

    if (!m_beta_scalar) {
        for (int n = 0; n < m_num_amr_levels; n++) {
#if defined(WARPX_DIM_1D_Z)
            // Missing dimensions is x,y in WarpX and y,z in AMReX
            m_curl_curl->setBeta({Array<MultiFab const*,3>{ (*m_bcoefs)[n][2],
                                                            (*m_bcoefs)[n][1],
                                                            (*m_bcoefs)[n][0]}});
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Missing dimensions is y,z in WarpX and y,z in AMReX
            m_curl_curl->setBeta({Array<MultiFab const*,3>{ (*m_bcoefs)[n][0],
                                                            (*m_bcoefs)[n][1],
                                                            (*m_bcoefs)[n][2]}});
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // Missing dimension is y in WarpX and z in AMReX
            m_curl_curl->setBeta({Array<MultiFab const*,3>{ (*m_bcoefs)[n][0],
                                                            (*m_bcoefs)[n][2],
                                                            (*m_bcoefs)[n][1]}});
#elif defined(WARPX_DIM_3D)
            m_curl_curl->setBeta({Array<MultiFab const*,3>{ (*m_bcoefs)[n][0],
                                                            (*m_bcoefs)[n][1],
                                                            (*m_bcoefs)[n][2]}});
#endif
        }
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

    BL_PROFILE("CurlCurlMLMGPC::Apply()");
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
        // Missing dimensions are x,y in WarpX and y,z in AMReX
        Array<MultiFab,3> solution { MultiFab(*x_mfarrvec[n][2], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][0], make_alias, 0, 1) };
        Array<MultiFab,3> rhs      { MultiFab(*b_mfarrvec[n][2], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][0], make_alias, 0, 1) };
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        // Missing dimensions are y,z in WarpX and y,z in AMReX
        Array<MultiFab,3> solution { MultiFab(*x_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*x_mfarrvec[n][2], make_alias, 0, 1) };
        Array<MultiFab,3> rhs      { MultiFab(*b_mfarrvec[n][0], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][1], make_alias, 0, 1),
                                     MultiFab(*b_mfarrvec[n][2], make_alias, 0, 1) };
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

        if (m_hall_schur) {
            ApplyHallSchur(solution, rhs);
            continue;
        }

        m_curl_curl->prepareRHS({&rhs});
        if (m_use_gmres) {
            m_gmres_solver->solve(solution, rhs, m_rtol, m_atol);
        } else {
            m_solver->solve({&solution}, {&rhs}, m_rtol, m_atol);
        }
    }
}

template <class T, class Ops>
void CurlCurlMLMGPC<T,Ops>::ApplyHallSchur (MFArr& a_sol, MFArr& a_rhs)
{
    BL_PROFILE("CurlCurlMLMGPC::ApplyHallSchur()");
    using namespace amrex;

#if !defined(WARPX_DIM_3D)
    amrex::ignore_unused(a_sol, a_rhs);
    WARPX_ABORT_WITH_MESSAGE("pc_curl_curl_mlmg.hall_schur is only implemented in 3D");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_num_amr_levels == 1,
        "hall_schur: single level only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_alpha_scalar && m_bunit != nullptr,
        "hall_schur requires spatially varying alpha and a b-unit field (hybrid solver)");

    // Lazily allocate workspaces matching the (edge-staggered) rhs layout and a
    // nodal 3-component scratch pair (1 ghost for the edge<->node interpolations).
    if (!m_ws_defined) {
        for (int d = 0; d < 3; ++d) {
            m_ws_rperp[d].define(a_rhs[d].boxArray(), a_rhs[d].DistributionMap(), 1, 1);
            m_ws_t[d].define(a_rhs[d].boxArray(), a_rhs[d].DistributionMap(), 1, 1);
            m_ws_u[d].define(a_rhs[d].boxArray(), a_rhs[d].DistributionMap(), 1, 1);
            m_ws_w[d].define(a_rhs[d].boxArray(), a_rhs[d].DistributionMap(), 1, 1);
        }
        const MultiFab& bu = *(*m_bunit)[0];
        m_ws_na.define(bu.boxArray(), bu.DistributionMap(), 3, 1);
        m_ws_nb.define(bu.boxArray(), bu.DistributionMap(), 3, 1);
        m_ws_defined = true;
    }

    const auto& period = m_geom[0].periodicity();
    const MultiFab& bu = *(*m_bunit)[0];

    // edge -> node average of a 3-component edge set into a nodal 3-comp MF
    auto edge2node = [&](const MFArr& e, MultiFab& n_mf) {
        for (MFIter mfi(n_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box tb = mfi.tilebox();
            auto const& nda = n_mf.array(mfi);
            auto const& ex = e[0].const_array(mfi);
            auto const& ey = e[1].const_array(mfi);
            auto const& ez = e[2].const_array(mfi);
            ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                nda(i,j,k,0) = 0.5_rt*(ex(i-1,j,k) + ex(i,j,k));
                nda(i,j,k,1) = 0.5_rt*(ey(i,j-1,k) + ey(i,j,k));
                nda(i,j,k,2) = 0.5_rt*(ez(i,j,k-1) + ez(i,j,k));
            });
        }
        n_mf.FillBoundary(period);
    };

    // node -> edge average of a nodal 3-comp MF, SUBTRACTED from (src) into (dst):
    // dst_c = src_c - interp(n_mf[c])
    auto sub_node2edge = [&](const MFArr& src, const MultiFab& n_mf, MFArr& dst) {
        for (int d = 0; d < 3; ++d) {
            for (MFIter mfi(dst[d], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box tb = mfi.tilebox();
                auto const& out = dst[d].array(mfi);
                auto const& in  = src[d].const_array(mfi);
                auto const& nda = n_mf.const_array(mfi);
                if (d == 0) {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) = in(i,j,k) - 0.5_rt*(nda(i,j,k,0) + nda(i+1,j,k,0));
                    });
                } else if (d == 1) {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) = in(i,j,k) - 0.5_rt*(nda(i,j,k,1) + nda(i,j+1,k,1));
                    });
                } else {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) = in(i,j,k) - 0.5_rt*(nda(i,j,k,2) + nda(i,j,k+1,2));
                    });
                }
            }
            dst[d].FillBoundary(period);
        }
    };

    // add node -> edge interp of n_mf into dst (dst_c += interp(n_mf[c]))
    auto add_node2edge = [&](const MultiFab& n_mf, MFArr& dst) {
        for (int d = 0; d < 3; ++d) {
            for (MFIter mfi(dst[d], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box tb = mfi.tilebox();
                auto const& out = dst[d].array(mfi);
                auto const& nda = n_mf.const_array(mfi);
                if (d == 0) {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) += 0.5_rt*(nda(i,j,k,0) + nda(i+1,j,k,0));
                    });
                } else if (d == 1) {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) += 0.5_rt*(nda(i,j,k,1) + nda(i,j+1,k,1));
                    });
                } else {
                    ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        out(i,j,k) += 0.5_rt*(nda(i,j,k,2) + nda(i,j,k+1,2));
                    });
                }
            }
        }
    };

    // ---- 1. rn = edge->node(rhs); rpar_n = b (b . rn)  (stored in m_ws_nb) ----
    for (int d = 0; d < 3; ++d) {
        MultiFab::Copy(m_ws_rperp[d], a_rhs[d], 0, 0, 1, 0);
        m_ws_rperp[d].FillBoundary(period);
    }
    edge2node(m_ws_rperp, m_ws_na);
    for (MFIter mfi(m_ws_nb, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox();
        auto const& rp = m_ws_nb.array(mfi);
        auto const& rn = m_ws_na.const_array(mfi);
        auto const& b  = bu.const_array(mfi);
        ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const Real dot = b(i,j,k,0)*rn(i,j,k,0)
                           + b(i,j,k,1)*rn(i,j,k,1)
                           + b(i,j,k,2)*rn(i,j,k,2);
            rp(i,j,k,0) = b(i,j,k,0)*dot;
            rp(i,j,k,1) = b(i,j,k,1)*dot;
            rp(i,j,k,2) = b(i,j,k,2)*dot;
        });
    }
    m_ws_nb.FillBoundary(period);

    // ---- 2. r_perp = rhs - node->edge(rpar_n) ----
    sub_node2edge(m_ws_rperp, m_ws_nb, m_ws_u);   // m_ws_u = r_perp
    for (int d = 0; d < 3; ++d) {
        MultiFab::Copy(m_ws_rperp[d], m_ws_u[d], 0, 0, 1, 1);
    }

    // ---- 3. t = curl(alpha curl r_perp) = A_mlmg(r_perp) - beta*r_perp ----
    m_curl_curl->apply(0, 0, m_ws_t, m_ws_u,
                       MLLinOpT<MFArr>::BCMode::Homogeneous,
                       MLLinOpT<MFArr>::StateMode::Correction);
    for (int d = 0; d < 3; ++d) {
        MultiFab::Subtract(m_ws_t[d], m_ws_rperp[d], 0, 0, 1, 0);
        m_ws_t[d].FillBoundary(period);
    }

    // ---- 4. sweep u = r_perp - (t x b-hat) ----
    edge2node(m_ws_t, m_ws_na);
    for (MFIter mfi(m_ws_na, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox();
        auto const& tn = m_ws_na.array(mfi);
        auto const& b  = bu.const_array(mfi);
        ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const Real tx = tn(i,j,k,0), ty = tn(i,j,k,1), tz = tn(i,j,k,2);
            const Real bx = b(i,j,k,0),  by = b(i,j,k,1),  bz = b(i,j,k,2);
            tn(i,j,k,0) = ty*bz - tz*by;
            tn(i,j,k,1) = tz*bx - tx*bz;
            tn(i,j,k,2) = tx*by - ty*bx;
        });
    }
    m_ws_na.FillBoundary(period);
    sub_node2edge(m_ws_rperp, m_ws_na, m_ws_u);   // u = r_perp - S r_perp

    // ---- 5. two nested (I + alpha curl curl) solves: w ~ (I - S^2)^{-1} u ----
    for (int d = 0; d < 3; ++d) { m_ws_w[d].setVal(0.0_rt); }
    m_curl_curl->prepareRHS({&m_ws_u});
    m_solver->solve({&m_ws_w}, {&m_ws_u}, m_rtol, m_atol);
    for (int d = 0; d < 3; ++d) {
        MultiFab::Copy(m_ws_u[d], m_ws_w[d], 0, 0, 1, 0);
        m_ws_w[d].setVal(0.0_rt);
    }
    m_curl_curl->prepareRHS({&m_ws_u});
    m_solver->solve({&m_ws_w}, {&m_ws_u}, m_rtol, m_atol);

    // ---- 6. sol = w + node->edge(rpar_n) ----
    for (int d = 0; d < 3; ++d) {
        MultiFab::Copy(a_sol[d], m_ws_w[d], 0, 0, 1, 0);
    }
    add_node2edge(m_ws_nb, a_sol);
#endif
}

#endif

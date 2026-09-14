/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef DARWIN_MLMG_PC_H_
#define DARWIN_MLMG_PC_H_

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Preconditioner.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>

/**
 * \brief Factored-Laplacian multigrid preconditioner for the semi-implicit
 *  Darwin solver's GMRES iteration.
 *
 *  The Darwin field operator applied to the auxiliary variable Z is
 *      A(Z) = nabla^4(Z) + curl(chi curl(Z)),
 *  with chi = (2 mu0/dt) x the mass-matrix response. On the solenoidal
 *  subspace (div Z = 0) the plasma-response term is -chi nabla^2(Z), so in
 *  the constant-chi limit the operator factors exactly as
 *      A = (-nabla^2)(-nabla^2 + chi),
 *  and a spectrally equivalent preconditioner is two successive scalar
 *  elliptic solves: a Poisson solve followed by a Helmholtz solve with the
 *  spatially varying chi(x) as the MLABecLaplacian acoef.
 *
 *  Z is B-staggered while MLABecLaplacian is cell-centered, so each vector
 *  component is collocated onto the cell-centered grid by index
 *  identification (node i -> cell i, a half-cell shift in the component's
 *  nodal dimension), solved, and mapped back the same way. Deliberately NOT
 *  averaged: pair-averaging multiplies the nodal-dimension Nyquist modes by
 *  cos(k dx/2) = 0, making the preconditioner exactly singular on those
 *  planes.
 *
 *  The MLMG solves run a fixed number of V-cycles (max_iter) so the
 *  preconditioner is a fixed linear operator across a GMRES solve.
 */

template <class T, class Ops>
class DarwinMLMGPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        DarwinMLMGPC () = default;

        /**
         * \brief Default destructor
         */
        ~DarwinMLMGPC () override = default;

        // Prohibit move and copy operations
        DarwinMLMGPC (const DarwinMLMGPC&) = delete;
        DarwinMLMGPC& operator= (const DarwinMLMGPC&) = delete;
        DarwinMLMGPC (DarwinMLMGPC&&) noexcept = delete;
        DarwinMLMGPC& operator= (DarwinMLMGPC&&) noexcept = delete;

        /**
         * \brief Define the preconditioner
         */
        void Define (const T&, Ops*) override;

        /**
         * \brief Update the preconditioner: refresh chi(x) from the freshly
         *  deposited mass matrices (via Ops::ComputeScaledMassMatrixCC).
         */
        void Update () override;

        /**
         * \brief Apply (approximately solve) the preconditioner given a RHS:
         *  per Z component, collocate onto cell centers, solve
         *  (-nabla^2) y = b then (-nabla^2 + chi) x = y with fixed-cycle
         *  MLMG, and map back to the B-staggering.
         */
        void Apply (T&, const T&) override;

        /**
         * \brief Print parameters
         */
        void printParameters () const override;

        /**
         * \brief Check if this preconditioner has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;

        bool m_verbose = false;
        bool m_bottom_verbose = false;
        bool m_agglomeration = true;
        bool m_consolidation = true;

        int m_max_iter = 2;
        int m_max_coarsening_level = 30;

        // Zero by default, so that the solve performs a fixed number of iterations,
        // and the preconditioner is thus a fixed linear operator, as required by GMRES.
        RT m_atol = 0.0;
        RT m_rtol = 0.0;

        Ops* m_ops = nullptr;

        amrex::Geometry m_geom;
        amrex::BoxArray m_grids;
        amrex::DistributionMapping m_dmap;

        /**
         * \brief Cell-centered susceptibility chi(x), refreshed in Update().
         */
        amrex::MultiFab m_chi_cc;

        /**
         * \brief Cell-centered scratch shared by the per-component solves:
         *  RHS, intermediate (Poisson) solution, and final (Helmholtz)
         *  solution. The final solution carries one guard layer for the map
         *  back to the B-staggering (the top node plane in the nodal
         *  dimension reads the periodic image).
         */
        amrex::MultiFab m_rhs_cc, m_mid_cc, m_sol_cc;

        std::unique_ptr<amrex::LPInfo> m_info;
        std::unique_ptr<amrex::MLABecLaplacian> m_poisson;
        std::unique_ptr<amrex::MLABecLaplacian> m_helmholtz;
        std::unique_ptr<amrex::MLMG> m_poisson_mg;
        std::unique_ptr<amrex::MLMG> m_helmholtz_mg;

        /**
         * \brief Read parameters
         */
        void readParameters ();

    private:

};

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::printParameters () const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_darwin_mlmg);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " bottom verbose:       " << (m_bottom_verbose?"true":"false") << "\n";
    Print() << pc_name << " max iter (V-cycles):  " << m_max_iter << "\n";
    Print() << pc_name << " agglomeration:        " << m_agglomeration << "\n";
    Print() << pc_name << " consolidation:        " << m_consolidation << "\n";
    Print() << pc_name << " max_coarsening_level: " << m_max_coarsening_level << "\n";
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::readParameters ()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_darwin_mlmg));
    pp.query("verbose", m_verbose);
    pp.query("bottom_verbose", m_bottom_verbose);
    pp.query("max_iter", m_max_iter);
    pp.query("agglomeration", m_agglomeration);
    pp.query("consolidation", m_consolidation);
    pp.query("max_coarsening_level", m_max_coarsening_level);
    pp.query("absolute_tolerance", m_atol);
    pp.query("relative_tolerance", m_rtol);
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Define ( const T& a_U,
                                   Ops* const a_ops )
{
    BL_PROFILE("DarwinMLMGPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "DarwinMLMGPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "DarwinMLMGPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Define() must be called with a B-staggered solver vector");

    m_ops = a_ops;
    readParameters();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ops->numAMRLevels() == 1,
        "DarwinMLMGPC::Define(): only a single AMR level is supported");

    const auto& u_mfarrvec = a_U.getArrayVec();
    m_geom = m_ops->GetGeometry(0);
    m_dmap = u_mfarrvec[0][0]->DistributionMap();
    m_grids = u_mfarrvec[0][0]->boxArray();
    m_grids.enclosedCells();

    // The Darwin solver is restricted to fully periodic domains (see
    // SemiImplicitDarwin::Define()), so the cell-centered operators are too.
    Array<LinOpBCType, AMREX_SPACEDIM> bc_lo, bc_hi;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_geom.isPeriodic(idim),
            "DarwinMLMGPC::Define(): only periodic boundaries are supported");
        bc_lo[idim] = LinOpBCType::Periodic;
        bc_hi[idim] = LinOpBCType::Periodic;
    }

    m_info = std::make_unique<LPInfo>();
    m_info->setAgglomeration(m_agglomeration);
    m_info->setConsolidation(m_consolidation);
    m_info->setMaxCoarseningLevel(m_max_coarsening_level);

    // All three components share the same cell-centered grid, boundary
    // conditions and coefficients, so one operator pair serves all of them.
    m_poisson = std::make_unique<MLABecLaplacian>(
        Vector<Geometry>{m_geom}, Vector<BoxArray>{m_grids},
        Vector<DistributionMapping>{m_dmap}, *m_info);
    m_poisson->setDomainBC(bc_lo, bc_hi);
    m_poisson->setLevelBC(0, nullptr);
    m_poisson->setScalars(RT(0.0), RT(1.0));
    m_poisson->setACoeffs(0, RT(0.0));
    m_poisson->setBCoeffs(0, RT(1.0));

    m_helmholtz = std::make_unique<MLABecLaplacian>(
        Vector<Geometry>{m_geom}, Vector<BoxArray>{m_grids},
        Vector<DistributionMapping>{m_dmap}, *m_info);
    m_helmholtz->setDomainBC(bc_lo, bc_hi);
    m_helmholtz->setLevelBC(0, nullptr);
    m_helmholtz->setScalars(RT(1.0), RT(1.0));
    m_helmholtz->setACoeffs(0, RT(0.0)); // chi(x) set in Update()
    m_helmholtz->setBCoeffs(0, RT(1.0));

    m_poisson_mg = std::make_unique<MLMG>(*m_poisson);
    m_helmholtz_mg = std::make_unique<MLMG>(*m_helmholtz);
    for (auto* mg : {m_poisson_mg.get(), m_helmholtz_mg.get()}) {
        mg->setFixedIter(m_max_iter);
        mg->setVerbose(static_cast<int>(m_verbose));
        mg->setBottomVerbose(static_cast<int>(m_bottom_verbose));
    }

    m_chi_cc.define(m_grids, m_dmap, 1, 0);
    m_rhs_cc.define(m_grids, m_dmap, 1, 0);
    m_mid_cc.define(m_grids, m_dmap, 1, 0);
    m_sol_cc.define(m_grids, m_dmap, 1, 1);

    m_is_defined = true;
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Update ()
{
    BL_PROFILE("DarwinMLMGPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "DarwinMLMGPC::Update() called on undefined object" );

    m_ops->ComputeScaledMassMatrixCC(m_chi_cc);
    m_helmholtz->setACoeffs(0, m_chi_cc);

    if (m_verbose) {
        const amrex::Real chi_mean =
            m_chi_cc.sum(0, false) / static_cast<amrex::Real>(m_chi_cc.boxArray().numPts());
        amrex::Print() << "Updating "
                       << amrex::getEnumNameString(PreconditionerType::pc_darwin_mlmg)
                       << ": mean chi = " << chi_mean << "\n";
    }
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("DarwinMLMGPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "DarwinMLMGPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Apply() - a_x must be a B-staggered solver vector");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Apply() - a_b must be a B-staggered solver vector");

    const auto& b_mfarrvec = a_b.getArrayVec();
    const auto& x_mfarrvec = a_x.getArrayVec();
    const int lev = 0;

    for (int comp = 0; comp < 3; ++comp) {

        // Collocate the B-staggered RHS component onto the cell-centered
        // grid by index identification (node i -> cell i; identity for a
        // fully cell-centered component). The dropped top plane in the nodal
        // dimension is the duplicate periodic image, so no information is
        // lost.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_rhs_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            Array4<Real> const& rc = m_rhs_cc.array(mfi);
            Array4<const Real> const& bs = b_mfarrvec[lev][comp]->const_array(mfi);
            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                rc(i,j,k) = bs(i,j,k);
            });
        }

        // The two factored solves: Poisson then Helmholtz. Fixed V-cycle
        // count (setFixedIter in Define) keeps this a fixed linear operator.
        m_mid_cc.setVal(0.0);
        m_poisson_mg->solve({&m_mid_cc}, {&m_rhs_cc}, m_rtol, m_atol);
        m_sol_cc.setVal(0.0);
        m_helmholtz_mg->solve({&m_sol_cc}, {&m_mid_cc}, m_rtol, m_atol);

        // Map back to the B-staggering (cell i -> node i). Only the top node
        // plane in the nodal dimension reads a guard cell: the periodic
        // image filled by FillBoundary here.
        m_sol_cc.FillBoundary(m_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*x_mfarrvec[lev][comp], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            Array4<Real> const& xs = x_mfarrvec[lev][comp]->array(mfi);
            Array4<const Real> const& sc = m_sol_cc.const_array(mfi);
            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                xs(i,j,k) = sc(i,j,k);
            });
        }
    }
}

#endif

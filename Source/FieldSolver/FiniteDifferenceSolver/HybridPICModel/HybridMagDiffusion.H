/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Hybrid implicit magnetic diffusion (FLASH MagDiff-inspired)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_MAG_DIFFUSION_H_
#define WARPX_HYBRID_MAG_DIFFUSION_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

/**
 * \brief Operator-split (semi)implicit magnetic diffusion for hybrid-PIC.
 *
 * Advances the Yee magnetic field with a theta-method discretization of
 *
 *   dB/dt = - curl( (eta/mu0) curl B )
 *
 * using AMReX MLCurlCurl (same operator family as CurlCurlMLMGPC) in
 * Cartesian geometry. In RZ, AMReX MLCurlCurl does not support a
 * cylindrical 2D operator, so constant eta uses a matrix-free FDTD curl
 * chain with GMRES.
 * Inspired by FLASH 4.8 MagDiff / Grid_advanceMagDiffusion (theta scheme +
 * linear solve); reimplemented in C++/AMReX — no FLASH source.
 *
 * Default-off: Enable with hybrid_pic_model.implicit_mag_diffusion = 1.
 */
class HybridMagDiffusion
{
public:
    HybridMagDiffusion () = default;

    void ReadParameters ();

    [[nodiscard]] bool Enabled () const { return m_enabled; }
    [[nodiscard]] amrex::Real Theta () const { return m_theta; }
    [[nodiscard]] amrex::Real EtaExplicitMax () const { return m_eta_explicit_max; }
    [[nodiscard]] bool HasConstantEta () const { return m_has_constant_eta; }
    [[nodiscard]] amrex::Real ConstantEta () const { return m_constant_eta; }
    [[nodiscard]] bool UsesVariableEta () const { return m_use_variable_eta; }
    [[nodiscard]] int Verbose () const { return m_verbose; }

    /**
     * \brief Theta-method curl-curl magnetic diffusion on one level.
     *
     * Solves
     *   curl(alpha curl B) + beta B = rhs
     * with alpha = theta * dt * (eta_SI/mu0), beta = 1.
     * For theta = 1 (backward Euler), rhs = B_in.
     *
     * \param[in,out] Bfield  Yee B components (WarpX ordering)
     * \param[in]     eta_SI  resistivity [Ohm m]
     * \param[in]     dt      step length for this operator-split advance [s]
     * \param[in]     lev     AMR level (must be 0 for hybrid)
     * \param[in]     lobc    low domain BCs for MLCurlCurl
     * \param[in]     hibc    high domain BCs for MLCurlCurl
     */
    void Advance (
        ablastr::fields::VectorField const& Bfield,
        amrex::Real eta_SI,
        amrex::Real dt,
        int lev,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc) const;

    void AdvanceVariable (
        ablastr::fields::VectorField const& Bfield,
        ablastr::fields::VectorField const& eta_SI,
        amrex::Real dt,
        int lev,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc) const;

    /**
     * \brief Map WarpX field boundary types on this geometry to LinOp BCs.
     * Periodic/PEC→Dirichlet-like for edge data / Neumann / default Neumann.
     */
    static void GetLinOpBCs (
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>& hibc);

private:
    bool m_enabled = false;
    amrex::Real m_theta = 1.0;
    amrex::Real m_rtol = 1.e-8;
    amrex::Real m_atol = 1.e-12;
    int m_max_iter = 200;
    int m_verbose = 0;
    amrex::Real m_eta_explicit_max = 0.0;
    bool m_has_constant_eta = false;
    amrex::Real m_constant_eta = 0.0;
    bool m_use_variable_eta = false;
};

#endif // WARPX_HYBRID_MAG_DIFFUSION_H_

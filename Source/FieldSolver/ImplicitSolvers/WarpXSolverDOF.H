/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WarpXSolverDOF_H_
#define WarpXSolverDOF_H_

#include "Fields.H"
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_INT.H>
#include <string>

// forward declaration
class WarpX;

/**
 * \brief
 * This is contains the local and global DOF data that maps each grid location and
 * component to its position in a serialized vector.
 */
struct WarpXSolverDOF
{
    amrex::Vector<std::array<std::unique_ptr<amrex::iMultiFab>,3>> m_array;
    amrex::Vector<std::array<std::unique_ptr<amrex::iMultiFab>,3>> m_array_lhs;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_scalar;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_scalar_lhs;

    warpx::fields::FieldType m_array_type = warpx::fields::FieldType::None;
    warpx::fields::FieldType m_scalar_type = warpx::fields::FieldType::None;

    amrex::Long m_nDoFs_l = 0; /*!< Local nDOF */
    amrex::Long m_nDoFs_g = 0; /*!< Global nDOF */

    void Define ( WarpX* const, int, const std::string&, const std::string&);
};

#endif

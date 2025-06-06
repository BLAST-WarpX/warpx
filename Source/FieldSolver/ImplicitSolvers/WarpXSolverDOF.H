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
    ablastr::fields::MultiLevelVectorField m_array;
    ablastr::fields::MultiLevelScalarField m_scalar;

    warpx::fields::FieldType m_array_type = warpx::fields::FieldType::None;
    warpx::fields::FieldType m_scalar_type = warpx::fields::FieldType::None;

    amrex::Long m_nDoFs_l = 0; /*!< Local nDOF */
    amrex::Long m_nDoFs_g = 0; /*!< Global nDOF */

    void Define ( WarpX* const, int, const std::string&, const std::string&);
};

#endif

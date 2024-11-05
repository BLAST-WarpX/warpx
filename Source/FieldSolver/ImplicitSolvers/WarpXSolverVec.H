/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WarpXSolverVec_H_
#define WarpXSolverVec_H_

#include "Utils/TextMsg.H"
#include "Fields.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>


// forward declaration
class WarpX;

/**
 * \brief
 *  This is a wrapper class around a Vector of pointers to MultiFabs that
 *  contains basic math operators and functionality needed to interact with nonlinear
 *  solvers in WarpX and linear solvers in AMReX, such as GMRES. The size of the
 *  Vector is the number of amr levels. Hardcoded for 1 right now.
 *
 *  A WarpXSolverVec can consist of an array size 3 of MultiFabs (for vector fields
 *  such as E, B, and A) or of a single MultiFab for scalar fields. Both the array
 *  size 3 and scalar fields must be of type warpx::fields::FieldType.
 *  Additionally, a WarpXSolverVec can in general contain both an array size 3 field and a
 *  scalar field. For example, the array size 3 field can be used for the vector potential A
 *  and the scalar field can be used for the scalar potential phi, which is the full state of
 *  unknowns for a Darwin electromagnetic model.
 */
class WarpXSolverVec
{
public:

    WarpXSolverVec() = default;

    WarpXSolverVec(const WarpXSolverVec&) = delete;

    ~WarpXSolverVec();

    using value_type = amrex::Real;
    using RT = value_type;

    [[nodiscard]] inline bool IsDefined () const { return m_is_defined; }

    void Define ( WarpX*  a_WarpX,
             const std::string&  a_vector_type_name,
             const std::string&  a_scalar_type_name = "none" );

    inline
    void Define ( const WarpXSolverVec&  a_solver_vec )
    {
        assertIsDefined( a_solver_vec );
        m_num_amr_levels = a_solver_vec.m_num_amr_levels;
        Define( WarpXSolverVec::m_WarpX,
                a_solver_vec.getVectorType(),
                a_solver_vec.getScalarType() );
    }

    [[nodiscard]] RT dotProduct( const WarpXSolverVec&  a_X ) const;

    void Copy ( warpx::fields::FieldType  a_array_type,
                warpx::fields::FieldType  a_scalar_type = warpx::fields::FieldType::None );

    inline
    void Copy ( const WarpXSolverVec&  a_solver_vec )
    {
        assertIsDefined( a_solver_vec );
        if (IsDefined()) { assertSameType( a_solver_vec ); }
        else { Define(a_solver_vec); }

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; ++n) {
                    const amrex::MultiFab* this_field = a_solver_vec.getArrayVec()[lev][n];
                    amrex::MultiFab::Copy( *m_array_vec[lev][n], *this_field, 0, 0, m_ncomp,
                                           amrex::IntVect::TheZeroVector() );
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                const amrex::MultiFab* this_scalar = a_solver_vec.getScalarVec()[lev];
                amrex::MultiFab::Copy( *m_scalar_vec[lev], *this_scalar, 0, 0, m_ncomp,
                                       amrex::IntVect::TheZeroVector() );
            }
        }
    }

    // Prohibit Copy assignment operator
    WarpXSolverVec& operator= ( const WarpXSolverVec&  a_solver_vec ) = delete;

    // Move assignment operator
    WarpXSolverVec(WarpXSolverVec&&) noexcept = default;
    WarpXSolverVec& operator= ( WarpXSolverVec&&  a_solver_vec ) noexcept
    {
        if (this != &a_solver_vec) {
            m_array_vec = std::move(a_solver_vec.m_array_vec);
            m_scalar_vec = std::move(a_solver_vec.m_scalar_vec);
            m_array_type = a_solver_vec.m_array_type;
            m_scalar_type = a_solver_vec.m_scalar_type;
            m_is_defined = true;
        }
        return *this;
    }

    inline
    void operator+= ( const WarpXSolverVec&  a_solver_vec )
    {
        assertIsDefined( a_solver_vec );
        assertSameType( a_solver_vec );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->plus(*(a_solver_vec.getArrayVec()[lev][0]), 0, 1, 0);
                m_array_vec[lev][1]->plus(*(a_solver_vec.getArrayVec()[lev][1]), 0, 1, 0);
                m_array_vec[lev][2]->plus(*(a_solver_vec.getArrayVec()[lev][2]), 0, 1, 0);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->plus(*(a_solver_vec.getScalarVec()[lev]), 0, 1, 0);
            }
        }
    }

    inline
    void operator-= (const WarpXSolverVec& a_solver_vec)
    {
        assertIsDefined( a_solver_vec );
        assertSameType( a_solver_vec );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->minus(*(a_solver_vec.getArrayVec()[lev][0]), 0, 1, 0);
                m_array_vec[lev][1]->minus(*(a_solver_vec.getArrayVec()[lev][1]), 0, 1, 0);
                m_array_vec[lev][2]->minus(*(a_solver_vec.getArrayVec()[lev][2]), 0, 1, 0);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->minus(*(a_solver_vec.getScalarVec()[lev]), 0, 1, 0);
            }
        }
    }

    /**
     * \brief Y = a*X + b*Y
     */
    inline
    void linComb (const RT a, const WarpXSolverVec& X, const RT b, const WarpXSolverVec& Y)
    {
        assertIsDefined( X );
        assertIsDefined( Y );
        assertSameType( X );
        assertSameType( Y );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; n++) {
                    amrex::MultiFab::LinComb(*m_array_vec[lev][n], a, *X.getArrayVec()[lev][n], 0,
                                                                   b, *Y.getArrayVec()[lev][n], 0,
                                                                   0, 1, 0);
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                amrex::MultiFab::LinComb(*m_scalar_vec[lev], a, *X.getScalarVec()[lev], 0,
                                                             b, *Y.getScalarVec()[lev], 0,
                                                             0, 1, 0);
            }
        }
    }

    /**
     * \brief Increment Y by a*X (Y += a*X)
     */
    void increment (const WarpXSolverVec& X, const RT a)
    {
        assertIsDefined( X );
        assertSameType( X );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; n++) {
                    amrex::MultiFab::Saxpy( *m_array_vec[lev][n], a, *X.getArrayVec()[lev][n],
                                            0, 0, 1, amrex::IntVect::TheZeroVector() );
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                amrex::MultiFab::Saxpy( *m_scalar_vec[lev], a, *X.getScalarVec()[lev],
                                        0, 0, 1, amrex::IntVect::TheZeroVector() );
            }
        }
    }

    /**
     * \brief Scale Y by a (Y *= a)
     */
    inline
    void scale (RT a_a)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            IsDefined(),
            "WarpXSolverVec::scale() called on undefined WarpXSolverVec");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->mult(a_a, 0, 1);
                m_array_vec[lev][1]->mult(a_a, 0, 1);
                m_array_vec[lev][2]->mult(a_a, 0, 1);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->mult(a_a, 0, 1);
            }
        }
    }

    inline
    void zero () { setVal(0.0); }

    inline
    void setVal ( const RT  a_val )
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            IsDefined(),
            "WarpXSolverVec::setVal() called on undefined WarpXSolverVec");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->setVal(a_val);
                m_array_vec[lev][1]->setVal(a_val);
                m_array_vec[lev][2]->setVal(a_val);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->setVal(a_val);
            }
        }
    }

    inline
    void assertIsDefined( const WarpXSolverVec&  a_solver_vec ) const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_solver_vec.IsDefined(),
            "WarpXSolverVec::function(X) called with undefined WarpXSolverVec X");
    }

    inline
    void assertSameType( const WarpXSolverVec&  a_solver_vec ) const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_solver_vec.getArrayVecType()==m_array_type &&
            a_solver_vec.getScalarVecType()==m_scalar_type,
            "WarpXSolverVec::function(X) called with WarpXSolverVec X of different type");
    }

    [[nodiscard]] inline RT norm2 () const
    {
        auto const norm = dotProduct(*this);
        return std::sqrt(norm);
    }

    [[nodiscard]] const ablastr::fields::MultiLevelVectorField& getArrayVec() const {return m_array_vec;}
    ablastr::fields::MultiLevelVectorField& getArrayVec() {return m_array_vec;}

    [[nodiscard]] const ablastr::fields::MultiLevelScalarField& getScalarVec() const {return m_scalar_vec;}
    ablastr::fields::MultiLevelScalarField& getScalarVec() {return m_scalar_vec;}

    // solver vector types are type warpx::fields::FieldType
    [[nodiscard]] warpx::fields::FieldType getArrayVecType () const { return m_array_type; }
    [[nodiscard]] warpx::fields::FieldType getScalarVecType () const { return m_scalar_type; }

    // solver vector type names
    [[nodiscard]] std::string getVectorType () const { return m_vector_type_name; }
    [[nodiscard]] std::string getScalarType () const { return m_scalar_type_name; }


private:

    bool m_is_defined = false;

    ablastr::fields::MultiLevelVectorField m_array_vec;
    ablastr::fields::MultiLevelScalarField m_scalar_vec;

    warpx::fields::FieldType m_array_type = warpx::fields::FieldType::None;
    warpx::fields::FieldType m_scalar_type = warpx::fields::FieldType::None;

    std::string m_vector_type_name = "none";
    std::string m_scalar_type_name = "none";

    static constexpr int m_ncomp = 1;
    int m_num_amr_levels = 1;

    inline static bool m_warpx_ptr_defined = false;
    inline static WarpX* m_WarpX = nullptr;

};

#endif

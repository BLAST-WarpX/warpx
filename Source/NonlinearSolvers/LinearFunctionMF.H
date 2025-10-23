/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef LinearFunctionMF_H_
#define LinearFunctionMF_H_

#include "LinearFunction.H"
#include "Utils/TextMsg.H"

#include <AMReX_Config.H>
#include <string>

/**
 * \brief This is a linear function class for computing the action of a
 *  Jacobian on a vector using a matrix-free finite-difference method.
 *  This class has all of the required functions to be used as the
 *  linear operator template parameter in AMReX_GMRES.
 */

template <class T, class Ops>
class LinearFunctionMF : public LinearFunction<T,Ops>
{
    public:

    using RT = typename T::value_type;

    LinearFunctionMF() = default;
    ~LinearFunctionMF() = default;

    // Default move and copy operations
    LinearFunctionMF(const LinearFunctionMF&) = default;
    LinearFunctionMF& operator=(const LinearFunctionMF&) = default;
    LinearFunctionMF(LinearFunctionMF&&) noexcept = default;
    LinearFunctionMF& operator=(LinearFunctionMF&&) noexcept = default;

    void apply ( T& a_dF, const T& a_dU ) override;

    inline
    void precond ( T& a_U, const T& a_X ) override
    {
        if (m_usePreCond) {
            a_U.zero();
            m_preCond->Apply(a_U, a_X);
        } else {
            a_U.Copy(a_X);
        }
    }

    inline
    void updatePreCondMat ( const T&  a_X ) override
    {
        if (m_usePreCond) { m_preCond->Update(a_X); }
    }

    T makeVecLHS () const override;
    T makeVecRHS () const override;

    inline
    bool isDefined() const { return m_is_defined;  }

    [[nodiscard]] inline
    bool usePreconditioner() const { return m_usePreCond;  }

    inline
    void setBaseSolution ( const T&  a_U )
    {
        m_Y0.Copy(a_U);
        m_normY0 = this->norm2(m_Y0);
    }

    inline
    void setBaseRHS ( const T&  a_R )
    {
        m_R0.Copy(a_R);
    }

    inline
    void setJFNKEps ( RT a_eps )
    {
        m_epsJFNK = a_eps;
    }

    inline
    void setIsLinear ( bool a_isLinear )
    {
        m_is_linear = a_isLinear;
    }

    inline
    void curTime ( RT a_time )
    {
        m_cur_time = a_time;
        if (m_usePreCond) { m_preCond->CurTime(a_time); }
    }

    inline
    void curTimeStep ( RT a_dt )
    {
        m_dt = a_dt;
        if (m_usePreCond) { m_preCond->CurTimeStep(a_dt); }
    }

    inline
    void printParams () const
    {
        if (m_pc_type != PreconditionerType::none) {
            m_preCond->printParameters();
        }
    }

    void define( const T&, Ops*, const PreconditionerType& ) override;

    inline
    PreconditionerType pcType () const override { return m_pc_type; }

    private:

    bool m_is_defined = false;
    bool m_is_linear = false;
    bool m_usePreCond = false;
    RT m_epsJFNK = RT(1.0e-6);
    RT m_normY0;
    RT m_cur_time, m_dt;

    PreconditionerType m_pc_type = PreconditionerType::none;

    T m_Z, m_Y0, m_R0, m_R;
    Ops* m_ops = nullptr;
    std::unique_ptr<Preconditioner<T,Ops>> m_preCond = nullptr;
};

template <class T, class Ops>
void LinearFunctionMF<T,Ops>::define ( const T& a_U,
                                         Ops* a_ops,
                                         const PreconditionerType& a_pc_type )
{
    BL_PROFILE("LinearFunctionMF::::define()");

    m_ops = a_ops;

    m_usePreCond = (a_pc_type != PreconditionerType::none);
    if (m_usePreCond) {
        m_pc_type = a_pc_type;
        m_preCond->Define(a_U, a_ops);
    }

    m_is_defined = true;
}

template <class T, class Ops>
auto LinearFunctionMF<T,Ops>::makeVecRHS () const -> T
{
    BL_PROFILE("LinearFunctionMF::::makeVecRHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
auto LinearFunctionMF<T,Ops>::makeVecLHS () const -> T
{
    BL_PROFILE("LinearFunctionMF::::makeVecLHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
void LinearFunctionMF<T,Ops>::apply (T& a_dF, const T& a_dU)
{
    BL_PROFILE("LinearFunctionMF::apply()");
    using namespace amrex::literals;
    RT const normY = this->norm2(a_dU); // always 1 when called from GMRES

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        isDefined(),
        "JacobianFunction::apply() called on undefined JacobianFunction");

    if (normY < 1.0e-15) { a_dF.zero(); }
    else {
        m_ops->ComputeRHS(a_dF, a_dU, m_cur_time, -1, true );
    }

}

#endif

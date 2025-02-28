/* Copyright 2024 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PRECONDITIONER_H_
#define WARPX_PRECONDITIONER_H_

#include <AMReX_Enum.H>

/**
 * \brief Types for preconditioners for field solvers
 */
AMREX_ENUM(PreconditionerType,
    pc_curl_curl_mlmg,
    pc_jacobi,
    none
);

/**
 * \brief Base class for preconditioners
 *
 *   This class is templated on a solution-type class T and an operator class Ops.
 *
 *   The Ops class must have the following function:
 *     (this will depend on the specific preconditioners inheriting from this class)
 *
 *   The T class must have the following functions:
 *     (this will depend on the specific preconditioners inheriting from this class)
 */

template <class T, class Ops>
class Preconditioner
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        Preconditioner () = default;

        /**
         * \brief Default destructor
         */
        virtual ~Preconditioner () = default;

        // Default move and copy operations
        Preconditioner(const Preconditioner&) = default;
        Preconditioner& operator=(const Preconditioner&) = default;
        Preconditioner(Preconditioner&&) noexcept = default;
        Preconditioner& operator=(Preconditioner&&) noexcept = default;

        /**
         * \brief Define the preconditioner
         */
        virtual void Define (const T&, Ops* const) = 0;

        /**
         * \brief Update the preconditioner
         */
        virtual void Update ( const T& a_U ) = 0;

        /**
         * \brief Apply (solve) the preconditioner given a RHS
         *
         *  Given a right-hand-side b, solve:
         *      A x = b
         *  where A is a linear operator.
         */
        virtual void Apply (T& a_x, const T& a_b) = 0;

        /**
         * \brief Check if the nonlinear solver has been defined.
         */
        [[nodiscard]] virtual bool IsDefined () const = 0;

        /**
         * \brief Print parameters
         */
        virtual void printParameters() const { }

        /**
         * \brief Set the current time.
         */
        inline void CurTime (const RT a_time) { m_time = a_time; }

        /**
         * \brief Set the current time step size.
         */
        inline void CurTimeStep (const RT a_dt) { m_dt = a_dt; }

    protected:

        RT m_time = 0.0;
        RT m_dt = 0.0;

    private:

};

#endif

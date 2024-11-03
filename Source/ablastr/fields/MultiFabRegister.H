/* Copyright 2024 The ABLASTR Community
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Axel Huebl
 */
#ifndef ABLASTR_FIELDS_MF_REGISTER_H
#define ABLASTR_FIELDS_MF_REGISTER_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Enum.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>


namespace
{
    // type trait helpers in lieu of an amrex::is_amrex_enum
    template <typename, typename = std::void_t<>>
    struct is_castable_to_string : std::false_type {};

    template <typename T>
    struct is_castable_to_string<T, std::void_t<decltype(static_cast<std::string>(std::declval<T>()))>> : std::true_type {};

    /** helper to either cast a string/char array to string to query an AMREX_ENUM */
    template<typename T>
    std::string getExtractedName (T name)
    {
        if constexpr(is_castable_to_string<T>())
        {
            // already a unique string key
            return std::string(name);
        } else
        {
            // user-defined AMREX_ENUM or compile error
            return amrex::getEnumNameString(name);
        }
    }
}

namespace ablastr::fields
{
    /** Components (base vector directions) of vector/tensor fields.
     *
     * Because of different staggering, the components of vector/tensor fields are stored
     * in separate (i)MultiFab.
     *
     * @todo: synchronize with AMReX "enum class Direction"
     */
    struct Direction
    {
        int dir = 0;

        bool operator<(const Direction& other) const
        {
            return other.dir < this->dir;
        }

        /* TODO: just temporary int compatibility */
        operator int() const { return dir; }
    };

    /** A scalar field (a MultiFab)
     *
     * Note: might still have components, e.g., for copies at different times.
     */
    using ScalarField = amrex::MultiFab*;

    /** A read-only scalar field (a MultiFab)
     *
     * Note: might still have components, e.g., for copies at different times.
     */
    using ConstScalarField = amrex::MultiFab const *;

    /** A vector field of three MultiFab
     */
    //using VectorField = ablastr::utils::ConstMap<Direction, amrex::MultiFab *>;
    using VectorField = std::array<amrex::MultiFab *, 3>;

    /** A read-only vector field of three MultiFab
     */
    //using VectorField = ablastr::utils::ConstMap<Direction, amrex::MultiFab const *>;
    using ConstVectorField = std::array<amrex::MultiFab const *, 3>;

    /** A multi-level scalar field
     */
    using MultiLevelScalarField = amrex::Vector<ScalarField>;

    /** A read-only multi-level scalar field
     */
    using ConstMultiLevelScalarField = amrex::Vector<ConstScalarField>;

    /** A multi-level vector field
    */
    using MultiLevelVectorField = amrex::Vector<VectorField>;

    /** A read-only multi-level vector field
    */
    using ConstMultiLevelVectorField = amrex::Vector<ConstVectorField>;

    /** A class to control the lifetime and properties of a MultiFab (field).
     *
     * This class is used to own the lifetime of an amrex::MultiFab and to store
     * associated information around it regarding unique naming, scalar/vector/tensor
     * properties, aliasing, load balancing, etc.
     */
    struct MultiFabOwner
    {
        // TODO: also add iMultiFab via std::variant

        /** owned (i)MultiFab */
        amrex::MultiFab m_mf;

        /** Components (base vector directions) of this MultiFab */
        std::optional<Direction> m_dir = std::nullopt;

        /** the MR level of this (i)MultiFab */
        int m_level = 0;

        /** remake distribution map on load balance, @see amrex::AmrCore::RemakeLevel */
        bool m_remake = true;

        /** redistribute on @see amrex::AmrCore::RemakeLevel */
        bool m_redistribute_on_remake = true;

        /** if m_mf is a non-owning alias, this string tracks the name of the owner */
        std::string m_owner;

        /** Is this part of a vector/tensor? */
        AMREX_INLINE
        bool
        is_vector () const { return m_dir.has_value(); }

        /** Is this an alias MultiFab?
         *
         * If yes, that means we do not own the memory.
         */
        AMREX_INLINE
        bool
        is_alias () const { return !m_owner.empty(); }
    };

    /** This is a register of fields aka amrex::MultiFabs.
     *
     * This is owned by a simulation instance. All used fields should be registered here.
     * Internally, this contains @see MultiFabOwner values.
     */
    struct MultiFabRegister
    {
        // Avoid accidental copies when passing to member functions
        MultiFabRegister() = default;
        MultiFabRegister(MultiFabRegister const &) = delete;
        MultiFabRegister(MultiFabRegister&&) = delete;
        MultiFabRegister& operator=(MultiFabRegister const &) = delete;
        MultiFabRegister& operator=(MultiFabRegister&&) = delete;
        ~MultiFabRegister() = default;

        /** Allocate and optionally initialize a MultiFab (field)
         *
         * This registers a new MultiFab under a unique name, allocates it and
         * optionally assigns it an initial value.
         *
         * @param name a unique name for this field
         * @param level the MR level to represent
         * @param ba the list of boxes to cover the field
         * @param dm the distribution mapping for load balancing with MPI
         * @param ncomp the number of components of the field (all with the same staggering)
         * @param ngrow the number of guard (ghost, halo) cells
         * @param initial_value the optional initial value
         * @param remake follow the default domain decomposition of the simulation
         * @param redistribute_on_remake redistribute on @see amrex::AmrCore::RemakeLevel
         * @return pointer to newly allocated MultiFab
         */
        template<typename T>
        amrex::MultiFab*
        alloc_init (
            T name,
            int level,
            amrex::BoxArray const & ba,
            amrex::DistributionMapping const & dm,
            int ncomp,
            amrex::IntVect const & ngrow,
            std::optional<amrex::Real const> initial_value = std::nullopt,
            bool remake = true,
            bool redistribute_on_remake = true
        )
        {
            return internal_alloc_init(
                getExtractedName(name),
                level,
                ba,
                dm,
                ncomp,
                ngrow,
                initial_value,
                remake,
                redistribute_on_remake
            );
        }

        /** Allocate and optionally initialize a MultiFab (field)
         *
         * This registers a new MultiFab under a unique name, allocates it and
         * optionally assigns it an initial value.
         *
         * @param name a unique name for this field
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level to represent
         * @param ba the list of boxes to cover the field
         * @param dm the distribution mapping for load balancing with MPI
         * @param ncomp the number of components of the field (all with the same staggering)
         * @param ngrow the number of guard (ghost, halo) cells
         * @param initial_value the optional initial value
         * @param remake follow the default domain decomposition of the simulation
         * @param redistribute_on_remake redistribute on @see amrex::AmrCore::RemakeLevel
         * @return pointer to newly allocated MultiFab
         */
        template<typename T>
        amrex::MultiFab*
        alloc_init (
            T name,
            Direction dir,
            int level,
            amrex::BoxArray const & ba,
            amrex::DistributionMapping const & dm,
            int ncomp,
            amrex::IntVect const & ngrow,
            std::optional<amrex::Real const> initial_value = std::nullopt,
            bool remake = true,
            bool redistribute_on_remake = true
        )
        {
            return internal_alloc_init(
                getExtractedName(name),
                dir,
                level,
                ba,
                dm,
                ncomp,
                ngrow,
                initial_value,
                remake,
                redistribute_on_remake
            );
        }

        /** Create an alias of a MultiFab (field)
         *
         * Registers a new name for an existing MultiFab (field) and optionally assigning
         * a value.
         *
         * @param new_name new name
         * @param alias_name owner name to alias
         * @param level the MR level to represent
         * @param initial_value the optional value to assign
         * @return the newly aliased MultiFab
         */
        template<typename N, typename A>
        amrex::MultiFab*
        alias_init (
            N new_name,
            A alias_name,
            int level,
            std::optional<amrex::Real const> initial_value = std::nullopt
        )
        {
            return internal_alias_init(
                getExtractedName(new_name),
                getExtractedName(alias_name),
                level,
                initial_value
            );
        }

        /** Create an alias of a MultiFab (field)
         *
         * Registers a new name for an existing MultiFab (field) and optionally assigning
         * a value.
         *
         * @param new_name new name
         * @param alias_name owner name to alias
         * @param dir the field component for vector fields ("direction" of the unit vector) both in the alias and aliased
         * @param level the MR level to represent
         * @param initial_value the optional value to assign
         * @return the newly aliased MultiFab
         */
        template<typename N, typename A>
        amrex::MultiFab*
        alias_init (
            N new_name,
            A alias_name,
            Direction dir,
            int level,
            std::optional<amrex::Real const> initial_value = std::nullopt
        )
        {
            return internal_alias_init(
                getExtractedName(new_name),
                getExtractedName(alias_name),
                dir,
                level,
                initial_value
            );
        }

        /** Check if a scalar MultiFab (field) is registered.
         *
         * @param name the name to check if registered
         * @param level the MR level to check
         * @return true if contained, otherwise false
         */
        template<typename T>
        [[nodiscard]] bool
        has (
            T name,
            int level
        ) const
        {
            return internal_has(
                getExtractedName(name),
                level
            );
        }

        /** Check if a MultiFab that is part of a vector/tensor field is registered.
         *
         * @param name the name to check if registered
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level to check
         * @return true if contained, otherwise false
         */
        template<typename T>
        [[nodiscard]] bool
        has (
            T name,
            Direction dir,
            int level
        ) const
        {
            return internal_has(
                getExtractedName(name),
                dir,
                level
            );
        }

        /** Check if a MultiFab vector field is registered.
         *
         * @param name the name to check if registered
         * @param level the MR level to check
         * @return true if contained, otherwise false
         */
        template<typename T>
        [[nodiscard]] bool
        has_vector (
            T name,
            int level
        ) const
        {
            return internal_has_vector(
                getExtractedName(name),
                level
            );
        }

        /** Return a scalar MultiFab (field).
         *
         * This throws a runtime error if the requested field is not present.
         *
         * @param name the name of the field
         * @param level the MR level
         * @return a non-owning pointer to the MultiFab (field)
         */
        template<typename T>
        [[nodiscard]] amrex::MultiFab*
        get (
            T name,
            int level
        )
        {
            return internal_get(
                getExtractedName(name),
                level
            );
        }

        /** Return a MultiFab that is part of a vector/tensor field.
         *
         * This throws a runtime error if the requested field is not present.
         *
         * @param name the name of the field
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level
         * @return a non-owning pointer to the MultiFab (field)
         */
        template<typename T>
        [[nodiscard]] amrex::MultiFab*
        get (
            T name,
            Direction dir,
            int level
        )
        {
            return internal_get(
                getExtractedName(name),
                dir,
                level
            );
        }

        /** Return a scalar MultiFab (field).
         *
         * This throws a runtime error if the requested field is not present.
         *
         * @param name the name of the field
         * @param level the MR level
         * @return a non-owning pointer to the MultiFab (field)
         */
        template<typename T>
        [[nodiscard]] amrex::MultiFab const *
        get (
            T name,
            int level
        ) const
        {
            return internal_get(
                getExtractedName(name),
                level
            );
        }

        /** Return a MultiFab that is part of a vector/tensor field.
         *
         * This throws a runtime error if the requested field is not present.
         *
         * @param name the name of the field
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level
         * @return a non-owning pointer to the MultiFab (field)
         */
        template<typename T>
        [[nodiscard]] amrex::MultiFab const *
        get (
            T name,
            Direction dir,
            int level
        ) const
        {
            return internal_get(
                getExtractedName(name),
                dir,
                level
            );
        }

        /** Return the MultiFab of a scalar field on all MR levels.
         *
         * This throws a runtime error if the requested field is not present.
         *
         * @param name the name of the field
         * @param finest_level the highest MR level to return
         * @return non-owning pointers to the MultiFab (field) on all levels
         */
        //@{
        template<typename T>
        [[nodiscard]] MultiLevelScalarField
        get_mr_levels (
            T name,
            int finest_level
        )
        {
            return internal_get_mr_levels(
                getExtractedName(name),
                finest_level
            );
        }
        template<typename T>
        [[nodiscard]] ConstMultiLevelScalarField
        get_mr_levels (
            T name,
            int finest_level
        ) const
        {
            return internal_get_mr_levels(
                getExtractedName(name),
                finest_level
            );
        }
        //@}

        /** title
         *
         * Same as get above, but returns all levels for a name.
         *
         * @param name the name of the field
         * @param level the MR level
         * @return non-owning pointers to all components of a vector field
         */
        //@{
        template<typename T>
        [[nodiscard]] VectorField
        get_alldirs  (
            T name,
            int level
        )
        {
            return internal_get_alldirs(
                getExtractedName(name),
                level
            );
        }
        template<typename T>
        [[nodiscard]] ConstVectorField
        get_alldirs  (
            T name,
            int level
        ) const
        {
            return internal_get_alldirs(
                getExtractedName(name),
                level
            );
        }
        //@}

        /** Return a vector field on all MR levels.
         *
         * Out loop: MR levels.
         * Inner loop: directions (components).
         *
         * @param name the name of the field
         * @param finest_level the highest MR level to return
         * @return non-owning pointers to all components of a vector field on all MR levels
         */
        //@{
        template<typename T>
        [[nodiscard]] MultiLevelVectorField
        get_mr_levels_alldirs (
            T name,
            int finest_level
        )
        {
            return internal_get_mr_levels_alldirs(
                getExtractedName(name),
                finest_level
            );
        }
        template<typename T>
        [[nodiscard]] ConstMultiLevelVectorField
        get_mr_levels_alldirs (
            T name,
            int finest_level
        ) const
        {
            return internal_get_mr_levels_alldirs(
                getExtractedName(name),
                finest_level
            );
        }
        //@}

        /** List the internal names of all registered fields.
         *
         * @return all currently allocated and registered fields
         */
        [[nodiscard]] std::vector<std::string>
        list () const;

        /** Deallocate and remove a scalar field.
         *
         * @param name the name of the field
         * @param level the MR level
         */
        template<typename T>
        void
        erase (
            T name,
            int level
        )
        {
            internal_erase(getExtractedName(name), level);
        }

        /** Deallocate and remove a vector field component.
         *
         * @param name the name of the field
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level
         */
        template<typename T>
        void
        erase (
            T name,
            Direction dir,
            int level
        )
        {
            internal_erase(getExtractedName(name), dir, level);
        }

        /** Erase all MultiFabs on a specific MR level.
         *
         * Calls @see erase for all MultiFabs on a specific level.
         *
         * @param level the MR level to erase all MultiFabs from
         */
        void
        clear_level (
            int level
        );

        /** Remake all (i)MultiFab with a new distribution mapping.
         *
         * If redistribute is true, we also copy from the old data into the new.
         *
         * @param level the MR level to erase all MultiFabs from
         * @param new_dm new distribution mapping
         */
        void
        remake_level (
            int other_level,
            amrex::DistributionMapping const & new_dm
        );

        /** Create the register name of scalar field and MR level
         *
         * @param name the name of the field
         * @param level the MR level
         * @return internal name of the field in the register
         */
        [[nodiscard]] std::string
        mf_name (
            std::string name,
            int level
        ) const;

        /** Create the register name of vector field, component direction and MR level
         *
         * @param name the name of the field
         * @param dir the field component for vector fields ("direction" of the unit vector)
         * @param level the MR level
         * @return internal name of the field in the register
         */
        [[nodiscard]] std::string
        mf_name (
            std::string name,
            Direction dir,
            int level
        ) const;

        /** Temporary test function for legacy Python bindings */
        [[nodiscard]] bool
        internal_has (
            std::string const & internal_name
        );
        [[nodiscard]] amrex::MultiFab *
        internal_get (
            std::string const & internal_name
        );

    private:

        [[nodiscard]] amrex::MultiFab const *
        internal_get (
            std::string const & internal_name
        ) const;

        amrex::MultiFab*
        internal_alloc_init (
            std::string const & name,
            int level,
            amrex::BoxArray const & ba,
            amrex::DistributionMapping const & dm,
            int ncomp,
            amrex::IntVect const & ngrow,
            std::optional<amrex::Real const> initial_value = std::nullopt,
            bool remake = true,
            bool redistribute_on_remake = true
        );
        amrex::MultiFab*
        internal_alloc_init (
            std::string const & name,
            Direction dir,
            int level,
            amrex::BoxArray const & ba,
            amrex::DistributionMapping const & dm,
            int ncomp,
            amrex::IntVect const & ngrow,
            std::optional<amrex::Real const> initial_value = std::nullopt,
            bool remake = true,
            bool redistribute_on_remake = true
        );

        amrex::MultiFab*
        internal_alias_init (
            std::string const & new_name,
            std::string const & alias_name,
            int level,
            std::optional<amrex::Real const> initial_value = std::nullopt
        );
        amrex::MultiFab*
        internal_alias_init (
            std::string const & new_name,
            std::string const & alias_name,
            Direction dir,
            int level,
            std::optional<amrex::Real const> initial_value = std::nullopt
        );

        [[nodiscard]] bool
        internal_has (
            std::string const & name,
            int level
        ) const;
        [[nodiscard]] bool
        internal_has (
            std::string const & name,
            Direction dir,
            int level
        ) const;
        [[nodiscard]] bool
        internal_has_vector (
            std::string const & name,
            int level
        ) const;

        [[nodiscard]] amrex::MultiFab *
        internal_get (
            std::string const & name,
            int level
        );
        [[nodiscard]] amrex::MultiFab const *
        internal_get (
            std::string const & name,
            int level
        ) const;
        [[nodiscard]] amrex::MultiFab *
        internal_get (
            std::string const & name,
            Direction dir,
            int level
        );
        [[nodiscard]] amrex::MultiFab const *
        internal_get (
            std::string const & name,
            Direction dir,
            int level
        ) const;
        [[nodiscard]] MultiLevelScalarField
        internal_get_mr_levels (
            std::string const & name,
            int finest_level
        );
        [[nodiscard]] ConstMultiLevelScalarField
        internal_get_mr_levels (
            std::string const & name,
            int finest_level
        ) const;
        [[nodiscard]] VectorField
        internal_get_alldirs (
            std::string const & name,
            int level
        );
        [[nodiscard]] ConstVectorField
        internal_get_alldirs (
            std::string const & name,
            int level
        ) const;
        [[nodiscard]] MultiLevelVectorField
        internal_get_mr_levels_alldirs (
            std::string const & name,
            int finest_level
        );
        [[nodiscard]] ConstMultiLevelVectorField
        internal_get_mr_levels_alldirs (
            std::string const & name,
            int finest_level
        ) const;

        void
        internal_erase (
            std::string const & name,
            int level
        );
        void
        internal_erase (
            std::string const & name,
            Direction dir,
            int level
        );

        /** data storage: ownership and lifetime control */
        std::map<
            std::string,
            MultiFabOwner
        > m_mf_register;

        /** the three directions of a vector field */
        std::vector<Direction> m_all_dirs = {Direction{0}, Direction{1}, Direction{2}};
    };

    /** Little temporary helper function to pass temporary MultiFabs as VectorField.
     *
     * @return pointers to externally managed vector field components (3 MultiFab)
     */
    VectorField
    a2m (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const & old_vectorfield
    );

} // namespace ablastr::fields

#endif  // ABLASTR_FIELDS_MF_REGISTER_H

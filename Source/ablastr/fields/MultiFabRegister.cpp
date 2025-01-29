/* Copyright 2024 The ABLASTR Community
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Axel Huebl
 */
#include "MultiFabRegister.H"

#include <AMReX_MakeType.H>

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace ablastr::fields
{
    amrex::MultiFab*
    MultiFabRegister::internal_alloc_init (
        std::string const & name,
        int level,
        amrex::BoxArray const & ba,
        amrex::DistributionMapping const & dm,
        int ncomp,
        amrex::IntVect const & ngrow,
        std::optional<amrex::Real const> initial_value,
        bool remake,
        bool redistribute_on_remake
    )
    {
        // checks
        if (has(name, level)) {
            throw std::runtime_error("MultiFabRegister::alloc_init failed because " + name + " already exists.");
        }

        // fully qualified name
        std::string const internal_name = mf_name(name, level);

        // allocate
        const auto tag = amrex::MFInfo().SetTag(internal_name);
        auto [it, success] = m_mf_register.emplace(
            internal_name,
            MultiFabOwner{
                {ba, dm, ncomp, ngrow, tag},
                std::nullopt,  // scalar: no direction
                level,
                remake,
                redistribute_on_remake,
                ""   // we own the memory
            }
        );
        if (!success) {
            throw std::runtime_error("MultiFabRegister::alloc_init failed for " + internal_name);
        }

        // a shorthand alias for the code below
        amrex::MultiFab & mf = it->second.m_mf;

        // initialize with value
        if (initial_value) {
            mf.setVal(*initial_value);
        }

        return &mf;
    }

    amrex::MultiFab*
    MultiFabRegister::internal_alloc_init (
        std::string const & name,
        Direction dir,
        int level,
        amrex::BoxArray const & ba,
        amrex::DistributionMapping const & dm,
        int ncomp,
        amrex::IntVect const & ngrow,
        std::optional<amrex::Real const> initial_value,
        bool remake,
        bool redistribute_on_remake
    )
    {
        // checks
        if (has(name, dir, level)) {
            throw std::runtime_error(
                "MultiFabRegister::alloc_init failed because " +
                mf_name(name, dir, level) +
                " already exists."
            );
        }

        // fully qualified name
        std::string const internal_name = mf_name(name, dir, level);

        // allocate
        const auto tag = amrex::MFInfo().SetTag(internal_name);
        auto [it, success] = m_mf_register.emplace(
            internal_name,
            MultiFabOwner{
                {ba, dm, ncomp, ngrow, tag},
                dir,
                level,
                remake,
                redistribute_on_remake,
                ""   // we own the memory
            }
        );
        if (!success) {
            throw std::runtime_error("MultiFabRegister::alloc_init failed for " + internal_name);
        }

        // a shorthand alias for the code below
        amrex::MultiFab & mf = it->second.m_mf;

        // initialize with value
        if (initial_value) {
            mf.setVal(*initial_value);
        }

        return &mf;
    }

    amrex::MultiFab*
    MultiFabRegister::internal_alias_init (
        std::string const & new_name,
        std::string const & alias_name,
        int level,
        std::optional<amrex::Real const> initial_value
    )
    {
        // checks
        if (has(new_name, level)) {
            throw std::runtime_error(
                "MultiFabRegister::alias_init failed because " +
                mf_name(new_name, level) +
                " already exists."
            );
        }
        if (!has(alias_name, level)) {
            throw std::runtime_error(
                    "MultiFabRegister::alias_init failed because " +
                    mf_name(alias_name, level) +
                    " does not exist."
            );
        }

        // fully qualified name
        std::string const internal_new_name = mf_name(new_name, level);
        std::string const internal_alias_name = mf_name(alias_name, level);

        MultiFabOwner & alias = m_mf_register[internal_alias_name];
        amrex::MultiFab & mf_alias = alias.m_mf;

        // allocate
        auto [it, success] = m_mf_register.emplace(
                internal_new_name,
            MultiFabOwner{
                {mf_alias, amrex::make_alias, 0, mf_alias.nComp()},
                std::nullopt,  // scalar: no direction
                level,
                alias.m_remake,
                alias.m_redistribute_on_remake,
                internal_alias_name
            }

        );
        if (!success) {
            throw std::runtime_error("MultiFabRegister::alias_init failed for " + internal_new_name);
        }

        // a shorthand alias for the code below
        amrex::MultiFab & mf = it->second.m_mf;

        // initialize with value
        if (initial_value) {
            mf.setVal(*initial_value);
        }

        return &mf;
    }

    amrex::MultiFab*
    MultiFabRegister::internal_alias_init (
            std::string const & new_name,
            std::string const & alias_name,
            Direction dir,
            int level,
            std::optional<amrex::Real const> initial_value
    )
    {
        // checks
        if (has(new_name, dir, level)) {
            throw std::runtime_error(
                "MultiFabRegister::alias_init failed because " +
                mf_name(new_name, dir, level) +
                " already exists."
            );
        }
        if (!has(alias_name, dir, level)) {
            throw std::runtime_error(
                "MultiFabRegister::alias_init failed because " +
                mf_name(alias_name, dir, level) +
                " does not exist."
            );
        }

        // fully qualified name
        std::string const internal_new_name = mf_name(new_name, dir, level);
        std::string const internal_alias_name = mf_name(alias_name, dir, level);

        MultiFabOwner & alias = m_mf_register[internal_alias_name];
        amrex::MultiFab & mf_alias = alias.m_mf;

        // allocate
        auto [it, success] = m_mf_register.emplace(
            internal_new_name,
            MultiFabOwner{
                {mf_alias, amrex::make_alias, 0, mf_alias.nComp()},
                dir,
                level,
                alias.m_remake,
                alias.m_redistribute_on_remake,
                internal_alias_name
            }
        );
        if (!success) {
            throw std::runtime_error("MultiFabRegister::alias_init failed for " + internal_new_name);
        }

        // a short-hand alias for the code below
        amrex::MultiFab & mf = it->second.m_mf;

        // initialize with value
        if (initial_value) {
            mf.setVal(*initial_value);
        }

        return &mf;
    }

    void
    MultiFabRegister::remake_level (
        int level,
        amrex::DistributionMapping const & new_dm
    )
    {
        // Owning MultiFabs
        for (auto & element : m_mf_register )
        {
            MultiFabOwner & mf_owner = element.second;

            // keep distribution map as it is?
            if (!mf_owner.m_remake) {
                continue;
            }

            // remake MultiFab with new distribution map
            if (mf_owner.m_level == level && !mf_owner.is_alias()) {
                const amrex::MultiFab & mf = mf_owner.m_mf;
                amrex::IntVect const & ng = mf.nGrowVect();
                const auto tag = amrex::MFInfo().SetTag(mf.tags()[0]);
                amrex::MultiFab new_mf(mf.boxArray(), new_dm, mf.nComp(), ng, tag);

                // copy data to new MultiFab: Only done for persistent data like E and B field, not for
                // temporary things like currents, etc.
                if (mf_owner.m_redistribute_on_remake) {
                    new_mf.Redistribute(mf, 0, 0, mf.nComp(), ng);
                }

                // replace old MultiFab with new one, deallocate old one
                mf_owner.m_mf = std::move(new_mf);
            }
        }

        // Aliases
        for (auto & element : m_mf_register )
        {
            MultiFabOwner & mf_owner = element.second;

            // keep distribution map as it is?
            if (!mf_owner.m_remake) {
                continue;
            }

            if (mf_owner.m_level == level && mf_owner.is_alias()) {
                const amrex::MultiFab & mf = m_mf_register[mf_owner.m_owner].m_mf;
                amrex::MultiFab new_mf(mf, amrex::make_alias, 0, mf.nComp());

                // no copy via Redistribute: the owner was already redistributed

                // replace old MultiFab with new one, deallocate old one
                mf_owner.m_mf = std::move(new_mf);
            }
        }
    }

    bool
    MultiFabRegister::internal_has (
        std::string const & name,
        int level
    ) const
    {
        std::string const internal_name = mf_name(name, level);

        return m_mf_register.count(internal_name) > 0;
    }

    bool
    MultiFabRegister::internal_has (
        std::string const & name,
        Direction dir,
        int level
    ) const
    {
        std::string const internal_name = mf_name(name, dir, level);

        return m_mf_register.count(internal_name) > 0;
    }

    bool
    MultiFabRegister::internal_has_vector (
        std::string const & name,
        int level
    ) const
    {
        unsigned long count = 0;
        for (Direction const & dir : m_all_dirs)
        {
            std::string const internal_name = mf_name(name, dir, level);
            count += m_mf_register.count(internal_name);
        }

        return count == 3;
    }

    bool
    MultiFabRegister::internal_has (
        std::string const & internal_name
    )
    {
        return m_mf_register.count(internal_name) > 0;
    }

    amrex::MultiFab*
    MultiFabRegister::internal_get (
        std::string const & internal_name
    )
    {
        if (m_mf_register.count(internal_name) == 0) {
            // FIXME: temporary, throw a std::runtime_error
            // throw std::runtime_error("MultiFabRegister::get name does not exist in register: " + key);
            return nullptr;
        }
        amrex::MultiFab & mf = m_mf_register.at(internal_name).m_mf;

        return &mf;
    }

    amrex::MultiFab const *
    MultiFabRegister::internal_get (
        std::string const & internal_name
    ) const
    {
        if (m_mf_register.count(internal_name) == 0) {
            // FIXME: temporary, throw a std::runtime_error
            // throw std::runtime_error("MultiFabRegister::get name does not exist in register: " + internal_name);
            return nullptr;
        }
        amrex::MultiFab const & mf = m_mf_register.at(internal_name).m_mf;

        return &mf;
    }

    amrex::MultiFab*
    MultiFabRegister::internal_get (
        std::string const & name,
        int level
    )
    {
        std::string const internal_name = mf_name(name, level);
        return internal_get(internal_name);
    }

    amrex::MultiFab*
    MultiFabRegister::internal_get (
        std::string const & name,
        Direction dir,
        int level
    )
    {
        std::string const internal_name = mf_name(name, dir, level);
        return internal_get(internal_name);
    }

    amrex::MultiFab const *
    MultiFabRegister::internal_get (
        std::string const & name,
        int level
    ) const
    {
        std::string const internal_name = mf_name(name, level);
        return internal_get(internal_name);
    }

    amrex::MultiFab const *
    MultiFabRegister::internal_get (
        std::string const & name,
        Direction dir,
        int level
    ) const
    {
        std::string const internal_name = mf_name(name, dir, level);
        return internal_get(internal_name);
    }

    MultiLevelScalarField
    MultiFabRegister::internal_get_mr_levels (
        std::string const & name,
        int finest_level
    )
    {
        MultiLevelScalarField field_on_level;
        field_on_level.reserve(finest_level+1);
        for (int lvl = 0; lvl <= finest_level; lvl++)
        {
            field_on_level.push_back(internal_get(name, lvl));
        }
        return field_on_level;
    }

    ConstMultiLevelScalarField
    MultiFabRegister::internal_get_mr_levels (
        std::string const & name,
        int finest_level
    ) const
    {
        ConstMultiLevelScalarField field_on_level;
        field_on_level.reserve(finest_level+1);
        for (int lvl = 0; lvl <= finest_level; lvl++)
        {
            field_on_level.push_back(internal_get(name, lvl));
        }
        return field_on_level;
    }

    VectorField
    MultiFabRegister::internal_get_alldirs (
        std::string const & name,
        int level
    )
    {
        // insert a new level
        VectorField vectorField;

        // insert components
        for (Direction const & dir : m_all_dirs)
        {
            vectorField[dir] = internal_get(name, dir, level);
        }
        return vectorField;
    }

    ConstVectorField
    MultiFabRegister::internal_get_alldirs (
        std::string const & name,
        int level
    ) const
    {
        // insert a new level
        ConstVectorField vectorField;

        // insert components
        for (Direction const & dir : m_all_dirs)
        {
            vectorField[dir] = internal_get(name, dir, level);
        }
        return vectorField;
    }

    MultiLevelVectorField
    MultiFabRegister::internal_get_mr_levels_alldirs (
        std::string const & name,
        int finest_level
    )
    {
        MultiLevelVectorField field_on_level;
        field_on_level.reserve(finest_level+1);

        for (int lvl = 0; lvl <= finest_level; lvl++)
        {
            // insert a new level
            field_on_level.push_back(VectorField{});

            // insert components
            for (Direction const & dir : m_all_dirs)
            {
                field_on_level[lvl][dir] = internal_get(name, dir, lvl);
            }
        }
        return field_on_level;
    }

    ConstMultiLevelVectorField
    MultiFabRegister::internal_get_mr_levels_alldirs (
        std::string const & name,
        int finest_level
    ) const
    {
        ConstMultiLevelVectorField field_on_level;
        field_on_level.reserve(finest_level+1);

        for (int lvl = 0; lvl <= finest_level; lvl++)
        {
            // insert a new level
            field_on_level.push_back(ConstVectorField{});

            // insert components
            for (Direction const & dir : m_all_dirs)
            {
                field_on_level[lvl][dir] = internal_get(name, dir, lvl);
            }
        }
        return field_on_level;
    }

    std::vector<std::string>
    MultiFabRegister::list () const
    {
        std::vector<std::string> names;
        names.reserve(m_mf_register.size());
        for (auto const & str : m_mf_register) { names.push_back(str.first); }

        return names;
    }

    void
    MultiFabRegister::internal_erase (
        std::string const & name,
        int level
    )
    {
        std::string const internal_name = mf_name(name, level);

        if (m_mf_register.count(internal_name) != 1) {
            throw std::runtime_error("MultiFabRegister::erase name does not exist in register: " + internal_name);
        }
        m_mf_register.erase(internal_name);
    }

    void
    MultiFabRegister::internal_erase (
        std::string const & name,
        Direction dir,
        int level
    )
    {
        std::string const internal_name = mf_name(name, dir, level);

        if (m_mf_register.count(internal_name) != 1) {
            throw std::runtime_error("MultiFabRegister::erase name does not exist in register: " + internal_name);
        }
        m_mf_register.erase(internal_name);
    }

    void
    MultiFabRegister::clear_level (
        int level
    )
    {
        // C++20: Replace with std::erase_if
        for (auto first = m_mf_register.begin(), last = m_mf_register.end(); first != last;)
        {
            if (first->second.m_level == level) {
                first = m_mf_register.erase(first);
            } else {
                ++first;
            }
        }
    }

    std::string
    MultiFabRegister::mf_name (
        std::string name,
        int level
    ) const
    {
        // Add the suffix "[level=level]"
        return name.append("[level=")
                .append(std::to_string(level))
                .append("]");
    }

    std::string
    MultiFabRegister::mf_name (
        std::string name,
        Direction dir,
        int level
    ) const
    {
        // Add the suffix for the direction [x] or [y] or [z]
        // note: since Cartesian is not correct for all our supported geometries,
        //       in the future we might want to break this to "[dir=0/1/2]".
        //       This will be a breaking change for (Python) users that rely on that string.
        constexpr int x_in_ascii = 120;
        std::string const component_name{char(x_in_ascii + dir.dir)};
        return mf_name(
            name
            .append("[")
            .append(component_name)
            .append("]"),
            level
        );
    }

    VectorField
    a2m (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const & old_vectorfield
    )
    {
        std::vector<Direction> const all_dirs = {Direction{0}, Direction{1}, Direction{2}};

        VectorField field_on_level;

        // insert components
        for (auto const dir : {0, 1, 2})
        {
            field_on_level[Direction{dir}] = old_vectorfield[dir].get();
        }
        return field_on_level;
    }
} // namespace ablastr::fields

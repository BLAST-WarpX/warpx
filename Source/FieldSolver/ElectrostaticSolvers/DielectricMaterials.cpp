/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "DielectricMaterials.H"

#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#ifdef AMREX_USE_EB
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_EBFabFactory.H>
#   if defined(WARPX_DIM_3D)
#       include <AMReX_EB_STL_utils.H>
#   endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

#ifdef AMREX_USE_EB
namespace
{
    class DielectricParserIF
        : public amrex::GPUable
    {
    public:
        explicit
        DielectricParserIF (amrex::ParserExecutor<3> const& parser)
            : m_parser(parser)
        {}

        AMREX_GPU_HOST_DEVICE
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept
        {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return m_parser(x, 0.0_rt, y);
#elif defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(x, y);
            return m_parser(0.0_rt, 0.0_rt, z);
#else
            return m_parser(x, y, z);
#endif
        }

        amrex::Real operator() (amrex::RealArray const& p) const noexcept
        {
            return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
        }

    private:
        amrex::ParserExecutor<3> m_parser;
    };
}
#endif

namespace
{
    void
    AssertPermittivityFunctionValues (
        amrex::MultiFab const& epsilon,
        std::string const& material_name)
    {
        bool const local = true;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            epsilon.is_finite(0, 1, epsilon.nGrowVect(), local),
            "Dielectric material '" + material_name +
            "' has non-finite values from permittivity_function(x,y,z,t). "
            "Relative permittivity must be finite and greater than or equal to 1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            epsilon.min(0, epsilon.nGrowVect().min(), local) >= 1.0_rt,
            "Dielectric material '" + material_name +
            "' has values less than 1 from permittivity_function(x,y,z,t). "
            "Relative permittivity must be finite and greater than or equal to 1.");
    }

    void
    MergeSignedDistanceAndMaskOnGrid (
        amrex::MultiFab& sdf,
        amrex::iMultiFab& material_id,
        amrex::MultiFab const& object_sdf,
        int const object_id)
    {
        bool const first_object = (object_id == 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(sdf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const& s = sdf.array(mfi);
            amrex::Array4<amrex::Real const> const& os = object_sdf.const_array(mfi);
            amrex::Array4<int> const& m = material_id.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real const d = os(i, j, k);
                if (first_object) {
                    s(i, j, k) = d;
                } else {
                    s(i, j, k) = amrex::min(s(i, j, k), d);
                }
                if (d <= 0.0_rt) {
                    m(i, j, k) = object_id;
                }
            });
        }
    }

    void
    SetImplicitConstantPermittivity (
        amrex::MultiFab& epsilon,
        amrex::Geometry const& geom,
        amrex::ParserExecutor<3> const shape_exec,
        amrex::Real const permittivity)
    {
        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(epsilon, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const& eps = epsilon.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#if defined(WARPX_DIM_3D)
                amrex::Real const x = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::Real const y = problo[1] + (j + 0.5_rt) * dx[1];
                amrex::Real const z = problo[2] + (k + 0.5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Real const x = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = problo[1] + (j + 0.5_rt) * dx[1];
                amrex::ignore_unused(k);
#else
                amrex::Real const x = 0.0_rt;
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::ignore_unused(j, k);
#endif
                if (shape_exec(x, y, z) >= 0.0_rt) {
                    eps(i, j, k) = permittivity;
                }
            });
        }
    }

    void
    SetImplicitFunctionPermittivity (
        amrex::MultiFab& epsilon,
        amrex::Geometry const& geom,
        amrex::ParserExecutor<3> const shape_exec,
        amrex::ParserExecutor<4> const permittivity_exec,
        amrex::Real const time)
    {
        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(epsilon, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const& eps = epsilon.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#if defined(WARPX_DIM_3D)
                amrex::Real const x = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::Real const y = problo[1] + (j + 0.5_rt) * dx[1];
                amrex::Real const z = problo[2] + (k + 0.5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Real const x = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = problo[1] + (j + 0.5_rt) * dx[1];
                amrex::ignore_unused(k);
#else
                amrex::Real const x = 0.0_rt;
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::ignore_unused(j, k);
#endif
                if (shape_exec(x, y, z) >= 0.0_rt) {
                    eps(i, j, k) = permittivity_exec(x, y, z, time);
                }
            });
        }
    }

    void
    SetSTLConstantPermittivity (
        amrex::MultiFab& epsilon,
        amrex::MultiFab const& stl_sdf,
        amrex::Real const permittivity)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(epsilon, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const& eps = epsilon.array(mfi);
            amrex::Array4<amrex::Real const> const& d = stl_sdf.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (d(i, j, k) <= 0.0_rt) {
                    eps(i, j, k) = permittivity;
                }
            });
        }
    }

    void
    SetSTLFunctionPermittivity (
        amrex::MultiFab& epsilon,
        amrex::MultiFab const& stl_sdf,
        amrex::Geometry const& geom,
        amrex::ParserExecutor<4> const permittivity_exec,
        amrex::Real const time)
    {
#if defined(WARPX_DIM_3D)
        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();
#else
        amrex::ignore_unused(geom);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(epsilon, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const& eps = epsilon.array(mfi);
            amrex::Array4<amrex::Real const> const& d = stl_sdf.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#if defined(WARPX_DIM_3D)
                amrex::Real const x = problo[0] + (i + 0.5_rt) * dx[0];
                amrex::Real const y = problo[1] + (j + 0.5_rt) * dx[1];
                amrex::Real const z = problo[2] + (k + 0.5_rt) * dx[2];
#else
                amrex::Real const x = 0.0_rt;
                amrex::Real const y = 0.0_rt;
                amrex::Real const z = 0.0_rt;
                amrex::ignore_unused(i, j, k);
#endif
                if (d(i, j, k) <= 0.0_rt) {
                    eps(i, j, k) = permittivity_exec(x, y, z, time);
                }
            });
        }
    }

}

DielectricMaterials::DielectricMaterials (int nlevs_max)
    : m_material_id(nlevs_max)
{}

void
DielectricMaterials::ReadParameters ()
{
    amrex::ParmParse const pp_warpx("warpx");
    std::string old_epsilon_function;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_warpx.query("epsilon_function(x,y,z)", old_epsilon_function),
        "warpx.epsilon_function(x,y,z) has been replaced by the dielectric "
        "material interface. Use dielectrics.names and "
        "<dielectric_name>.permittivity or "
        "<dielectric_name>.permittivity_function(x,y,z,t).");

    amrex::ParmParse const pp_dielectrics("dielectrics");
    std::vector<std::string> names;
    pp_dielectrics.queryarr("names", names);

    if (names.empty()) {
        return;
    }

#if !defined(AMREX_USE_EB)
    WARPX_ABORT_WITH_MESSAGE(
        "Dielectric materials require an AMReX EB build because AMReX EB "
        "classes are used to build the dielectric signed-distance function.");
#endif

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "Dielectric materials are currently implemented only for 2D XZ, RZ, and 3D.");
#endif

    int default_stl_use_bvh = 1;
    pp_dielectrics.query("stl_use_bvh", default_stl_use_bvh);

    m_materials.clear();
    m_has_time_dependent_permittivity = false;
    m_materials.reserve(names.size());
    for (auto const& name : names) {
        ReadMaterial(name, default_stl_use_bvh);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame ||
        WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic,
        "dielectrics.names is currently supported only with "
        "warpx.do_electrostatic = labframe or labframe-electromagnetostatic.");
}

void
DielectricMaterials::ReadMaterial (std::string const& name, int default_stl_use_bvh)
{
    amrex::ParmParse const pp_material(name);

    Material material;
    material.name = name;
    material.stl_use_bvh = default_stl_use_bvh;

    bool const has_implicit_function =
        pp_material.query("implicit_function", material.implicit_function);
    bool const has_stl_file =
        pp_material.query("stl_file", material.stl_file);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        has_implicit_function != has_stl_file,
        "Dielectric material '" + name + "' must specify exactly one of "
        "implicit_function or stl_file.");

    if (has_implicit_function) {
        material.shape_type = ShapeType::ImplicitFunction;
    } else {
        material.shape_type = ShapeType::STL;
#if !defined(WARPX_DIM_3D)
        WARPX_ABORT_WITH_MESSAGE(
            "Dielectric material '" + name + "' uses stl_file, but STL "
            "dielectrics are currently available only in 3D.");
#endif
        utils::parser::queryWithParser(pp_material, "stl_scale", material.stl_scale);
        pp_material.query("stl_reverse_normal", material.stl_reverse_normal);
        pp_material.query("stl_use_bvh", material.stl_use_bvh);

        std::vector<amrex::Real> center{
            material.stl_center[0], material.stl_center[1], material.stl_center[2]};
        utils::parser::queryArrWithParser(pp_material, "stl_center", center);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            center.size() == 3,
            "Dielectric material '" + name + "' stl_center must have three values.");
        material.stl_center = {{center[0], center[1], center[2]}};
    }

    bool const has_constant =
        utils::parser::queryWithParser(pp_material, "permittivity", material.permittivity);
    bool const has_function =
        pp_material.query(
            "permittivity_function(x,y,z,t)", material.permittivity_function);
    std::string permittivity_file;
    bool const has_file = pp_material.query("permittivity_from_file", permittivity_file);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(has_constant) + static_cast<int>(has_function) +
            static_cast<int>(has_file) == 1,
        "Dielectric material '" + name + "' must specify exactly one of "
        "permittivity, permittivity_function(x,y,z,t), or permittivity_from_file.");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !has_file,
        "Dielectric material '" + name + "' uses permittivity_from_file, but the "
        "file format is not defined yet.");

    if (has_function) {
        material.permittivity_type = PermittivityType::Function;
        m_has_time_dependent_permittivity = true;
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(material.permittivity) && material.permittivity >= 1.0_rt,
            "Dielectric material '" + name + "' has invalid permittivity. "
            "Relative permittivity must be finite and greater than or equal to 1.");
        material.permittivity_type = PermittivityType::Constant;
    }

    m_materials.push_back(std::move(material));
}

void
DielectricMaterials::AllocLevelData (
    WarpX& warpx,
    int const lev,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm)
{
    if (!hasDielectrics()) { return; }

    using warpx::fields::FieldType;

    amrex::IntVect const ng_epsilon(2);
    warpx.GetMultiFabRegister().alloc_init(
        FieldType::dielectric_epsilon, lev, ba, dm, 1, ng_epsilon, 1.0_rt);

    amrex::IntVect const ng_sdf(2);
    amrex::BoxArray const nodal_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    warpx.GetMultiFabRegister().alloc_init(
        FieldType::dielectric_signed_distance, lev, nodal_ba, dm, 1, ng_sdf, 0.0_rt);

    warpx.AllocInitMultiFab(
        m_material_id[lev], nodal_ba, dm, 1, ng_sdf, lev, "dielectric_material_id", 0);
}

void
DielectricMaterials::RemakeLevelData (
    WarpX& warpx,
    int const lev,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm)
{
    if (!hasDielectrics()) { return; }

    amrex::IntVect const ng_sdf(2);
    amrex::BoxArray const nodal_ba = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    warpx.AllocInitMultiFab(
        m_material_id[lev], nodal_ba, dm, 1, ng_sdf, lev, "dielectric_material_id", 0);
}

void
DielectricMaterials::ClearLevel (int const lev)
{
    if (lev < static_cast<int>(m_material_id.size())) {
        m_material_id[lev].reset();
    }
}

void
DielectricMaterials::InitData (WarpX& warpx)
{
    if (!hasDielectrics()) { return; }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        InitLevelData(warpx, lev);
    }
}

void
DielectricMaterials::InitLevelData (WarpX& warpx, int const lev)
{
    if (!hasDielectrics()) { return; }

    using warpx::fields::FieldType;

    auto& fields = warpx.GetMultiFabRegister();
    auto* sdf = fields.get(FieldType::dielectric_signed_distance, lev);
    auto* epsilon = fields.get(FieldType::dielectric_epsilon, lev);
    auto* material_id = m_material_id[lev].get();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        material_id != nullptr,
        "Dielectric material mask was not allocated before initialization.");

    FillSignedDistanceAndMask(*sdf, *material_id, warpx.Geom(lev));
    FillEpsilon(*epsilon, warpx.Geom(lev), warpx.gett_new(lev));
}

void
DielectricMaterials::UpdateEpsilon (WarpX& warpx, int const max_level, amrex::Real const time)
{
    if (!hasTimeDependentPermittivity()) { return; }

    using warpx::fields::FieldType;

    auto& fields = warpx.GetMultiFabRegister();
    for (int lev = 0; lev <= max_level; ++lev) {
        FillEpsilon(*fields.get(FieldType::dielectric_epsilon, lev), warpx.Geom(lev), time);
    }
}

amrex::iMultiFab const*
DielectricMaterials::MaterialID (int const lev) const
{
    if (lev >= static_cast<int>(m_material_id.size())) { return nullptr; }
    return m_material_id[lev].get();
}

void
DielectricMaterials::FillSignedDistanceAndMask (
    amrex::MultiFab& sdf,
    amrex::iMultiFab& material_id,
    amrex::Geometry const& geom) const
{
    sdf.setVal(0.0_rt);
    material_id.setVal(0);

    int object_id = 1;
    for (auto const& material : m_materials) {
        amrex::MultiFab object_sdf(
            sdf.boxArray(), sdf.DistributionMap(), 1, sdf.nGrowVect());

        if (material.shape_type == ShapeType::ImplicitFunction) {
            FillParserSignedDistance(object_sdf, material, geom);
        } else {
            FillSTLSignedDistance(object_sdf, material, geom);
        }

        MergeSignedDistanceAndMask(sdf, material_id, object_sdf, object_id);
        amrex::Gpu::streamSynchronize();
        ++object_id;
    }

    sdf.FillBoundary(geom.periodicity());
    ablastr::utils::communication::FillBoundary(material_id, geom.periodicity());
}

void
DielectricMaterials::FillParserSignedDistance (
    amrex::MultiFab& object_sdf,
    Material const& material,
    amrex::Geometry const& geom) const
{
#if defined(AMREX_USE_EB)
    auto parser = utils::parser::makeParser(material.implicit_function, {"x", "y", "z"});
    DielectricParserIF const parser_if(parser.compile<3>());
    auto gshop = amrex::EB2::makeShop(parser_if, parser);

    int const box_type = gshop.getBoxType(geom.Domain(), geom, amrex::RunOn::Cpu);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        box_type != decltype(gshop)::allcovered,
        "Dielectric material " + material.name +
        " covers the entire simulation domain. Full-domain dielectric materials "
        "are not supported yet.");

    amrex::EB2::Build(gshop, geom, 0, 0, std::max(4, object_sdf.nGrowVect().max()));
    amrex::EB2::IndexSpace* dielectric_index_space = &amrex::EB2::IndexSpace::top();

    {
        amrex::EB2::Level const& eb_level = dielectric_index_space->getLevel(geom);
        int const factory_ngrow = std::max(4, object_sdf.nGrowVect().max());
        auto eb_factory = amrex::makeEBFabFactory(
            &eb_level, object_sdf.boxArray(), object_sdf.DistributionMap(),
            amrex::Vector<int>{factory_ngrow, factory_ngrow, factory_ngrow},
            amrex::EBSupport::full);

        amrex::FillSignedDistance(object_sdf, eb_level, *eb_factory, 1, true);
    }

    amrex::EB2::IndexSpace::erase(dielectric_index_space);
#else
    amrex::ignore_unused(object_sdf, material, geom);
#endif
}

void
DielectricMaterials::FillSTLSignedDistance (
    amrex::MultiFab& object_sdf,
    Material const& material,
    amrex::Geometry const& geom) const
{
#if defined(AMREX_USE_EB) && defined(WARPX_DIM_3D)
    amrex::STLtools stl_tools;
    stl_tools.setBVHOptimization(material.stl_use_bvh != 0);
    stl_tools.read_stl_file(
        material.stl_file, material.stl_scale, material.stl_center,
        material.stl_reverse_normal);
    stl_tools.fillSignedDistance(object_sdf, object_sdf.nGrowVect(), geom);
#else
    amrex::ignore_unused(object_sdf, material, geom);
    WARPX_ABORT_WITH_MESSAGE("STL dielectric signed-distance generation requires 3D EB support.");
#endif
}

void
DielectricMaterials::MergeSignedDistanceAndMask (
    amrex::MultiFab& sdf,
    amrex::iMultiFab& material_id,
    amrex::MultiFab const& object_sdf,
    int const object_id) const
{
    MergeSignedDistanceAndMaskOnGrid(sdf, material_id, object_sdf, object_id);
}

void
DielectricMaterials::FillEpsilon (
    amrex::MultiFab& epsilon,
    amrex::Geometry const& geom,
    amrex::Real const time) const
{
    // Start from vacuum on every fill so time-dependent updates remove stale material values.
    epsilon.setVal(1.0_rt);

    for (auto const& material : m_materials) {
        // STL materials need a grid signed-distance function; parser materials evaluate directly.
        std::unique_ptr<amrex::MultiFab> stl_sdf;
        if (material.shape_type == ShapeType::STL) {
            stl_sdf = std::make_unique<amrex::MultiFab>(
                epsilon.boxArray(), epsilon.DistributionMap(), 1, epsilon.nGrowVect());
            FillSTLSignedDistance(*stl_sdf, material, geom);
        }

        if (material.shape_type == ShapeType::ImplicitFunction) {
            auto shape_parser = utils::parser::makeParser(
                material.implicit_function, {"x", "y", "z"});
            auto shape_exec = shape_parser.compile<3>();

            if (material.permittivity_type == PermittivityType::Constant) {
                SetImplicitConstantPermittivity(
                    epsilon, geom, shape_exec, material.permittivity);
            } else {
                auto permittivity_parser = utils::parser::makeParser(
                    material.permittivity_function, {"x", "y", "z", "t"});
                auto permittivity_exec = permittivity_parser.compile<4>();
                SetImplicitFunctionPermittivity(
                    epsilon, geom, shape_exec, permittivity_exec, time);
                // Function-valued permittivity is validated after evaluating every covered cell.
                AssertPermittivityFunctionValues(epsilon, material.name);
            }
        } else {
            if (material.permittivity_type == PermittivityType::Constant) {
                SetSTLConstantPermittivity(epsilon, *stl_sdf, material.permittivity);
            } else {
                auto permittivity_parser = utils::parser::makeParser(
                    material.permittivity_function, {"x", "y", "z", "t"});
                auto permittivity_exec = permittivity_parser.compile<4>();
                SetSTLFunctionPermittivity(epsilon, *stl_sdf, geom, permittivity_exec, time);
                // Function-valued permittivity is validated after evaluating every covered cell.
                AssertPermittivityFunctionValues(epsilon, material.name);
            }
        }
    }

    epsilon.FillBoundary(geom.periodicity());
}

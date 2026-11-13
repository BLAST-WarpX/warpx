/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIELECTRICMATERIALS_H_
#define WARPX_DIELECTRICMATERIALS_H_

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>
#include <vector>

class WarpX;

/**
 * \brief Runtime dielectric material descriptions and the grid data derived from them.
 *
 * Dielectric geometry is independent of the conductor EB geometry. The EB/STL machinery
 * is used here only to build a signed-distance function and material-id mask.
 */
class DielectricMaterials
{
public:
    explicit DielectricMaterials (int nlevs_max);

    void ReadParameters ();

    [[nodiscard]] bool hasDielectrics () const noexcept { return !m_materials.empty(); }

    void AllocLevelData (
        WarpX& warpx,
        int lev,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm);

    void RemakeLevelData (
        WarpX& warpx,
        int lev,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm);

    void ClearLevel (int lev);

    void InitData (WarpX& warpx);
    void InitLevelData (WarpX& warpx, int lev);

    [[nodiscard]] amrex::iMultiFab const* MaterialID (int lev) const;

private:
    enum class ShapeType
    {
        ImplicitFunction,
        STL
    };

    enum class PermittivityType
    {
        Constant,
        Function
    };

    struct Material
    {
        std::string name;

        ShapeType shape_type = ShapeType::ImplicitFunction;
        std::string implicit_function;

        std::string stl_file;
        amrex::Real stl_scale = 1.0;
        amrex::Array<amrex::Real, 3> stl_center{{0.0, 0.0, 0.0}};
        int stl_reverse_normal = 0;
        int stl_use_bvh = 1;

        PermittivityType permittivity_type = PermittivityType::Constant;
        amrex::Real permittivity = 1.0;
        std::string permittivity_function;
    };

    void ReadMaterial (std::string const& name, int default_stl_use_bvh);

    void FillLevelFields (
        amrex::MultiFab& sdf,
        amrex::iMultiFab& material_id,
        amrex::MultiFab& epsilon,
        amrex::Geometry const& geom) const;

    void SetMaterialPermittivity (
        amrex::MultiFab& epsilon,
        amrex::MultiFab const& object_sdf,
        Material const& material,
        amrex::Geometry const& geom) const;

    void FillParserSignedDistance (
        amrex::MultiFab& object_sdf,
        Material const& material,
        amrex::Geometry const& geom) const;

    void FillSTLSignedDistance (
        amrex::MultiFab& object_sdf,
        Material const& material,
        amrex::Geometry const& geom) const;

    void MergeSignedDistanceAndMask (
        amrex::MultiFab& sdf,
        amrex::iMultiFab& material_id,
        amrex::MultiFab const& object_sdf,
        int object_id) const;

    std::vector<Material> m_materials;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > m_material_id;
};

#endif // WARPX_DIELECTRICMATERIALS_H_

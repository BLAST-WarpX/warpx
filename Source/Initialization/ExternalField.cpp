/* Copyright 2023 Luca Fedeli
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalField.H"

#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_BaseFabUtility.H>

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
#   include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace
{
    enum class EMFieldType{E, B};

    template <EMFieldType T>
    ExternalFieldType string_to_external_field_type(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);

        if constexpr (T == EMFieldType::E){
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s != "parse_b_ext_grid_function",
                "parse_B_ext_grid_function can be used only for B_ext_grid_init_style");
        }
        else{
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s != "parse_e_ext_grid_function",
                "parse_E_ext_grid_function can be used only for E_ext_grid_init_style");
        }

        if ( s.empty() || s == "default"){
            return ExternalFieldType::default_zero;
        }
        else if ( s == "constant"){
            return ExternalFieldType::constant;
        }
        else if ( s == "parse_b_ext_grid_function" || s == "parse_e_ext_grid_function"){
            return ExternalFieldType::parse_ext_grid_function;
        }
        else if ( s == "read_from_file"){
            return ExternalFieldType::read_from_file;
        }
        else if ( s == "load_from_python"){
            return ExternalFieldType::load_from_python;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE(
                "'" + s + "' is an unknown external field type!");
        }

        return ExternalFieldType::default_zero;
    }
}

ExternalFieldParams::ExternalFieldParams(const amrex::ParmParse& pp_warpx)
{
    // default values of E_external_grid and B_external_grid
    // are used to set the E and B field when "constant" or
    // "parser" is not explicitly used in the input.
    std::string B_ext_grid_s;
    pp_warpx.query("B_ext_grid_init_style", B_ext_grid_s);
    B_ext_grid_type = string_to_external_field_type<EMFieldType::B>(B_ext_grid_s);

    std::string E_ext_grid_s;
    pp_warpx.query("E_ext_grid_init_style", E_ext_grid_s);
    E_ext_grid_type = string_to_external_field_type<EMFieldType::E>(E_ext_grid_s);

    //
    //  Constant external field
    //

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    auto v_B = std::vector<amrex::Real>(3);
    if (B_ext_grid_type == ExternalFieldType::constant) {
        utils::parser::getArrWithParser(pp_warpx, "B_external_grid", v_B);
    }
    std::copy(v_B.begin(), v_B.end(), B_external_grid.begin());

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    auto v_E = std::vector<amrex::Real>(3);
    if (E_ext_grid_type == ExternalFieldType::constant) {
        utils::parser::getArrWithParser(pp_warpx, "E_external_grid", v_E);
    }
    std::copy(v_E.begin(), v_E.end(), E_external_grid.begin());
    //___________________________________________________________________________


    //
    //  External E field with parser
    //

    // if the input string for the B-field is "parse_b_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (B_ext_grid_type == ExternalFieldType::parse_ext_grid_function) {

        //! Strings storing parser function to initialize the components of the magnetic field on the grid
        std::string str_Bx_ext_grid_function;
        std::string str_By_ext_grid_function;
        std::string str_Bz_ext_grid_function;

#if defined(WARPX_DIM_RZ)
        std::stringstream warnMsg;
        warnMsg << "Parser for external B (r and theta) fields does not work with cylindrical and spherical\n"
            << "The initial Br and Bt fields are currently hardcoded to 0.\n"
            << "The initial Bz field should only be a function of z.\n";
        ablastr::warn_manager::WMRecordWarning(
          "Inputs", warnMsg.str(), ablastr::warn_manager::WarnPriority::high);
        str_Bx_ext_grid_function = "0";
        str_By_ext_grid_function = "0";
#else
        utils::parser::Store_parserString(pp_warpx, "Bx_external_grid_function(x,y,z)",
          str_Bx_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "By_external_grid_function(x,y,z)",
          str_By_ext_grid_function);
#endif
        utils::parser::Store_parserString(pp_warpx, "Bz_external_grid_function(x,y,z)",
            str_Bz_ext_grid_function);

        Bxfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bx_ext_grid_function,{"x","y","z","t"}));
        Byfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_By_ext_grid_function,{"x","y","z","t"}));
        Bzfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bz_ext_grid_function,{"x","y","z","t"}));
    }
    //___________________________________________________________________________


    //
    //  External B field with parser
    //

    // if the input string for the E-field is "parse_e_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (E_ext_grid_type == ExternalFieldType::parse_ext_grid_function) {

#ifdef WARPX_DIM_RZ
        WARPX_ABORT_WITH_MESSAGE(
            "E parser for external fields does not work with RZ -- TO DO");
#endif

        //! Strings storing parser function to initialize the components of the electric field on the grid
        std::string str_Ex_ext_grid_function;
        std::string str_Ey_ext_grid_function;
        std::string str_Ez_ext_grid_function;

        utils::parser::Store_parserString(pp_warpx, "Ex_external_grid_function(x,y,z)",
            str_Ex_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "Ey_external_grid_function(x,y,z)",
           str_Ey_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "Ez_external_grid_function(x,y,z)",
           str_Ez_ext_grid_function);

        Exfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ex_ext_grid_function,{"x","y","z","t"}));
        Eyfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ey_ext_grid_function,{"x","y","z","t"}));
        Ezfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ez_ext_grid_function,{"x","y","z","t"}));
    }
    //___________________________________________________________________________


    //
    //  External fields from file
    //
    if (E_ext_grid_type == ExternalFieldType::read_from_file ||
        B_ext_grid_type == ExternalFieldType::read_from_file){
            const std::string read_fields_from_path="./";
            pp_warpx.query("read_fields_from_path", external_fields_path);
    }
    //___________________________________________________________________________
}

ExternalFieldReader::ExternalFieldReader (
    std::string read_fields_from_path,
    std::string F_name, std::string F_component,
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& pdx,
    amrex::Box const& dombox, bool distributed)
    : m_file(std::move(read_fields_from_path)),
      m_name(std::move(F_name)),
      m_component(std::move(F_component)),
      m_problo(problo),
      m_probdx(pdx),
      m_dombox(dombox),
      m_distributed(distributed)
{}

void ExternalFieldReader::load_data (amrex::RealBox const& pbox)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    using namespace amrex;

    auto series = openPMD::Series(m_file, openPMD::Access::READ_ONLY);
    auto iseries = series.iterations.begin()->second;
    auto F = iseries.meshes[m_name];

    bool c_order = F.getAttribute("dataOrder").get<std::string>() == "C";
    amrex::ignore_unused(c_order);

    auto axisLabels = F.getAttribute("axisLabels").get<std::vector<std::string>>();
    auto fileGeom = F.getAttribute("geometry").get<std::string>();

    bool xyz_order = true;

#if defined(WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "3D can only read from files with cartesian geometry");
    if (axisLabels.at(0) == "x" && axisLabels.at(1) == "y" && axisLabels.at(2) == "z") {
        xyz_order = true;
    } else if (axisLabels.at(2) == "x" && axisLabels.at(1) == "y" && axisLabels.at(0) == "z") {
        xyz_order = false;
    } else {
        WARPX_ABORT_WITH_MESSAGE("3D expects axisLabels {x, y, z} or {z, y, x}");
    }
#elif defined(WARPX_DIM_XZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "XZ can only read from files with cartesian geometry");
    if (axisLabels.at(0) == "x" && axisLabels.at(1) == "z") {
        xyz_order = true;
    } else if (axisLabels.at(1) == "x" && axisLabels.at(0) == "z") {
        xyz_order = false;
    } else {
        WARPX_ABORT_WITH_MESSAGE("XZ expects axisLabels {x, z} or {z, x}");
    }
#elif defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "thetaMode", "RZ can only read from files with 'thetaMode'  geometry");
    if (axisLabels.at(0) == "r" && axisLabels.at(1) == "z") {
        xyz_order = true;
    } else if (axisLabels.at(1) == "r" && axisLabels.at(0) == "z") {
        xyz_order = false;
    } else {
        WARPX_ABORT_WITH_MESSAGE("RZ expects axisLabels {r, z} or {z, r}");
    }
#elif defined(WARPX_DIM_1D_Z)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "1D3V can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels.at(0) == "z", "1D3V expects axisLabel {z}");
#endif

    const auto d = F.gridSpacing<long double>();
    if (xyz_order) {
        AMREX_D_TERM(m_dx[0] = Real(d.at(0));,
                     m_dx[1] = Real(d.at(1));,
                     m_dx[2] = Real(d.at(2)));
    } else {
        AMREX_D_TERM(m_dx[0] = Real(d.at(AMREX_SPACEDIM-1));,
                     m_dx[1] = Real(d.at(AMREX_SPACEDIM-2));,
                     m_dx[2] = Real(d.at(AMREX_SPACEDIM-3)));
    }

    const auto offset = F.gridGlobalOffset();
    if (xyz_order) {
        AMREX_D_TERM(m_offset[0] = Real(offset.at(0));,
                     m_offset[1] = Real(offset.at(1));,
                     m_offset[2] = Real(offset.at(2)));
    } else {
        AMREX_D_TERM(m_offset[0] = Real(offset.at(AMREX_SPACEDIM-1));,
                     m_offset[1] = Real(offset.at(AMREX_SPACEDIM-2));,
                     m_offset[2] = Real(offset.at(AMREX_SPACEDIM-3)));
    }

    // Load the first component if m_component is empty
    auto FC = m_component.empty() ? F.begin()->second : F[m_component];
    const auto extent = FC.getExtent();
    for (auto ex : extent) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ex < decltype(ex)(std::numeric_limits<int>::max()),
                                         "The openPMD file is too big");
    }
#if defined(WARPX_DIM_RZ)
    // extent[0] is for theta
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(extent.size() == 3 && extent[0] == 1,
                                     "External field reading is not implemented for more than one RZ mode (see #3829)");
    if (xyz_order) {
        m_size[0] = extent[1];
        m_size[1] = extent[2];
    } else {
        m_size[0] = extent[2];
        m_size[1] = extent[1];
    }
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(extent.size() == AMREX_SPACEDIM,
                                     "The openPMD file has wrong dimension.");
    if (xyz_order) {
        AMREX_D_TERM(m_size[0] = int(extent.at(0));,
                     m_size[1] = int(extent.at(1));,
                     m_size[2] = int(extent.at(2)));
    } else {
        AMREX_D_TERM(m_size[0] = int(extent.at(AMREX_SPACEDIM-1));,
                     m_size[1] = int(extent.at(AMREX_SPACEDIM-2));,
                     m_size[2] = int(extent.at(AMREX_SPACEDIM-3)));
    }
#endif

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_domain.setLo(idim, m_offset[idim]);
        m_domain.setHi(idim, m_offset[idim]+(m_size[idim]-1)*m_dx[idim]);
    }

    // Determine the full extent of the data we need
    IntVect lo, hi;
    if (m_distributed) {
        bool is_empty = false;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            auto plo = pbox.lo(idim);
            auto phi = pbox.hi(idim);
            auto ilo = int(std::floor( (plo-m_offset[idim])/m_dx[idim] ));
            auto ihi = int(std::floor( (phi-m_offset[idim])/m_dx[idim] ))+1; // +1 for interpolation
            --ilo; // in case there are roundoff errors
            ++ihi;
            lo[idim] = std::max(ilo, 0);
            hi[idim] = std::min(ihi, m_size[idim]-1);
            if (hi[idim] < lo[idim]) { is_empty = true; }
        }
        if (is_empty) { return; } // The openPMD file does not have the data we need.
    } else {
        lo = IntVect(0);
        hi = m_size-1;
    }

    // Determine the chunk data that will be loaded.
    BoxArray grids;
    DistributionMapping dmap;
    bool has_load = true;
    if (m_distributed && !m_moving_window) {
        // At this point, the data is distributed in an arbitrary way.  For
        // moving window, the data is loaded in a different way. We
        // duplicate the data needed for initializing the newly added
        // region, because the amount of data is relatively small and
        // because the duplicated approach is much simpler.
        grids = amrex::decompose(Box(lo,hi), ParallelDescriptor::NProcs());
        Vector<int> pmap(grids.size());
        std::iota(pmap.begin(), pmap.end(), 0);
        dmap.define(std::move(pmap));
        if (ParallelDescriptor::MyProc() < grids.size()) {
            auto const& b = grids[ParallelDescriptor::MyProc()];
            lo = b.smallEnd();
            hi = b.  bigEnd();
        } else {
            has_load = false;
        }
    }

    openPMD::Offset chunk_offset(extent.size(),0);
    openPMD::Extent chunk_extent(extent.size(),1);
#if defined(WARPX_DIM_RZ)
    if (xyz_order) {
        chunk_offset[1] = lo[0];
        chunk_offset[2] = lo[1];
        chunk_extent[1] = hi[0]-lo[0]+1;
        chunk_extent[2] = hi[1]-lo[1]+1;
    } else {
        chunk_offset[2] = lo[0];
        chunk_offset[1] = lo[1];
        chunk_extent[2] = hi[0]-lo[0]+1;
        chunk_extent[1] = hi[1]-lo[1]+1;
    }
#else
    if (xyz_order) {
        AMREX_D_TERM(chunk_offset[0] = lo[0];,
                     chunk_offset[1] = lo[1];,
                     chunk_offset[2] = lo[2]);
        AMREX_D_TERM(chunk_extent[0] = hi[0]-lo[0]+1;,
                     chunk_extent[1] = hi[1]-lo[1]+1;,
                     chunk_extent[2] = hi[2]-lo[2]+1);
    } else {
        AMREX_D_TERM(chunk_offset[AMREX_SPACEDIM-1] = lo[0];,
                     chunk_offset[AMREX_SPACEDIM-2] = lo[1];,
                     chunk_offset[AMREX_SPACEDIM-3] = lo[2]);
        AMREX_D_TERM(chunk_extent[AMREX_SPACEDIM-1] = hi[0]-lo[0]+1;,
                     chunk_extent[AMREX_SPACEDIM-2] = hi[1]-lo[1]+1;,
                     chunk_extent[AMREX_SPACEDIM-3] = hi[2]-lo[2]+1);
    }
#endif

    if (has_load) {
        m_FC_data_cpu = FC.loadChunk<double>(chunk_offset,chunk_extent);
    }
    series.flush();

    if (has_load) {
        Box box(lo,hi);
#ifdef AMREX_USE_GPU
        m_fab.resize(box, 1);
        Gpu::htod_memcpy_async(m_fab.dataPtr(), m_FC_data_cpu.get(), m_fab.nBytes());
        Gpu::streamSynchronize();
        m_FC_data_cpu.reset();
#else
        m_fab = BaseFab<double>(box, 1, m_FC_data_cpu.get());
#endif

#if (AMREX_SPACEDIM > 1)
        if ((xyz_order && c_order) || (!xyz_order && !c_order)) {
            BaseFab<double> tmp(box, 1);
            amrex::transposeCtoF(m_fab.dataPtr(), tmp.dataPtr(),
                                 AMREX_D_DECL(box.length(0),
                                              box.length(1),
                                              box.length(2)));
            amrex::Gpu::streamSynchronize();
            std::swap(m_fab,tmp);
            m_FC_data_cpu.reset();
        }
#endif
    }

    if (! grids.empty()) {
        m_mf.define(grids, dmap, 1, 0, MFInfo{}.SetAlloc(false));
        if (has_load) {
            m_mf.setFab(ParallelDescriptor::MyProc(),
                        BaseFab<double>(m_fab,amrex::make_alias,0,1));
        }
    }

#else
    amrex::ignore_unused(pbox);
    WARPX_ABORT_WITH_MESSAGE("ExternalFieldReader requires openPMD and it is not supported for 1D RCYLINDER and RSPHERE");
#endif
}

void ExternalFieldReader::prepare (amrex::BoxArray const& grids,
                                   amrex::DistributionMapping const& dmap,
                                   amrex::IntVect const& ngrow,
                                   std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_moving_window == false);

    amrex::RealBox rbox;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        rbox.setLo(idim, m_problo[idim] + m_dombox.smallEnd(idim)*m_probdx[idim]);
        rbox.setHi(idim, m_problo[idim] + m_dombox.  bigEnd(idim)*m_probdx[idim]);
    }
    if (get_zlab) {
        auto zlo = get_zlab(rbox.lo(AMREX_SPACEDIM-1));
        auto zhi = get_zlab(rbox.hi(AMREX_SPACEDIM-1));
        rbox.setLo(AMREX_SPACEDIM-1, zlo);
        rbox.setHi(AMREX_SPACEDIM-1, zhi);
    }
    load_data(rbox);

    if (m_distributed) {
        BoxList bl;
        bl.reserve(grids.size());
        for (int ibox = 0; ibox < int(grids.size()); ++ibox) {
            Box b = grids[ibox];
            b.surroundingNodes().grow(ngrow);
            IntVect lo, hi;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                auto plo = m_problo[idim] + b.smallEnd(idim)*m_probdx[idim];
                auto phi = m_problo[idim] + b.  bigEnd(idim)*m_probdx[idim];
                if (get_zlab && (idim == AMREX_SPACEDIM-1)) {
                    plo = get_zlab(plo);
                    phi = get_zlab(phi);
                }
                auto ilo = int(std::floor( (plo-m_offset[idim])/m_dx[idim] ));
                auto ihi = int(std::floor( (phi-m_offset[idim])/m_dx[idim] ))+1; // +1 for interpolation
                --ilo; // in case there are roundoff errors
                ++ihi;
                lo[idim] = std::max(ilo, 0);
                hi[idim] = std::min(ihi, m_size[idim]-1);
            }
            Box box(lo, hi);
            if (box.isEmpty()) {
                box = Box(IntVect(0),IntVect(0));
            }
            bl.push_back(box);
        }
        BoxArray ba(std::move(bl));
        FabArray<BaseFab<double>> tmpmf(ba,dmap,1,0);
        tmpmf.ParallelCopy(m_mf);
        m_mf = std::move(tmpmf);
        m_fab.clear();
        m_FC_data_cpu.reset();
    }
}

void ExternalFieldReader::make_cache_box (amrex::RealBox const& pbox, int moving_dir, int moving_sign)
{
    m_cache_domain = pbox;
    int dir = std::abs(moving_dir);
    amrex::Real factor = 10.0;
    if (moving_sign > 0) {
        amrex::Real newhi = m_cache_domain.hi(dir) + factor*m_cache_domain.length(dir);
        m_cache_domain.setHi(dir, newhi);
    } else {
        amrex::Real newlo = m_cache_domain.lo(dir) - factor*m_cache_domain.length(dir);
        m_cache_domain.setLo(dir, newlo);
    }
    // Grow the box a little bit so that m_cache_domain.contains(pbox) is true.
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_cache_domain.setLo(idim, m_cache_domain.lo(idim) - m_dx[idim]);
        m_cache_domain.setHi(idim, m_cache_domain.hi(idim) + m_dx[idim]);
    }
}

void ExternalFieldReader::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                   std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (! m_distributed) { return; }

    auto pboxz = pbox;
    if (get_zlab) {
        auto zlo = get_zlab(pbox.lo(AMREX_SPACEDIM-1));
        auto zhi = get_zlab(pbox.hi(AMREX_SPACEDIM-1));
        pboxz.setLo(AMREX_SPACEDIM-1, zlo);
        pboxz.setHi(AMREX_SPACEDIM-1, zhi);
    }

    if (!m_moving_window) {
        m_moving_window = true;
        m_mf.clear();
        make_cache_box(pboxz, moving_dir, moving_sign);
        load_data(m_cache_domain);
    } else {
        if (! m_cache_domain.contains(pboxz)) {
            make_cache_box(pboxz, moving_dir, moving_sign);
            if (m_cache_domain.intersects(m_domain)) {
                load_data(m_cache_domain);
            }
        }
    }
}

ExternalFieldView ExternalFieldReader::getView (int li) const
{
    if (m_distributed && !m_moving_window) {
        return make_view(m_mf.atLocalIdx(li));
    } else {
        return make_view(m_fab);
    }
}

ExternalFieldView ExternalFieldReader::getView () const
{
    return make_view(m_fab);
}

ExternalFieldView ExternalFieldReader::make_view (amrex::BaseFab<double> const& fab) const
{
    ExternalFieldView view;
    view.dx = m_dx;
    view.offset = m_offset;
    view.global_size = m_size;

    auto* p = fab.dataPtr();
    if (p) {
        auto const& b = fab.box();
        view.table = decltype(view.table)(const_cast<double*>(p),
#if (AMREX_SPACEDIM == 1)
                                          b.smallEnd(0), b.bigEnd(0)+1
#else
                                          {AMREX_D_DECL(b.smallEnd(0),
                                                        b.smallEnd(1),
                                                        b.smallEnd(2))},
                                          {AMREX_D_DECL(b.  bigEnd(0)+1,
                                                        b.  bigEnd(1)+1,
                                                        b.  bigEnd(2)+1)}
#endif
            );
    }
    return view;
}

/* Copyright 2019-2020 Axel Huebl, Ligia Diana Amorim, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorDensity.H"

#include <AMReX_OpenMP.H>

using namespace amrex;

void InjectorDensity::clear ()
{
    switch (type)
    {
    case Type::parser:
    {
        break;
    }
    case Type::fromfile:
    {
        object.fromfile.clear();
        break;
    }
    default:
        return;
    }
}

void InjectorDensity::prepare (amrex::BoxArray const& grids,
                               amrex::DistributionMapping const& dmap,
                               amrex::IntVect const& ngrow,
                               std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (type == Type::fromfile) {
        object.fromfile.prepare(grids,dmap,ngrow,get_zlab);
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    if (this->distributed()) {
        auto const nthreads = amrex::OpenMP::get_max_threads();
        inj_rho_data = std::unique_ptr<void,amrex::DataDeleter>
            (amrex::The_Cpu_Arena()->alloc(sizeof(InjectorDensity)*nthreads),
             amrex::DataDeleter{amrex::The_Cpu_Arena()});
        auto* p = reinterpret_cast<InjectorDensity*>(inj_rho_data.get());
        for (int tid = 0; tid < nthreads; ++tid) {
            inj_rho_omp.push_back(p++);
        }
        for (auto* q : inj_rho_omp) {
            std::memcpy((void*)q, (void const*)this, sizeof(InjectorDensity));
        }
    }
#endif
}

void InjectorDensity::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                               std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (type == Type::fromfile) {
        object.fromfile.prepare(pbox, moving_dir, moving_sign, get_zlab);
    }
}

void InjectorDensity::prepare (int li, InjectorDensity** inj_rho)
{
    if (type == Type::fromfile) {
#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
        if (inj_rho_data) {
            auto* my_inj_rho = inj_rho_omp[amrex::OpenMP::get_thread_num()];
            my_inj_rho->object.fromfile.prepare(li);
            *inj_rho = my_inj_rho;
        } else
#endif
        {
            object.fromfile.prepare(li);
#ifdef AMREX_USE_GPU
            amrex::Gpu::htod_memcpy_async(*inj_rho, this, sizeof(InjectorDensity));
#else
            *inj_rho = this;
#endif
        }
    }
}

bool InjectorDensity::needPreparation () const
{
    if (type == Type::fromfile) {
        return true;
    } else {
        return false;
    }
}

bool InjectorDensity::distributed () const
{
    if (type == Type::fromfile) {
        return object.fromfile.distributed();
    } else {
        return false;
    }
}

InjectorDensityFromFile::InjectorDensityFromFile (std::string const& a_file_name,
                                                  amrex::Geometry const& a_geom,
                                                  bool a_distributed)
{
    m_external_field_reader = new ExternalFieldReader
        (a_file_name, "density", "", a_geom.ProbLoArray(), a_geom.CellSizeArray(),
         amrex::convert(a_geom.Domain(),amrex::IntVect(1)), a_distributed);
}

void InjectorDensityFromFile::clear ()
{
    delete m_external_field_reader;
    m_external_field_reader = nullptr;
}

void InjectorDensityFromFile::prepare (amrex::BoxArray const& grids,
                                       amrex::DistributionMapping const& dmap,
                                       amrex::IntVect const& ngrow,
                                       std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (m_external_field_reader) {
        m_external_field_reader->prepare(grids,dmap,ngrow,get_zlab);
        m_external_field_view = m_external_field_reader->getView();
    }
}

void InjectorDensityFromFile::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                       std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (m_external_field_reader) {
        m_external_field_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_external_field_view = m_external_field_reader->getView();
    }
}

void InjectorDensityFromFile::prepare (int li)
{
    if (m_external_field_reader) {
        m_external_field_view = m_external_field_reader->getView(li);
    }
}

bool InjectorDensityFromFile::distributed () const
{
    if (m_external_field_reader) {
        return m_external_field_reader->distributed();
    } else {
        return false;
    }
}

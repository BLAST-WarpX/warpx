/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Maxence Thevenet,
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorMomentum.H"

#include <AMReX_OpenMP.H>

using namespace amrex;

void InjectorMomentum::clear () // NOLINT(readability-make-member-function-const)
{
    switch (type)
    {
    case Type::parser:
    case Type::gaussian:
    case Type::gaussianflux:
    case Type::uniform:
    case Type::maxwellian:
    case Type::juttner:
    case Type::constant:
    {
        break;
    }
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    inj_mom_omp.clear();
    inj_mom_data.reset();
#endif
}

void InjectorMomentum::prepare (amrex::BoxArray const& grids,
                                amrex::DistributionMapping const& dmap,
                                amrex::IntVect const& ngrow,
                                std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    switch (type)
    {
    case Type::maxwellian:
    {
        object.maxwellian.prepare(grids,dmap,ngrow,get_zlab);
        break;
    }
    case Type::juttner:
    {
        object.juttner.prepare(grids,dmap,ngrow,get_zlab);
        break;
    }
    case Type::parser:
    {
        object.parser.prepare(grids,dmap,ngrow,get_zlab);
        break;
    }
    default:
        break;
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    if (this->distributed()) {
        auto const nthreads = amrex::OpenMP::get_max_threads();
        inj_mom_data = std::unique_ptr<void,amrex::DataDeleter>
            (amrex::The_Cpu_Arena()->alloc(sizeof(InjectorMomentum)*nthreads),
             amrex::DataDeleter{amrex::The_Cpu_Arena()});
        auto* p = reinterpret_cast<InjectorMomentum*>(inj_mom_data.get());
        inj_mom_omp.clear();
        for (int tid = 0; tid < nthreads; ++tid) {
            inj_mom_omp.push_back(p++);
        }
        for (auto* q : inj_mom_omp) {
            std::memcpy((void*)q, (void const*)this, sizeof(InjectorMomentum));
        }
    }
#endif
}

void InjectorMomentum::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    switch (type)
    {
    case Type::maxwellian:
    {
        object.maxwellian.prepare(pbox, moving_dir, moving_sign, get_zlab);
        break;
    }
    case Type::juttner:
    {
        object.juttner.prepare(pbox, moving_dir, moving_sign, get_zlab);
        break;
    }
    case Type::parser:
    {
        object.parser.prepare(pbox, moving_dir, moving_sign, get_zlab);
        break;
    }
    default:
        break;
    }
}

void InjectorMomentum::prepareObject (int li)
{
    switch (type)
    {
    case Type::maxwellian:
    {
        object.maxwellian.prepare(li);
        break;
    }
    case Type::juttner:
    {
        object.juttner.prepare(li);
        break;
    }
    case Type::parser:
    {
        object.parser.prepare(li);
        break;
    }
    default:
        break;
    }
}

void InjectorMomentum::prepare (int li, InjectorMomentum** inj_mom)
{
    if (this->needPreparation()) {
#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
        if (inj_mom_data) {
            auto* my_inj_mom = inj_mom_omp[amrex::OpenMP::get_thread_num()];
            my_inj_mom->prepareObject(li);
            *inj_mom = my_inj_mom;
        } else
#endif
        {
            prepareObject(li);
#ifdef AMREX_USE_GPU
            amrex::Gpu::htod_memcpy_async(*inj_mom, this, sizeof(InjectorMomentum));
#else
            *inj_mom = this;
#endif
        }
    }
}

bool InjectorMomentum::needPreparation () const
{
    switch (type)
    {
    case Type::maxwellian:
    {
        return object.maxwellian.needPreparation();
    }
    case Type::juttner:
    {
        return object.juttner.needPreparation();
    }
    case Type::parser:
    {
        return object.parser.needPreparation();
    }
    default:
        return false;
    }
}

bool InjectorMomentum::distributed () const
{
    switch (type)
    {
    case Type::maxwellian:
    {
        return object.maxwellian.distributed();
    }
    case Type::juttner:
    {
        return object.juttner.distributed();
    }
    case Type::parser:
    {
        return object.parser.distributed();
    }
    default:
        return false;
    }
}

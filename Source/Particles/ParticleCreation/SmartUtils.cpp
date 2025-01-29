/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SmartUtils.H"

#include <AMReX_GpuContainers.H>

#include <algorithm>
#include <utility>
#include <vector>

PolicyVec getPolicies (const NameMap& names) noexcept
{
    std::vector<InitializationPolicy> h_policies;
    h_policies.resize(names.size());
    for (const auto& kv : names)
    {
        h_policies[kv.second] = initialization_policies[kv.first];
    }

    // host-to-device copy
    PolicyVec policies;
    policies.resize(names.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_policies.begin(), h_policies.end(), policies.begin());
    amrex::Gpu::Device::streamSynchronize();

    return policies;
}

SmartCopyTag getSmartCopyTag (const NameMap& src, const NameMap& dst) noexcept
{
    SmartCopyTag tag;

    std::vector<int> h_src_comps;
    std::vector<int> h_dst_comps;

    // we use the fact that maps are sorted
    auto i_src = src.begin();
    auto i_dst = dst.begin();
    while ( (i_src != src.end()) and (i_dst != dst.end()) )
    {
        if (i_src->first < i_dst->first)
        {
            // names are not the same and src is lower
            ++i_src;
        }
        else if (i_src->first > i_dst->first)
        {
            // names are not the same and dst is lower
            ++i_dst;
        }
        else
        {
            // name is in both...
            tag.common_names.push_back(i_src->first);
            h_src_comps.push_back(i_src->second);
            h_dst_comps.push_back(i_dst->second);
            ++i_src;
            ++i_dst;
        }
    }

    // host-to-device copies
    tag.src_comps.resize(h_src_comps.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_src_comps.begin(), h_src_comps.end(), tag.src_comps.begin());

    tag.dst_comps.resize(h_dst_comps.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dst_comps.begin(), h_dst_comps.end(), tag.dst_comps.begin());

    amrex::Gpu::Device::streamSynchronize();

    return tag;
}

/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONSERVATIVE_PHOTON_WEIGHT_TREE_H_
#define WARPX_CONSERVATIVE_PHOTON_WEIGHT_TREE_H_

#include "ConservativePhotonWeightProposal.H"

#include <array>
#include <functional>

namespace warpx::radiation
{
struct PhotonWeightTreeNode
{
    long double probability = 0;
    // Binary-heap FULL PATH identity: root 0, children 1/2, then 3..6.
    // Distinct surviving sets at the same depth never share a counter index.
    std::size_t counter_index = 0;
    std::array<int, 2> next{-1, -1}; // >=0 node, <0 leaf encoded as -index-1.
};

struct PhotonWeightTreeLeaf
{
    std::vector<double> weights; // Caller order, including retained zero rows.
    long double probability = 0; // Ideal endpoint-distance probability.
    std::size_t choices = 0;
};

struct PhotonWeightTree
{
    bool valid = false;
    int root = -1;
    std::vector<PhotonWeightTreeNode> nodes;
    std::vector<PhotonWeightTreeLeaf> leaves;
    long double weight_expectation_roundoff_bound = 0;
};

/** Host-only bounded complete represented tree, not a population target/cap.
 * <=14 rows and 1..3 attempted eliminations: <=7 nodes and <=8 terminal leaves.
 * EVERY node's BOTH stored numeric endpoints are prepared before ANY draw.
 * Rank/representation/asymmetric-probability stops retain that branch prefix;
 * no accepted ancestor can be rolled back based on the selected outcome.
 * Native callers must remeasure ALL terminal/prefix leaves and preflight their
 * whole physical envelope before sampling. Key order/pivots are canonical.
 */
inline PhotonWeightTree PrepareConservativePhotonWeightTree (
    std::vector<PhotonWeightRecord> const& input, std::size_t attempts)
{
    PhotonWeightTree out;
    if (input.size() > 14 || attempts == 0 || attempts > 3) { return out; }
    PhotonWeightProposalOptions validation;
    validation.target = std::max<std::size_t>(1, std::count_if(input.begin(), input.end(),
        [] (auto const& record) { return record.weight > 0; }));
    auto const checked = ProposeConservativePhotonWeights(input, validation);
    if (!checked.valid) { return out; }
    out.valid = true;
    std::vector<std::size_t> canonical(input.size());
    std::iota(canonical.begin(), canonical.end(), 0);
    std::sort(canonical.begin(), canonical.end(), [&] (auto a, auto b)
              { return input[a].key < input[b].key; });
    std::vector<PhotonWeightRecord> initial;
    for (auto index : canonical) { initial.push_back(input[index]); }
    out.nodes.reserve(7); out.leaves.reserve(8);
    auto leaf = [&] (auto const& rows, std::size_t depth, long double probability)
    {
        PhotonWeightTreeLeaf terminal;
        terminal.weights.resize(rows.size());
        for (std::size_t p = 0; p < rows.size(); ++p)
        { terminal.weights[canonical[p]] = rows[p].weight; }
        terminal.probability = probability; terminal.choices = depth;
        out.leaves.push_back(std::move(terminal));
        return -static_cast<int>(out.leaves.size());
    };
    std::function<std::pair<int, long double>(std::vector<PhotonWeightRecord> const&,
        std::size_t, std::size_t, long double)> build;
    build = [&] (auto const& rows, std::size_t depth, std::size_t path,
                 long double probability) -> std::pair<int, long double>
    {
        auto stop = [&] () -> std::pair<int, long double>
        { return {leaf(rows, depth, probability), 0}; };
        if (depth == attempts || rows.empty()) { return stop(); }
        std::vector<std::size_t> active;
        for (std::size_t p = 0; p < rows.size(); ++p)
        { if (rows[p].weight > 0) { active.push_back(p); } }
        if (active.size() < 2) { return stop(); }
        auto const positive_rows = active.size();
        // Match the unchanged primitive's exact first-feature-count+1 block.
        active.resize(std::min(active.size(), rows.front().features.size() + 1));
        auto const direction = photon_weight_detail::NullDirection(rows, active);
        if (direction.empty()) { return stop(); }
        long double plus = std::numeric_limits<long double>::infinity(), minus = plus;
        for (std::size_t p = 0; p < direction.size(); ++p)
        {
            if (direction[p] > 0) { plus = std::min(plus, rows[active[p]].weight / direction[p]); }
            if (direction[p] < 0) { minus = std::min(minus, -rows[active[p]].weight / direction[p]); }
        }
        auto const branch_probability = minus / (plus + minus);
        if (!(branch_probability >= 0x1.0p-52L && branch_probability <= 1 - 0x1.0p-52L))
        { return stop(); }
        PhotonWeightProposalOptions one;
        one.target = positive_rows - 1;
        std::array<PhotonWeightProposal, 2> endpoints{
            ProposeConservativePhotonWeights(rows, one, [] (std::size_t) { return 0.; }),
            ProposeConservativePhotonWeights(rows, one,
                [] (std::size_t) { return std::nextafter(1., 0.); })};
        for (auto const& endpoint : endpoints)
        { if (!endpoint.valid || endpoint.eliminations != 1) { return stop(); } }
        auto const index = static_cast<int>(out.nodes.size());
        out.nodes.push_back({branch_probability, path, {-1, -1}});
        long double child_bound = 0;
        for (int side = 0; side < 2; ++side)
        {
            auto child_rows = rows;
            for (std::size_t p = 0; p < rows.size(); ++p)
            { child_rows[p].weight = endpoints[side].weights[p]; }
            auto const child = build(child_rows, depth + 1, 2 * path + 1 + side,
                probability * (side == 0 ? branch_probability : 1 - branch_probability));
            out.nodes[index].next[side] = child.first;
            child_bound = std::max(child_bound, child.second);
        }
        auto bound = child_bound + std::max(endpoints[0].weight_expectation_roundoff_bound,
                                           endpoints[1].weight_expectation_roundoff_bound);
        if (bound > 0)
        { bound = std::nextafter(bound, std::numeric_limits<long double>::infinity()); }
        return {index, bound};
    };
    auto const root = build(initial, 0, 0, 1);
    out.root = root.first; out.weight_expectation_roundoff_bound = root.second;
    return out;
}

struct PhotonWeightTreeSample
{
    bool valid = false;
    std::vector<double> weights;
    std::size_t choices = 0;
};

template <typename Draw>
PhotonWeightTreeSample SampleConservativePhotonWeightTree (PhotonWeightTree const& tree, Draw draw)
{
    PhotonWeightTreeSample out;
    if (!tree.valid) { return out; }
    auto cursor = tree.root;
    while (cursor >= 0)
    {
        auto const& node = tree.nodes[cursor];
        double const uniform = draw(node.counter_index);
        if (!(uniform >= 0 && uniform < 1) || !std::isfinite(uniform)) { return out; }
        cursor = node.next[uniform < node.probability ? 0 : 1];
        ++out.choices;
    }
    out.weights = tree.leaves[-cursor - 1].weights;
    out.valid = true;
    return out;
}
} // namespace warpx::radiation
#endif

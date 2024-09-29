#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <functional>
#include <mutex>
#include <ios>

#include <omp.h>

#include <girgs/Generator.h>
#include <girgs/SpatialTree.h>
#include <girgs/WeightScaling.h>


namespace girgs {

static double evalBDFDistance(const std::vector<double> &v1, const std::vector<double> &v2, const std::vector<std::vector<int>> &minMaxSet) {
    double minMaxNorm = std::numeric_limits<double>::max();
    for (const auto &set : minMaxSet) {
        double currentMaxNorm = 0.0;
        for (int idx : set) {
            double diff = std::abs(v1[idx] - v2[idx]);
            double torusDist = std::min(diff, 1.0 - diff);
            currentMaxNorm = std::max(currentMaxNorm, torusDist);
        }
        minMaxNorm = std::min(minMaxNorm, currentMaxNorm);
    }
    return minMaxNorm;
}

static std::vector<std::vector<double>> filterVectors(const std::vector<std::vector<double>> &inputVectors, const std::vector<int> &indices) {
    std::vector<std::vector<double>> filteredVectors;

    for (const auto &vec : inputVectors) {
        std::vector<double> filteredVec;
        for (int idx : indices) {
            filteredVec.push_back(vec[idx]);  // Select only the dimensions specified in 'indices'
        }
        filteredVectors.push_back(filteredVec);  // Add the filtered vector to the result
    }

    return filteredVectors;
}

static std::vector<double> adjustWeights(const std::vector<double> &weights, const double thr_con, const double dimension) {
    std::vector<double> result(weights.size());
    const double scale = std::pow(thr_con, dimension) * std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int i = 0; i < weights.size(); i++) {
        result[i] = scale * weights[i];
    }
    return result;
}

std::vector<double> generateWeights(int n, double ple, int weightSeed, bool parallel) {
    const auto threads = parallel ? std::max(1, std::min(omp_get_max_threads(), n / 10000)) : 1;
    auto result = std::vector<double>(n);

    #pragma omp parallel num_threads(threads)
    {
        const auto tid = omp_get_thread_num();
        auto gen = default_random_engine{weightSeed >= 0 ? (weightSeed+tid) : std::random_device()()};
        auto dist = std::uniform_real_distribution<>{};

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            result[i] = std::pow((std::pow(0.5*n, -ple + 1) - 1) * dist(gen) + 1, 1 / (-ple + 1));
        }
    }

    return result;
}

std::vector<std::vector<double>> generatePositions(int n, int dimension, int positionSeed, bool parallel) {
    const auto threads = parallel ? std::max(1, std::min(omp_get_max_threads(), n / 10000)) : 1;
    auto result = std::vector<std::vector<double>>(n, std::vector<double>(dimension));

    #pragma omp parallel num_threads(threads)
    {
        const auto tid = omp_get_thread_num();
        auto gen = default_random_engine{positionSeed >= 0 ? (positionSeed+tid) : std::random_device()()};
        auto dist = std::uniform_real_distribution<>{};

        #pragma omp for schedule(static)
        for(int i=0; i<n; ++i)
            for (int d=0; d<dimension; ++d)
                result[i][d] = dist(gen);
    }

    return result;
}

double scaleWeightPolynomial(const std::vector<double> &weights, double desiredAvgDegree,
                                    const std::vector<int> &volume_poly, int length, double depth_vol) {
    return estimateWeightScalingBDF(weights, desiredAvgDegree, volume_poly, length, depth_vol);
}

double scaleWeights(std::vector<double>& weights, double desiredAvgDegree, int dimension, double alpha) {
    // estimate scaling with binary search
    double scaling;
    if(alpha > 8.0)
        scaling = estimateWeightScalingThreshold(weights, desiredAvgDegree, dimension);
    else if(alpha > 0.0 && alpha != 1.0)
        scaling = estimateWeightScaling(weights, desiredAvgDegree, dimension, alpha);
    else
        throw("I do not know how to scale weights for desired alpha :(");

    // scale weights
    for(auto& each : weights)
        each *= scaling;
    return scaling;
}

std::vector<std::pair<int, int>> generateEdges(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
        double alpha, int samplingSeed) {

    using edge_vector = std::vector<std::pair<int, int>>;
    edge_vector result;

    std::vector<std::pair<
            edge_vector,
            uint64_t[31] /* avoid false sharing */
    > > local_edges(omp_get_max_threads());

    constexpr auto block_size = size_t{1} << 20;

    std::mutex m;
    auto flush = [&] (const edge_vector& local) {
        std::lock_guard<std::mutex> lock(m);
        result.insert(result.end(), local.cbegin(), local.cend());
    };

    auto addEdge = [&](int u, int v, int tid) {
        auto& local = local_edges[tid].first;
        local.emplace_back(std::min(u,v), std::max(u,v));
        if (local.size() == block_size) {
            flush(local);
            local.clear();
        }
    };

    auto dimension = positions.front().size();

    switch(dimension) {
        case 1: makeSpatialTree<1>(weights, positions, alpha, addEdge).generateEdges(samplingSeed); break;
        case 2: makeSpatialTree<2>(weights, positions, alpha, addEdge).generateEdges(samplingSeed); break;
        case 3: makeSpatialTree<3>(weights, positions, alpha, addEdge).generateEdges(samplingSeed); break;
        case 4: makeSpatialTree<4>(weights, positions, alpha, addEdge).generateEdges(samplingSeed); break;
        case 5: makeSpatialTree<5>(weights, positions, alpha, addEdge).generateEdges(samplingSeed); break;
        default:
            std::cout << "Dimension " << dimension << " not supported." << std::endl;
            std::cout << "No edges generated." << std::endl;
            break;
    }

    for(const auto& v : local_edges)
        flush(v.first);

    return result;
}

std::vector<std::pair<int, int>> generateBDFEdges(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
                                                  const std::vector<std::vector<int>> &minMaxSet,  const std::vector<std::vector<int>> &reducedMinMaxSet,
                                                  const int depth_vol, const double thr_con, const double thr_con_generation) {

    using edge_vector = std::vector<std::pair<int, int>>;
    edge_vector result;

    std::vector<std::pair<
            edge_vector,
            uint64_t[31] /* avoid false sharing */
    > > local_edges(omp_get_max_threads());

    constexpr auto block_size = size_t{1} << 20;

    std::mutex m;
    auto flush = [&] (const edge_vector& local) {
        std::lock_guard<std::mutex> lock(m);
        result.insert(result.end(), local.cbegin(), local.cend());
    };

    // To avoid computing exponent multiple times
    std::vector<double> threshold_weights(weights.size());
    for (int i = 0; i < weights.size(); i++) {
        threshold_weights[i] = std::pow(weights[i], 1.0 / depth_vol);
    }

    // add edges also filters the edges. Avoids modifying the other code
    auto addEdge = [&](int u, int v, int tid) {
        const auto dist = evalBDFDistance(positions[u], positions[v], minMaxSet);
        if (dist > thr_con * threshold_weights[u] * threshold_weights[v]) {
            return;
        }
        auto& local = local_edges[tid].first;
        local.emplace_back(std::min(u, v),std::max(u, v));
        if (local.size() == block_size) {
            flush(local);
            local.clear();
        }
    };

    auto alpha = std::numeric_limits<double>::infinity();
    auto r_weights = adjustWeights(weights, thr_con, depth_vol);
    for (const auto maxSet : reducedMinMaxSet){
        auto r_positions = filterVectors(positions, maxSet);
        auto dimension = depth_vol;
        switch(dimension) {
            case 1: makeSpatialTree<1>(r_weights, r_positions, alpha, addEdge, true).generateEdges(0); break;
            case 2: makeSpatialTree<2>(r_weights, r_positions, alpha, addEdge, true).generateEdges(0); break;
            case 3: makeSpatialTree<3>(r_weights, r_positions, alpha, addEdge, true).generateEdges(0); break;
            case 4: makeSpatialTree<4>(r_weights, r_positions, alpha, addEdge, true).generateEdges(0); break;
            case 5: makeSpatialTree<5>(r_weights, r_positions, alpha, addEdge, true).generateEdges(0); break;
            default:
                std::cout << "Dimension " << dimension << " not supported." << std::endl;
                std::cout << "No edges generated." << std::endl;
                break;
        }
    }
    for(const auto& v : local_edges)
        flush(v.first);

    return result;
}

std::vector<std::pair<int, int>> generateBDFEdgesTrivial(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
                                                         const std::vector<std::vector<int>> &minMaxSet, const double depth_vol, const double thr_con) {

    std::vector<std::pair<int, int>> closePairs;
    std::vector<double> threshold_weights(weights.size());
    for (int i = 0; i < weights.size(); i++) {
        threshold_weights[i] = std::pow(weights[i], 1.0 / depth_vol);
    }
    for (int i = 0; i < positions.size(); ++i) {
        for (int j = i + 1; j < positions.size(); ++j) {
            const auto dist = evalBDFDistance(positions[i], positions[j], minMaxSet);
            if (dist <= thr_con * threshold_weights[i] * threshold_weights[j]) {
                closePairs.emplace_back(i,j);
            }
        }
    }
    return closePairs;
}


std::vector<std::pair<int, int>> checkBDFEdges(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions, std::vector<std::pair<int, int>> &edges,
                                                         const std::vector<std::vector<int>> &minMaxSet, const double depth_vol, const double thr_con) {

    std::vector<std::pair<int, int>> result;
    std::vector<double> threshold_weights(weights.size());
    for (int i = 0; i < weights.size(); i++) {
        threshold_weights[i] = std::pow(weights[i], 1.0 / depth_vol);
    }

    for (int count = 0; count < edges.size(); ++count) {
        auto i = edges[count].first;
        auto j = edges[count].second;
        const auto dist = evalBDFDistance(positions[i], positions[j], minMaxSet);
        if (dist <= thr_con * threshold_weights[i] * threshold_weights[j]) {
            result.emplace_back(i,j);
        }
    }
    return result;
}


void saveDot(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
             const std::vector<std::pair<int, int>> &graph, const std::string &file) {

    std::ofstream f{file};
    if(!f.is_open())
        throw std::runtime_error{"Error: failed to open file \"" + file + '\"'};
    f << "graph girg {\n\toverlap=scale;\n\n";
    f << std::fixed;
    for (int i = 0; i < weights.size(); ++i) {
        f << '\t' << i << " [label=\""
          << std::setprecision(2) << weights[i] << std::setprecision(6)
          << "\", pos=\"";
        for (auto d = 0u; d < positions[i].size(); ++d)
            f << (d == 0 ? "" : ",") << positions[i][d];
        f << "\"];\n";
    }
    f << '\n';
    for (auto &edge : graph)
        f << '\t' << edge.first << "\t-- " << edge.second << ";\n";
    f << "}\n";
}

} // namespace girgs

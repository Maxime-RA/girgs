#pragma once

#include <vector>
#include <string>

#include <girgs/girgs_api.h>


namespace girgs {


/**
 * @brief
 *  The weights are sampled according to a power law distribution between [1, n)
 *
 * @param n
 *  The size of the graph. Should match with size of positions.
 * @param ple
 *  The power law exponent to sample the new weights. Should be 2.0 to 3.0.
 * @param weightSeed
 *  A seed for weight sampling. Should not be equal to the position seed.
 *
 * @return
 *  The weights according to the desired distribution.
 */
GIRGS_API std::vector<double> generateWeights(int n, double ple, int weightSeed, bool parallel = true);

/**
 * @brief
 *  Samples d dimensional coordinates for n points on a torus \f$[0,1)^d\f$.
 *
 * @param n
 *  Size of the graph.
 * @param dimension
 *  Dimension of the geometry.
 * @param positionSeed
 *  Seed to sample the positions.
 *
 * @return
 *  The positions on a torus. All inner vectors have the same length.
 */
GIRGS_API std::vector<std::vector<double>> generatePositions(int n, int dimension, int positionSeed, bool parallel = true);

/**
 *
 * Samples the edges of a BDF-GIRG using the trivial O(n^2) algorithm
 *
 *
 * @param weights weights of the vertices
 * @param positions positions of the vertices
 * @param minMaxSet min-max set of the BDF
 * @param depth_vol the volumetric depth of the BDF
 * @param thr_con the threshold constant
 *
 * @return edges of BDF-GIRG
 *
 */
GIRGS_API std::vector<std::pair<int, int>> generateBDFEdgesTrivial(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
                                                                   const std::vector<std::vector<int>> &minMaxSet, const double depth_vol, const double thr_con);
/**
 * Samples the edge of a BDF by sampling the edges of each min-max set separately. May sample double edges.
 *
 * @param weights weights of the vertices
 * @param positions positions of the vertices
 * @param minMaxSet min-max set of the BDF
 * @param reducedMinMaxSet simplified min-max set
 * @param depth_vol the volumetric depth of the BDF
 * @param thr_con the threshold constant
 * @param thr_con_generation the threshold constant used for the generation. Must be larger then thr_con, best runtime fo avg. degree ~ 5 * dimension.
 *
 * @return edges of BDF-GIRG
 *
 */
GIRGS_API std::vector<std::pair<int, int>> generateBDFEdges(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions,
                                                      const std::vector<std::vector<int>> &minMaxSet,  const std::vector<std::vector<int>> &reducedMinMaxSet,
                                                      const int depth_vol, const double thr_con, const double thr_con_generation);

/**
 *
 * Checks edges given a min-max set
 *
 *
 * @param weights weights of the vertices
 * @param positions positions of the vertices
 * @param edges edges to check
 * @param minMaxSet min-max set of the BDF
 * @param depth_vol the volumetric depth of the BDF
 * @param thr_con the threshold constant
 *
 * @return only the correct edges
 *
 */
GIRGS_API std::vector<std::pair<int, int>> checkBDFEdges(const std::vector<double> &weights, const std::vector<std::vector<double>> &positions, std::vector<std::pair<int, int>> &edges,
                                                       const std::vector<std::vector<int>> &minMaxSet, const double depth_vol, const double thr_con);


/**
 * Determines the threshold constant based on the weights, the desired average degree and a polynomial describing the volume
 *
 * @param weights weights of the vertices
 * @param desiredAvgDegree desired average degree
 * @param volume_poly polynomial of the volume
 * @param length length of the BDF (length of the reduced min-max set may be used for a better runtime)
 * @param depth_vol volumetric depth
 * @return threshold_constant * (1/n^depth_vol)
 */
GIRGS_API double scaleWeightPolynomial(const std::vector<double> &weights, double desiredAvgDegree,
                                    const std::vector<int> &volume_poly, int length, double depth_vol);

/**
 * @brief
 *  Scales all weights so that the expected average degree equals desiredAvgDegree.
 *  Implemented as binary search over an estimation function.
 *
 * @bug
 *  For \f$\alpha > 10\f$ we use the estimation for threshold graphs due to numerical difficulties.
 *  This leads to slightly higher degrees than desired.
 *  Also I experienced inaccurate results for \f$9 \leq \alpha < 10\f$.
 *
 * @param weights
 *  The weights to be modified.
 * @param desiredAvgDegree
 *  The desired average degree.
 * @param dimension
 *  Dimension of the underlying geometry. Should equal the dimensionality of the positions.
 * @param alpha
 *  Parameter of the algorithm. Should be the same as for the generation process.
 *
 * @return
 *  The scaling s applied to all weights.
 *  The constant c hidden in the theta of the edge probabilities is \f$s^\alpha\f$ for \f$\alpha < \infty\f$
 *  and \f$s^{1/d}\f$ in the threshold case.
 */
GIRGS_API double scaleWeights(std::vector<double>& weights, double desiredAvgDegree, int dimension, double alpha);

/**
 * @brief
 *  Samples edges according to weights and positions.
 *  An edge between node u and v is formed with probability \f$ \left(\frac{w_u w_v / W}{|| x_u - x_v ||^d}\right)^\alpha \f$ or 1.0 if the term exceeds 1.0.
 *
 * @param weights
 *  Power law distributed weights.
 * @param positions
 *  Positions on a torus. All inner vectors should have the same length indicating the dimension of the torus.
 * @param alpha
 *  Edge probability parameter.
 * @param samplingSeed
 *  Seed to sample the edges.
 *
 * @return
 *  An edge list with zero based indices.
 */
GIRGS_API std::vector<std::pair<int,int>> generateEdges(const std::vector<double>& weights, const std::vector<std::vector<double>>& positions,
        double alpha, int samplingSeed);


/**
 * @brief
 *  Saves the graph in .dot format (graphviz).
 *  The weight is saved as a label and the coordinates as a position attribute for each Node.
 *
 * @param weights
 *  Power law distributed weights.
 * @param positions
 *  The positions on a torus.
 * @param graph
 *  An edge list with zero based indices.
 * @param file
 *  The name of the output file.
 */
GIRGS_API void saveDot(const std::vector<double>& weights, const std::vector<std::vector<double>>& positions,
        const std::vector<std::pair<int,int>> &graph, const std::string &file);



} // namespace girgs

#pragma once

#include <vector>

#include <girgs/girgs_api.h>


namespace girgs {

GIRGS_API double estimateWeightScalingBDF(const std::vector<double> &weights, double desiredAvgDegree,
                                          const std::vector<int> &volume_poly, int length, double depth_vol);

GIRGS_API double estimateWeightScaling(const std::vector<double> &weights, double desiredAvgDegree, int dimension, double alpha);

GIRGS_API double estimateWeightScalingThreshold(const std::vector<double>& weights, double desiredAvgDegree, int dimension);

} // namespace girgs


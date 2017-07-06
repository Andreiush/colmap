// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "optim/support_measurement.h"

namespace colmap {

InlierSupportMeasurer::Support InlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.residual_sum = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.residual_sum += residual;
    }
  }

  return support;
}

bool InlierSupportMeasurer::Compare(const Support& support1,
                                    const Support& support2) {
  if (support1.num_inliers > support2.num_inliers) {
    return true;
  } else {
    return support1.num_inliers == support2.num_inliers &&
           support1.residual_sum < support2.residual_sum;
  }
}

MEstimatorSupportMeasurer::Support MEstimatorSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.score = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.score += residual;
    } else {
      support.score += max_residual;
    }
  }

  return support;
}

bool MEstimatorSupportMeasurer::Compare(const Support& support1,
                                        const Support& support2) {
  return support1.score < support2.score;
}

LRTSupportMeasurer::Support LRTSupportMeasurer::Evaluate(const double p_s, const double eps, const double sigma, const size_t n)
{
    Support support;
    support.sigma = sigma;
    support.inRatio = eps;
    double a = eps*std::log(eps/p_s);
    double b = (1-eps)*(std::log(1-eps)/(1-p_s));
    support.LRT = 2*n*(a+b);
    return support;
}

bool LRTSupportMeasurer::Compare(const Support &support1, const Support &support2)
{
    if (support1.LRT > support2.LRT) {
      return true;
    } else {
      return support1.LRT == support2.LRT &&
             support1.sigma < support2.sigma;
    }
}

}  // namespace colmap

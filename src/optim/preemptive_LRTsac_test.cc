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

#define TEST_NAME "optim/preemptive_LRTsac"
#include "util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/pose.h"
#include "base/similarity_transform.h"
#include "estimators/similarity_transform.h"
#include "optim/preemptive_LRTsac.h"
#include "util/random.h"

using namespace colmap;


BOOST_AUTO_TEST_CASE(TestReport) {
  preemptive_LRTsac<SimilarityTransformEstimator<3>>::Report report;
  BOOST_CHECK_EQUAL(report.success, false);
  BOOST_CHECK_EQUAL(report.num_trials, 0);
  BOOST_CHECK_EQUAL(report.num_hyp, 0);
  BOOST_CHECK_EQUAL(report.support.inRatio, 0);
  BOOST_CHECK_EQUAL(report.support.sigma, 0);
  BOOST_CHECK_EQUAL(report.support.LRT, 0);
  BOOST_CHECK_EQUAL(report.num_chunks_checked, 0);
  BOOST_CHECK_EQUAL(report.inlier_mask.size(), 0);
}


BOOST_AUTO_TEST_CASE(TestSimilarityTransform) {
  SetPRNGSeed(0);

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation.
  const SimilarityTransform3 orig_tform(2, ComposeIdentityQuaternion(),
                                        Eigen::Vector3d(100, 10, 10));

  // Generate exact data.
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_samples; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    dst[i] = Eigen::Vector3d(RandomReal(-3000.0, -2000.0),
                             RandomReal(-4000.0, -3000.0),
                             RandomReal(-5000.0, -4000.0));
  }

  // Robustly estimate transformation using RANSAC.
  LRToptions options;
  options.sigma_min = 0.1;
  options.sigma_max = 15;
  options.delta_sigma = 0.5;
  options.dim = 3;
  options.D = 1000;
  options.max_num_trials = 500;
  options.B = 100;

  preemptive_LRTsac<SimilarityTransformEstimator<3>> prelrtsac(options);
  const auto report = prelrtsac.Estimate(src, dst);

  BOOST_CHECK_EQUAL(report.success, true);
  BOOST_CHECK_GT(report.num_trials, 0);

  std::cout << "trials for hypotheses: "<< report.num_trials << std::endl;
  std::cout << "iterations: "<< report.num_hyp << std::endl;
  std::cout << "sigma: "<< report.support.sigma << std::endl;

  // Make sure outliers were detected correctly.
  BOOST_CHECK_EQUAL(report.support.inRatio * num_samples, num_samples -
                    num_outliers);
  for (size_t i = 0; i < num_samples; ++i) {
    if (i < num_outliers) {
      BOOST_CHECK(!report.inlier_mask[i]);
    } else {
      BOOST_CHECK(report.inlier_mask[i]);
    }
  }

  // Make sure original transformation is estimated correctly.
  const double matrix_diff =
      (orig_tform.Matrix().topLeftCorner<3, 4>() - report.model).norm();
  BOOST_CHECK(std::abs(matrix_diff) < 1e-6);
}

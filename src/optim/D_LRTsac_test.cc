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

#define TEST_NAME "optim/D_LRTsac"
#include "util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/pose.h"
#include "base/similarity_transform.h"
#include "estimators/essential_matrix.h"
#include "optim/D_LRTsac.h"
#include "util/random.h"

using namespace colmap;


BOOST_AUTO_TEST_CASE(TestEssential) {
  SetPRNGSeed(0);

  const size_t num_samples = 900;
  const size_t num_outliers = 300;
  double s = -0.5;
  double c = std::sqrt(3.0)/2.0;
  // Create Rotation -30deg around y and Translation
  Eigen::Matrix3d R;
  R << c, 0, s,
       0, 1, 0,
      -s, 0, c;
  Eigen::Vector3d T(15,0,0);
//  Eigen::Matrix3d Tx;
//  Tx << 0, -T[2], T[1],
//        T[2], 0, -T[0],
//        -T[1], T[0], 0;

//  Eigen::Matrix3d E = Tx*R;

  // Generate exact data.
  std::vector<Eigen::Vector2d> src;
  std::vector<Eigen::Vector2d> dst;
  double maxx = 0,maxy = 0;
  for(int x = 0; x<30; ++x)
    for(int y = 0; y<30; ++y)
    {
      double z = RandomReal(20.0,30.0);
      Eigen::Vector3d X(x*10*z,y*10*z,z);
      Eigen::Vector3d X2 = R*X+T;

      Eigen::Vector2d p(X2[0]/X2[2],X2[1]/X2[2]);
      maxx = std::max(p[0],maxx);
      maxy = std::max(p[1],maxy);
      src.emplace_back(x*10,y*10);
      dst.push_back(p);
    }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    dst[i] = Eigen::Vector2d(RandomReal(500.0, 600.0),
                             RandomReal(500.0, 600.0));
  }

  // Robustly estimate transformation using RANSAC.
  LRToptions options;
  options.sigma_min = 0.1;
  options.sigma_max = 10;
  options.delta_sigma = 0.5;
  options.dim = 1;
  options.D = 425;
  options.A = 900;
  options.D2 = std::sqrt(maxx*maxx + maxy*maxy);
  options.A2 = maxx*maxy;

  D_LRTsac<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      lrtsac(options);

  const auto report = lrtsac.Estimate(src, dst);

  BOOST_CHECK_EQUAL(report.success, true);
  BOOST_CHECK_GT(report.num_trials, 0);

  std::cout << "vpm: "<< report.vpm << std::endl;
  std::cout << "iterations: "<< report.num_trials << std::endl;
  std::cout << "sigma1: "<< report.support.sigma1 << std::endl;
  std::cout << "sigma2: "<< report.support.sigma2 << std::endl;

  // Make sure outliers were detected correctly.
  BOOST_CHECK_EQUAL(report.support.num_inliers, num_samples -
                    num_outliers);
  for (size_t i = 0; i < num_samples; ++i) {
    if (i < num_outliers) {
      BOOST_CHECK(!report.inlier_mask[i]);
    } else {
      BOOST_CHECK(report.inlier_mask[i]);
    }
  }


}

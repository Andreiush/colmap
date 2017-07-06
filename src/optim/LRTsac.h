#ifndef LRTSAC_H
#define LRTSAC_H

#include <cfloat>
#include <random>
#include <stdexcept>
#include <vector>

#include "optim/random_sampler.h"
#include "optim/support_measurement.h"
#include "util/alignment.h"
#include "util/logging.h"

namespace colmap {

struct LRToptions {
  // C is minimum value of LRT for a model to be valid. Related to a chosen
  // alpha (type I error) and min. sample size via chi squared distribution.
  // not used because we will just use a min inlier ratio for the applications
  // of reconstruction
  double c = 25;
  // Type II error beta
  double beta = 0.05;
  // A priori assumed minimum inlier ratio, which determines the maximum number
  // of iterations. Only applies if smaller than `max_num_trials`.
  double min_inlier_ratio = 0.1;

  // Parameters for area: depending on the dimension of the error, it is the
  // probability of  falling in the inlier area
  unsigned int dim = 1;
  double D;
  double A;

  // Abort the iteration if minimum probability that one sample is free from
  // outliers is reached.
  double confidence = 0.99;

  // Number of random trials to estimate model from random subset.
  size_t min_num_trials = 0;
  size_t max_num_trials = std::numeric_limits<size_t>::max();

  void Check() const {
    CHECK_GT(c, 0);
    CHECK_GE(beta, 0);
    CHECK_LE(beta, 1);
    CHECK_GE(min_inlier_ratio, 0);
    CHECK_LE(min_inlier_ratio, 1);
    CHECK_GE(dim, 1);
    CHECK_LE(dim, 2);
    CHECK_GT(D, 0);
    CHECK_GT(A, 0);
    CHECK_GE(confidence, 0);
    CHECK_LE(confidence, 1);
    CHECK_LE(min_num_trials, max_num_trials);
  }
};

template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class LRTsac {
 public:
  struct Report {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Whether the estimation was successful.
    bool success = false;

    // The number of trials / iterations.
    size_t num_trials = 0;

    // The support of the estimated model.
    typename SupportMeasurer::Support support;

    // Number of points evaluated per model
    double vpm = 0;

    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;

    // The estimated model.
    typename Estimator::M_t model;
  };

  explicit LRTsac(const LRToptions& options);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y);

  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

 protected:
  LRToptions options_;

  inline void makeSigmas(std::vector<double> &sigmas, double sigma_min, double sigma_max, double delta_sigma);
  inline double area(int dim, double D, double A, double sigma);
  inline double LRT(double p_s, double eps, int n);
  double bisectionForEps(double sigma, double c, int n);
  size_t iterations(double eps, double conf, double beta, int nSample);
  inline void makeTaos(std::vector<double> &taos, int B, int n, double beta);
  inline void computeEps(std::vector<double> &eps, const std::vector<int> &ks, int n);
  int iterRansac(double eps, double conf, int nSample);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename SupportMeasurer, typename Sampler>
LRTsac<Estimator, SupportMeasurer, Sampler>::LRTsac(
    const LRToptions& options)
  : sampler(Sampler(Estimator::kMinNumSamples)), options_(options) {
  options.Check();

  // Determine max_num_trials based on assumed `min_inlier_ratio`.
  const size_t kNumSamples = 100000;
  const size_t dyn_max_num_trials = ComputeNumTrials(
                                      static_cast<size_t>(options_.min_inlier_ratio * kNumSamples), kNumSamples,
                                      options_.confidence);
  options_.max_num_trials =
      std::min<size_t>(options_.max_num_trials, dyn_max_num_trials);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
size_t LRTsac<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
    const size_t num_inliers, const size_t num_samples,
    const double confidence) {
  const double inlier_ratio = num_inliers / static_cast<double>(num_samples);

  const double nom = 1 - confidence;
  if (nom <= 0) {
    return std::numeric_limits<size_t>::max();
  }

  const double denom = 1 - std::pow(inlier_ratio, Estimator::kMinNumSamples);
  if (denom <= 0) {
    return 1;
  }

  return static_cast<size_t>(std::ceil(std::log(nom) / std::log(denom)));
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename LRTsac<Estimator, SupportMeasurer, Sampler>::Report
LRTsac<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
  CHECK_EQ(X.size(), Y.size());

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  const double max_residual = options_.max_error * options_.max_error;

  std::vector<double> residuals(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  sampler.Initialize(num_samples);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    sampler.SampleXY(X, Y, &X_rand, &Y_rand);

    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand, Y_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      estimator.Residuals(X, Y, sample_model, &residuals);
      CHECK_EQ(residuals.size(), X.size());

      const auto support = support_measurer.Evaluate(residuals, max_residual);

      // Save as best subset if better than all previous subsets.
      if (support_measurer.Compare(support, best_support)) {
        best_support = support;
        best_model = sample_model;

        dyn_max_num_trials = ComputeNumTrials(best_support.num_inliers,
                                              num_samples, options_.confidence);
      }

      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  report.support = best_support;
  report.model = best_model;

  // No valid model was found.
  if (report.support.num_inliers < estimator.kMinNumSamples) {
    return report;
  }

  report.success = true;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), X.size());

  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] <= max_residual) {
      report.inlier_mask[i] = true;
    } else {
      report.inlier_mask[i] = false;
    }
  }

  return report;
}

}  // namespace colmap

#endif // LRTSAC_H

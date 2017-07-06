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

  //bucket size for bailout
  size_t B;

  //range for sigma search
  double sigma_min;
  double sigma_max;
  double delta_sigma;

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
    CHECK_GT(B, 0);
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

  inline void makeSigmas(std::vector<double> &sigmas);
  inline double area(double sigma);
  inline double LRT(double ps, double eps, double n);
  double bisectionForEps(double sigma, double c, int n);
  size_t iterations(double eps);
  inline void makeTaos(std::vector<double> &taos, int n);
  inline void computeEps(std::vector<double> &eps, const std::vector<int> &ks,
                         int n);
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
  const size_t dyn_max_num_trials = iterations(options_.min_inlier_ratio);
  options_.max_num_trials =
      std::min<size_t>(options_.max_num_trials, dyn_max_num_trials);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline void LRTsac<Estimator, SupportMeasurer, Sampler>::makeSigmas(
    std::vector<double> &sigmas)
{
    int nSigma = (options_.sigma_max-options_.sigma_min)/options_.delta_sigma;
    sigmas.resize(nSigma);
    for(int i = 0; i < nSigma; ++i)
        sigmas[i] = sigma_min+options_.delta_sigma*i;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline double LRTsac<Estimator, SupportMeasurer, Sampler>::area(double sigma)
{
    if(dim == 1)
        return 2*sigma*options_.D/options_.A;
    else
        return M_PI*(sigma*sigma)/(options_.D);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline double LRTsac<Estimator, SupportMeasurer, Sampler>::LRT(double p_s,
                                                               double eps,
                                                               int n)
{
    double a = eps*std::log(eps/p_s);
    double b = (1-eps)*(std::log(1-eps)/(1-p_s));
    return 2*n*(a+b);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
double LRTsac<Estimator, SupportMeasurer, Sampler>::bisectionForEps(double p_s,
                                                                    double c,
                                                                    int n)
{
    double a = p_s;
    double b = 1;
    double m = (a+b)/2.0;
    double val = LRT(p_s,m,n);
    while (abs(val-c) > 0.01)
    {
        if(val > c)
            b = m;
        else
            a = m;
        m = (a+b)/2;
        val = LRT(p_s,m,n);
    }
    return m;
}


template <typename Estimator, typename SupportMeasurer, typename Sampler>
size_t LRTsac<Estimator, SupportMeasurer, Sampler>::iterations(double eps)
{
  double nom = 1 - options_.confidence;
  if (nom <= 0) {
    return std::numeric_limits<size_t>::max();
  }
  double den = 1 - (std::pow(eps,Estimator::kMinNumSamples)*(1-options_.beta));
  if (den <= 0) {
    return 1;
  }
  return static_cast<size_t>(std::ceil(std::log(nom) / std::log(den)));
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline void LRTsac<Estimator, SupportMeasurer, Sampler>::makeTaos(
    std::vector<double> &taos, int n)
{
  int Q = std::ceil((double)n/options_.B);
  taos.resize(Q);
  for(int i = 0; i < Q; ++i)
  {
    double a = std::log((double)Q)-std::log(options_.beta);
    double m = (i+1)*B;
    taos[i] = std::sqrt(a/(2*m));
  }
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline void LRTsac<Estimator, SupportMeasurer, Sampler>::computeEps(
    std::vector <double> &eps, std::vector<int> const &ks, int n)
{
  eps.resize(ks.size());
  eps[0] = ks[0];
  for(int i = 1; i < ks.size(); ++i)
    eps[i] = eps[i-1]+ks[i];
  for(int i = 0; i < eps.size(); ++i)
    eps[i]/=n;
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

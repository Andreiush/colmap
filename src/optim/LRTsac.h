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
#include "util/random.h"

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
  double D2;
  double A2;
  // Abort the iteration if minimum probability that one sample is free from
  // outliers is reached.
  double confidence = 0.99;

  // Number of random trials to estimate model from random subset.
  size_t min_num_trials = 0;
  size_t max_num_trials = std::numeric_limits<size_t>::max();

  //bucket size for bailout
  size_t B = 20;

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
    CHECK_LE(dim, 3);
    CHECK_GT(D, 0);
    CHECK_GT(A, 0);
    CHECK_GE(D2, 0);
    CHECK_GE(A2, 0);
    CHECK_GE(confidence, 0);
    CHECK_LE(confidence, 1);
    CHECK_LE(min_num_trials, max_num_trials);
    CHECK_GT(B, 0);
  }
};

template <typename Estimator, typename SupportMeasurer = LRTSupportMeasurer,
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
  LRTsac(const LRToptions& options, bool extra);
  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y);
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y,
                  const std::vector<double>& scales);
  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

protected:
  LRToptions options_;

  inline void makeSigmas(std::vector<double> &sigmas);
  inline double area(double sigma);
  inline double LRT(double ps, double eps, int n);
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
LRTsac<Estimator, SupportMeasurer, Sampler>::LRTsac(
    const LRToptions& options, bool extra)
  : sampler(Sampler(Estimator::kMinNumSamples+1)), options_(options) {
  options.Check();
}


template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline void LRTsac<Estimator, SupportMeasurer, Sampler>::makeSigmas(
    std::vector<double> &sigmas)
{
  int nSigma = (options_.sigma_max-options_.sigma_min)/options_.delta_sigma;
  sigmas.resize(nSigma);
  for(int i = 0; i < nSigma; ++i)
    sigmas[i] = options_.sigma_min + options_.delta_sigma*i;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline double LRTsac<Estimator, SupportMeasurer, Sampler>::area(double sigma)
{
  switch (options_.dim) {
    case 1:
      return 2*sigma*options_.D/options_.A;
      break;
    case 2:
      return M_PI*(sigma*sigma)/(options_.A);
      break;
    case 3:
      return 4*M_PI*(sigma*sigma*sigma)/std::pow(options_.D, 3);
      break;
    default:
      return 0;
  }
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
    double m = (i+1) * options_.B;
    taos[i] = std::sqrt(a/(2*m));
  }
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
inline void LRTsac<Estimator, SupportMeasurer, Sampler>::computeEps(
    std::vector <double> &eps, std::vector<int> const &ks, int n)
{
  eps.resize(ks.size());
  if(eps.empty())
    return;
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
    const std::vector<typename Estimator::Y_t>& Y)
{
  std::vector<double> scales(X.size(),1.0);
  return Estimate(X, Y, scales);
}


template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename LRTsac<Estimator, SupportMeasurer, Sampler>::Report
LRTsac<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y,
    const std::vector<double> &scales) {
  CHECK_EQ(X.size(), Y.size());
  CHECK_EQ(X.size(), scales.size());
  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;
  report.vpm = 0;
  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  std::vector<double> sigmas;
  makeSigmas(sigmas);
  size_t nSigmas = sigmas.size();
  const size_t n = X.size();

  double maxScale = *std::max_element(scales.begin(), scales.end());

  std::vector<double> p_s(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    p_s[i] = area(maxScale*sigmas[i]);

  std::vector<double> minEps(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    minEps[i] = options_.min_inlier_ratio;

  size_t bucket_size = options_.B;
  std::vector<double> residuals(1);

  std::vector<double> taos;
  makeTaos(taos,n);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  std::vector<typename Estimator::X_t> X_test(1);
  std::vector<typename Estimator::Y_t> Y_test(1);

  std::vector<int> random_indices(num_samples);
  std::iota(random_indices.begin(), random_indices.end(), 0);
  Shuffle(static_cast<uint32_t>(num_samples), &random_indices);

  sampler.Initialize(num_samples);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());


  size_t models_tried = 0;
  for (report.num_trials = 0; report.num_trials < max_num_trials && nSigmas > 0;
       ++report.num_trials)
  {
    sampler.SampleXY(X, Y, &X_rand, &Y_rand);

    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand, Y_rand);

    std::vector<size_t> numIterations(nSigmas);
    for(int k = nSigmas-1; k >= 0; --k)
      numIterations[k]=iterations(minEps[k]);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models)
    {
      models_tried++;
      bool bailout = false;
      std::vector<int> ks(nSigmas,0);
      std::vector<double> curEps;
      //for each data point
      for(size_t j = 0; j < num_samples; ++j)
      {
        X_test[0] = X[random_indices[j]];
        Y_test[0] = Y[random_indices[j]];
        estimator.Residuals(X_test, Y_test, sample_model, &residuals);
        CHECK_EQ(residuals.size(), X_test.size());
        double err = std::sqrt(residuals[0])/scales[random_indices[j]];
        int ind =  std::max(ceil((err-options_.sigma_min)/options_. delta_sigma),
                            0.0);
        report.vpm ++;
        if(ind < nSigmas && ind >= 0)
          ks[ind] += 1;
        // we tested B points, check for bailout
        if( (j != 0 && j % bucket_size == 0) || j == num_samples-1 )
        {
          bailout = true;
          computeEps(curEps, ks, j+1);
          int ind = j/bucket_size - 1;
          for(int k = 0; k < curEps.size(); ++k)
            if(curEps[k] >= std::max(minEps[k] - taos[ind],0.0005))
            {
              bailout = false;
              break;
            }
          if(bailout)
            break;
        }
      }//end for each point
      //check if we didn't bailout, then we found a good model
      if(!bailout)
      {
        int bestk = -1;
        for(size_t k = 0; k < curEps.size(); ++k)
        {
          const auto support = support_measurer.Evaluate(p_s[k], curEps[k],
                                                         sigmas[k], n);

          // Save as best subset if better than all previous subsets.
          if (support_measurer.Compare(support, best_support))
          {
            best_support = support;
            best_model = sample_model;
            report.success = true;
            bestk = k;
          }
        }
        if(bestk >= 0)
        {
          for(int k = 0; k < nSigmas; ++k)
            minEps[k] = bisectionForEps(p_s[k],best_support.LRT, n);
          for(int k = nSigmas-1; k >= 0; --k)
            numIterations[k]=iterations(minEps[k]);
          max_num_trials = std::min(numIterations[std::max(0,bestk-2)],
              options_.max_num_trials);
        }
      }
      for(int k = nSigmas-1; k >= 0; --k)
      {
        size_t itNew = numIterations[k];
        if(itNew < report.num_trials)
        {
          nSigmas--;
         // std::cout<<"nsigmas "<<nSigmas<<std::endl;
        }
        else
          break;
      }
    }
  }


  report.support = best_support;
  report.model = best_model;
  report.vpm = report.vpm/models_tried;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), X.size());
  double max_residual = std::pow(report.support.sigma, 2);
  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    double s = scales[i]*scales[i];
    if (residuals[i] <= max_residual/s) {
      report.inlier_mask[i] = true;
    } else {
      report.inlier_mask[i] = false;
    }
  }

  return report;
}

}  // namespace colmap

#endif // LRTSAC_H

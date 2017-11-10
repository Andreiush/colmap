#ifndef LO_LRTSAC_H
#define LO_LRTSAC_H

#include <cfloat>
#include <random>
#include <stdexcept>
#include <vector>

#include "optim/random_sampler.h"
#include "optim/support_measurement.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/random.h"
#include "LRTsac.h"

namespace colmap {
template <typename Estimator, typename LocalEstimator,
          typename Sampler = RandomSampler>
class LO_LRTsac : public LRTsac<Estimator, LRTSupportMeasurer, Sampler>
{
public:
  using typename LRTsac<Estimator, LRTSupportMeasurer, Sampler>::Report;
  explicit LO_LRTsac(const LRToptions& options);

  // Robustly estimate model with LRTsac
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y);


  // Objects used in RANSAC procedure.
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::estimator;
  LocalEstimator local_estimator;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::sampler;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::support_measurer;


private:
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::options_;

  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::makeSigmas;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::area;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::LRT;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::bisectionForEps;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::iterations;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::makeTaos;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::computeEps;
  inline void createKs(std::vector<int> ks,
                       const std::vector<double> &residuals);

};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename LocalEstimator, typename Sampler>
LO_LRTsac<Estimator, LocalEstimator, Sampler>::LO_LRTsac(
    const LRToptions& options)
  :LRTsac<Estimator, LRTSupportMeasurer, Sampler>(options)  {}

template <typename Estimator, typename LocalEstimator, typename Sampler>
inline void LO_LRTsac<Estimator, LocalEstimator, Sampler>::createKs(
    std::vector<int> ks, const std::vector<double> &residuals)
{
  int nSigmas = ks.size();
  for(auto &res : residuals)
  {
    double err = std::sqrt(res);
    int ind =  std::max(ceil((err-options_.sigma_min)/options_. delta_sigma),
                        0.0);
    if(ind < nSigmas && ind >= 0)
      ks[ind] += 1;
  }
  return;
}

template <typename Estimator, typename LocalEstimator, typename Sampler>
typename LO_LRTsac<Estimator, LocalEstimator, Sampler>::Report
LO_LRTsac<Estimator, LocalEstimator, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
  CHECK_EQ(X.size(), Y.size());

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;
  report.vpm = 0;
  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  LRTSupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;
  bool best_model_is_local = false;

  std::vector<double> sigmas;
  makeSigmas(sigmas);
  size_t nSigmas = sigmas.size();
  const size_t n = X.size();

  std::vector<double> p_s(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    p_s[i] = area(sigmas[i]);

  std::vector<double> minEps(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    minEps[i] = options_.min_inlier_ratio;

  size_t bucket_size = options_.B;
  std::vector<double> residuals(1);
  std::vector<double> residualsAll(num_samples);
  std::vector<double> taos;
  makeTaos(taos,n);

  std::vector<typename LocalEstimator::X_t> X_inlier;
  std::vector<typename LocalEstimator::Y_t> Y_inlier;

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  std::vector<typename Estimator::X_t> X_test(1);
  std::vector<typename Estimator::Y_t> Y_test(1);

  std::vector<int> random_indices(num_samples);
  std::iota(random_indices.begin(), random_indices.end(), 0);

  sampler.Initialize(num_samples);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
  if(max_num_trials <= 0)
    max_num_trials = options_.max_num_trials;

  size_t models_tried = 0;
  for (report.num_trials = 0; report.num_trials < max_num_trials && nSigmas > 0;
       ++report.num_trials)
  {
    Shuffle(static_cast<uint32_t>(num_samples), &random_indices);

    //sampler.SampleXY(X, Y, &X_rand, &Y_rand);
    for(size_t j = 0; j < Estimator::kMinNumSamples; ++j)
    {
      X_rand[j] = X[random_indices[j]];
      Y_rand[j] = Y[random_indices[j]];
    }
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
        residualsAll[j] = residuals[0];
        double err = std::sqrt(residuals[0]);
        int ind =  std::max(ceil((err-options_.sigma_min)/options_. delta_sigma),
                            0.0);
        report.vpm ++;
        if(ind < nSigmas && ind >= 0)
          ks[ind] += 1;
        // we tested B points, LO_LRTsaceck for bailout
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
      }// end for each point
      // if we didn't bailout, then chances are we found a good model
      if(!bailout)
      {
        int bestk = -1;
        for(size_t k = 0; k < curEps.size(); ++k)
        {
          const auto support = support_measurer.Evaluate(p_s[k], curEps[k],
                                                         sigmas[k], n);

          // Save as best model if better than all previous models.
          if (support_measurer.Compare(support, best_support))
          {
            best_support = support;
            best_model = sample_model;
            bestk = k;
            report.success = true;
          }
        }
        // If we found a better model, do local optimization
        if(bestk >= 0)
        {
          best_model_is_local = false;
          size_t num_inliers = best_support.inRatio * num_samples;

          if(num_inliers > Estimator::kMinNumSamples &&
             num_inliers >= LocalEstimator::kMinNumSamples)
          {
            X_inlier.clear();
            Y_inlier.clear();
            X_inlier.reserve(num_inliers);
            Y_inlier.reserve(num_inliers);
            double max_residual = std::pow(best_support.sigma, 2);
            for (size_t i = 0; i < residualsAll.size(); ++i) {
              if (residualsAll[i] <= max_residual) {
                X_inlier.push_back(X[i]);
                Y_inlier.push_back(Y[i]);
              }
            }
            const std::vector<typename LocalEstimator::M_t> local_models =
                local_estimator.Estimate(X_inlier, Y_inlier);

            for (const auto& local_model : local_models)
            {
              local_estimator.Residuals(X, Y, local_model, &residuals);
              CHECK_EQ(residuals.size(), X.size());

              //evaluate the model with all points
              std::vector<int> ks(nSigmas);
              createKs(ks, residuals);
              computeEps(curEps, ks, num_samples);
              for(size_t k = 0; k < curEps.size(); ++k)
              {
                const auto local_support =
                    support_measurer.Evaluate(p_s[k], curEps[k], sigmas[k], n);
                // Check if non-locally optimized model is better.
                if (support_measurer.Compare(local_support, best_support))
                {
                  best_support = local_support;
                  best_model = local_model;
                  best_model_is_local = true;
                }
              }
            }

            //update min inlier ratios and number of iterations
            for(int k = 0; k < nSigmas; ++k)
              minEps[k] = bisectionForEps(p_s[k],best_support.LRT, n);
            for(int k = nSigmas-1; k >= 0; --k)
              numIterations[k]=iterations(minEps[k]);
            max_num_trials = std::min(numIterations[std::max(0,bestk-2)],
                options_.max_num_trials);
            if(max_num_trials <= 0)
              max_num_trials = options_.max_num_trials;
          }// end local optimization (if enough inliers)
        }// end if bestk (if a better model was found)
      }//end if !bailout
    }//end for each model
    // get rid of the sigmas for which we've already done enough iterations
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
  }//end for each iteration


  report.support = best_support;
  report.model = best_model;
  report.vpm = report.vpm/models_tried;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  if (best_model_is_local) {
    local_estimator.Residuals(X, Y, report.model, &residuals);
  } else {
    estimator.Residuals(X, Y, report.model, &residuals);
  }

  CHECK_EQ(residuals.size(), X.size());
  double max_residual = std::pow(report.support.sigma, 2);
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


}


#endif // LO_LRTSAC_H

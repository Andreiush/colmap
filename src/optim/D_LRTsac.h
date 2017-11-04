#ifndef D_LRTSAC_H
#define D_LRTSAC_H

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
class D_LRTsac : public LRTsac<Estimator, DLRTSupportMeasurer, Sampler>
{
public:
  using typename LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::Report;
  explicit D_LRTsac(const LRToptions& options);

  // Robustly estimate essential or fundamental ONLY with LRTsac with
  // different noise levels per image AND local optimization
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y);


  // Objects used in RANSAC procedure.
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::estimator;
  LocalEstimator local_estimator;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::sampler;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::support_measurer;

private:
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::options_;

  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::makeSigmas;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::area;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::LRT;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::bisectionForEps;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::iterations;
  using LRTsac<Estimator, DLRTSupportMeasurer, Sampler>::makeTaos;

  void computeEps(std::vector<double> &eps, const Eigen::MatrixXi &ks,
                  int n);
  void createKs(Eigen::MatrixXi& ks,
                const std::vector<double> &residuals1,
                const std::vector<double> &residuals2);
  double area2(double sigma);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename LocalEstimator, typename Sampler>
D_LRTsac<Estimator, LocalEstimator, Sampler>::D_LRTsac(
    const LRToptions& options)
  :LRTsac<Estimator, DLRTSupportMeasurer, Sampler>(options)  {}

template <typename Estimator, typename LocalEstimator, typename Sampler>
void D_LRTsac<Estimator, LocalEstimator, Sampler>::createKs(
    Eigen::MatrixXi &ks, const std::vector<double> &residuals1,
    const std::vector<double> &residuals2)
{

  for(size_t i = 0; i < residuals1.size(); ++i)
  {
    double err1 = residuals1[i];
    double err2 = residuals2[i];

    int ind1 =  std::max(ceil((err1-options_.sigma_min)/options_. delta_sigma),
                         0.0);
    int ind2 =  std::max(ceil((err2-options_.sigma_min)/options_. delta_sigma),
                         0.0);

    if(ind1 < ks.rows() && ind1 >= 0 && ind2 < ks.cols() && ind2 >= 0)
      ks(ind1,ind2) += 1;

  }
  return;
}

template <typename Estimator, typename LocalEstimator, typename Sampler>
double D_LRTsac<Estimator, LocalEstimator, Sampler>::area2(double sigma)
{
  switch (options_.dim) {
    case 1:
      return 2*sigma*options_.D2/options_.A2;
      break;
    case 2:
      return M_PI*(sigma*sigma)/(options_.A2);
      break;
    case 3:
      return 4*M_PI*(sigma*sigma*sigma)/std::pow(options_.D2, 3);
      break;
    default:
      return 0;
  }
}

template <typename Estimator, typename LocalEstimator, typename Sampler>
void D_LRTsac<Estimator, LocalEstimator, Sampler>::computeEps(
    std::vector <double> &eps, const Eigen::MatrixXi &ks, int n)
{
  eps.resize(ks.size());
  Eigen::MatrixXi integral = Eigen::MatrixXi::Zero(ks.rows()+1,ks.cols()+1);
  for(int i = 0; i < ks.rows(); ++i)
    for(int j = 0; j < ks.cols(); ++j)
      integral(i+1,j+1) = ks(i,j) - integral(i,j) + integral(i, j+1) +
                          integral(i+1, j);
  for(int i = 0; i < ks.rows(); ++i)
    for(int j = 0; j < ks.cols(); ++j)
      eps[i*ks.cols()+j]= double (integral(i+1, j+1)) / n;
}

template <typename Estimator, typename LocalEstimator, typename Sampler>
typename D_LRTsac<Estimator, LocalEstimator, Sampler>::Report
D_LRTsac<Estimator, LocalEstimator, Sampler>::Estimate(
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

  DLRTSupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;
  bool best_model_is_local = false;

  std::vector<double> sigmas1, sigmas2;
  makeSigmas(sigmas1);
  size_t nSigmas1 = sigmas1.size();
  makeSigmas(sigmas2);
  size_t nSigmas2 = sigmas2.size();


  const size_t n = X.size();

  std::vector<double> p_s1(nSigmas1);
  for(int i = 0; i < nSigmas1; ++i)
    p_s1[i] = area(sigmas1[i]);

  std::vector<double> p_s2(nSigmas2);
  for(int i = 0; i < nSigmas2; ++i)
    p_s2[i] = area2(sigmas2[i]);

  size_t nSigmas = nSigmas1 * nSigmas2;
  std::vector<double> p_x(nSigmas);

  for(int i = 0; i < nSigmas1; ++i)
    for(int j = 0; j < nSigmas2; ++j)
      p_x[i*nSigmas2+j] = (p_s1[i]*p_s2[j]);

  std::vector<double> minEps(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    minEps[i] = options_.min_inlier_ratio;

  size_t bucket_size = options_.B;
  std::vector<double> residuals1(1), residuals2(1);
  std::vector<double> residualsAll1(num_samples), residualsAll2(num_samples);
  std::vector<double> taos;
  makeTaos(taos,n);

  Eigen::MatrixXi ks;

  std::vector<typename LocalEstimator::X_t> X_inlier;
  std::vector<typename LocalEstimator::Y_t> Y_inlier;

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
  for (report.num_trials = 0; report.num_trials < max_num_trials && nSigmas1 > 0;
       ++report.num_trials)
  {
    sampler.SampleXY(X, Y, &X_rand, &Y_rand);

    // Estimate model for current minimal subset.
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
      ks = Eigen::MatrixXi::Zero(nSigmas1, nSigmas2);
      std::vector<double> curEps;
      //for each data point
      for(size_t j = 0; j < num_samples; ++j)
      {
        X_test[0] = X[random_indices[j]];
        Y_test[0] = Y[random_indices[j]];
        estimator.Residuals(X_test, Y_test, sample_model, &residuals1,
                            &residuals2);
        CHECK_EQ(residuals1.size(), X_test.size());

        residualsAll1[j] = residuals1[0];
        residualsAll2[j] = residuals2[0];

        double err1 = residuals1[0];
        double err2 = residuals2[0];

        int ind1 =  std::max(ceil((err1-options_.sigma_min)/options_. delta_sigma),
                             0.0);
        int ind2 =  std::max(ceil((err2-options_.sigma_min)/options_. delta_sigma),
                             0.0);

        report.vpm ++;

        if(ind1 < nSigmas1 && ind1 >= 0 && ind2 < nSigmas2 && ind2 >= 0)
          ks(ind1,ind2) += 1;


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
      }// end for each point
      // if we didn't bailout, then chances are we found a good model
      if(!bailout)
      {
        int bestk = -1;
        for(size_t k = 0; k < curEps.size(); ++k)
        {
          int k1 = k % nSigmas2;
          int k2 = k / nSigmas2;
          const auto support = support_measurer.Evaluate(p_x[k], curEps[k],
                                                         sigmas1[k1],
                                                         sigmas2[k2], n);

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
            double max_residual1 = best_support.sigma1;
            double max_residual2 = best_support.sigma2;
            for (size_t i = 0; i < residualsAll1.size(); ++i) {
              if (residualsAll1[i] <= max_residual1 &&
                  residualsAll2[i] <= max_residual2) {
                X_inlier.push_back(X[i]);
                Y_inlier.push_back(Y[i]);
              }
            }
            const std::vector<typename LocalEstimator::M_t> local_models =
                local_estimator.Estimate(X_inlier, Y_inlier);

            for (const auto& local_model : local_models)
            {
              local_estimator.Residuals(X, Y, local_model, &residuals1,
                                        &residuals2);
              CHECK_EQ(residuals1.size(), X.size());
              CHECK_EQ(residuals1.size(), residuals2.size());

              //evaluate the model with all points
              ks = Eigen::MatrixXi::Zero(nSigmas1, nSigmas2);
              createKs(ks, residuals1, residuals2);
              computeEps(curEps, ks, num_samples);
              for(size_t k = 0; k < curEps.size(); ++k)
              {
                int k1 = k % nSigmas2;
                int k2 = k / nSigmas2;
                const auto local_support =
                    support_measurer.Evaluate(p_x[k], curEps[k], sigmas1[k1],
                                              sigmas2[k2], n);
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
              minEps[k] = bisectionForEps(p_x[k],best_support.LRT, n);
//            for(int k = nSigmas-1; k >= 0; --k)
//              numIterations[k]=iterations(minEps[k]);
            max_num_trials = std::min(iterations(minEps[bestk]),
                options_.max_num_trials);
          }// end local optimization (if enough inliers)
        }// end if bestk (if a better model was found)
      }//end if !bailout

      // get rid of the sigmas for which we've already done enough iterations
//      for(int k = nSigmas-1; k >= 0; --k)
//      {
//        size_t itNew = numIterations[k];
//        if(itNew < report.num_trials)
//        {
//          nSigmas--;
//          // std::cout<<"nsigmas "<<nSigmas<<std::endl;
//        }
//        else
//          break;
//      }
    }//end for each model
  }//end for each iteration


  report.support = best_support;
  report.model = best_model;
  report.vpm = report.vpm/models_tried;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  if (best_model_is_local) {
    local_estimator.Residuals(X, Y, report.model, &residuals1, &residuals2);
  } else {
    estimator.Residuals(X, Y, report.model, &residuals1, &residuals2);
  }

  CHECK_EQ(residuals1.size(), X.size());
  CHECK_EQ(residuals1.size(), residuals2.size());

  double max_residual1 = report.support.sigma1;
  double max_residual2 = report.support.sigma2;
  int inCount = 0;
  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals1.size(); ++i) {
    if (residuals1[i] <= max_residual1 && residuals2[i] <= max_residual2) {
      report.inlier_mask[i] = true;
      inCount++;
    } else {
      report.inlier_mask[i] = false;
    }
  }
  report.support.num_inliers = inCount;
  return report;
}


}


#endif // D_LRTSAC_H

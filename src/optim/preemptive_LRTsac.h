#ifndef PREEMPTIVE_LRTSAC_H
#define PREEMPTIVE_LRTSAC_H

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
template <typename Estimator, typename Sampler = RandomSampler>
class preemptive_LRTsac : public LRTsac<Estimator, LRTSupportMeasurer, Sampler>
{
public:
  struct Report {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Whether the estimation was successful.
    bool success = false;

    // The number of trials for hypothesis generation
    size_t num_trials = 0;

    // The actual number of hypoteshes generated and tested (iterations)
    size_t num_hyp = 0;

    // The support of the estimated model.
    typename LRTSupportMeasurer::Support support;

    // Number of chunks checked in total
    size_t num_chunks_checked = 0;

    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;

    // The estimated model.
    typename Estimator::M_t model;
  };

  struct hypothesis
  {
    typename Estimator::M_t model;
    int index;
    std::vector<int> ks;
    typename LRTSupportMeasurer::Support support;

    bool operator< (const hypothesis &t) const
    {
      return LRTSupportMeasurer::Compare(this->support, t.support);
    }
  };

  explicit preemptive_LRTsac(const LRToptions& options);

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
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::sampler;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::support_measurer;


private:
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::options_;

  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::makeSigmas;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::area;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::LRT;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::makeTaos;
  using LRTsac<Estimator, LRTSupportMeasurer, Sampler>::computeEps;
  std::vector<typename Estimator::M_t> check_models(
      const std::vector<typename Estimator::M_t> &models,
      const std::vector<typename Estimator::X_t>& X,
      const std::vector<typename Estimator::Y_t>& Y);
  double maxSigma(int n);

};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename Sampler>
preemptive_LRTsac<Estimator,  Sampler>::preemptive_LRTsac(
    const LRToptions& options)
  :LRTsac<Estimator, LRTSupportMeasurer, Sampler>(options, false)  {}

template <typename Estimator, typename Sampler>
std::vector<typename Estimator::M_t>
preemptive_LRTsac<Estimator, Sampler>::check_models(
    const std::vector<typename Estimator::M_t> &models,
    const std::vector<typename Estimator::X_t> &X_test,
    const std::vector<typename Estimator::Y_t> &Y_test) {

  std::vector<double> residuals(1);
  std::vector<typename Estimator::M_t> best_model;
  double bestErr = options_.sigma_max;
  int mod_index = -1;
  for(size_t i = 0; i < models.size(); ++i)
  {
    estimator.Residuals(X_test, Y_test, models[i], &residuals);
    CHECK_EQ(residuals.size(), X_test.size());
    if(residuals[0] <= bestErr)
    {
      bestErr = residuals[0];
      mod_index = i;
    }
  }

  if(mod_index >= 0)
    best_model.push_back(models[mod_index]);
  return best_model;
}

template <typename Estimator, typename Sampler>
double preemptive_LRTsac<Estimator, Sampler>::maxSigma(int n)
{
  double aux = std::exp(-options_.c/(2*n));
  switch (options_.dim) {
    case 1:
      return aux * options_.A/(2*options_.D);
      break;
    case 2:
      return std::sqrt(aux * options_.A/M_PI);
      break;
    case 3:
      return std::cbrt(aux * 0.75) * options_.D;
      break;

    default:
      return 0;
  }
}

template <typename Estimator, typename Sampler>
typename preemptive_LRTsac<Estimator, Sampler>::Report
preemptive_LRTsac<Estimator, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
  CHECK_EQ(X.size(), Y.size());

  const size_t num_samples = X.size();

  Report report;
  report.success = false;

  if (num_samples < Estimator::kMinNumSamples+1) {
    return report;
  }

  report.success = true;

  options_.B = std::min(num_samples, options_.B);
  size_t chunk_size = options_.B;

  std::vector<double> sigmas;
  makeSigmas(sigmas);
  size_t nSigmas = sigmas.size();
  const size_t n = X.size();

  std::vector<double> p_s(nSigmas);
  for(int i = 0; i < nSigmas; ++i)
    p_s[i] = area(sigmas[i]);

  std::vector<double> residuals(1);

  std::vector<double> taos;
  makeTaos(taos,n);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples+1);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples+1);

  std::vector<typename Estimator::X_t> X_mod(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_mod(Estimator::kMinNumSamples);

  std::vector<typename Estimator::X_t> X_test(1);
  std::vector<typename Estimator::Y_t> Y_test(1);

  sampler.Initialize(num_samples);

  std::vector<hypothesis> hypotheses;

  // Generate random indices for access data.
  std::vector<int> random_indices(num_samples);
  std::iota(random_indices.begin(), random_indices.end(), 0);
  Shuffle(static_cast<uint32_t>(num_samples), &random_indices);


  // In the case of most of the data being near-degenerate, our hypothesis
  // generator will fail often. Handle this, by generating fewer hypothesis
  // and allowing up to 5 hypothesis-generation trials.
  size_t max_num_hypotheses = options_.max_num_trials;
  report.num_trials = 5 * max_num_hypotheses;
  int trial = 0;
  int hypothesis_index = 0;
  for (; trial < report.num_trials && hypothesis_index < max_num_hypotheses;
       ++trial) {

    sampler.SampleXY(X, Y, &X_rand, &Y_rand);
    std::copy(X_rand.begin(), X_rand.end()-1, X_mod.begin());
    std::copy(Y_rand.begin(), Y_rand.end()-1, Y_mod.begin());
    X_test[0] = X_rand.back();
    Y_test[0] = Y_rand.back();
    std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_mod, Y_mod);
    std::vector<typename Estimator::M_t> sols = check_models(sample_models,
                                                             X_test, Y_test);
    const int num_solutions = sols.size();
    if (num_solutions <= 0) {
      continue;
    }

    if (num_solutions == 1) {
      hypothesis hyp;
      hyp.model = sols[0];
      hyp.ks.resize(nSigmas,0);
      // Copy the model generated to our local compact structure vector.
      hypotheses.push_back(hyp);
      hypotheses[hypothesis_index].index = hypothesis_index;
      ++hypothesis_index;
    } else {
      std::cerr << "Preemptive RANSAC is designed to use a hypothesis "
                   "generator that "
                   "returns only one solution.";
    }
  }
  report.num_hyp = hypothesis_index;
  report.num_trials = trial;
  int num_hypotheses_remaining = hypothesis_index;

  if (num_hypotheses_remaining == 0) {
    return report;
  }


  // We will evaluate the hypotheses using chunks of data of "chunk_size"
  // each. Each datum is thus evaluated at most once per each hypothesis.
  int datum_index = 0;
  size_t chunk_index = 0;
  while (chunk_index < num_samples)
  {
    const int current_chunk_size =
        std::min(num_samples, chunk_index + chunk_size);
    // Score all hypothesis that remain.
    for (int hyp_index = 0; hyp_index < num_hypotheses_remaining;
         ++hyp_index)
    {
      // Use data inside the current chunk.
      for (datum_index = chunk_index; datum_index < current_chunk_size;
           ++datum_index)
      {
        X_test[0] = X[random_indices[datum_index]];
        Y_test[0] = Y[random_indices[datum_index]];
        estimator.Residuals(X_test, Y_test, hypotheses[hyp_index].model,
                            &residuals);
        CHECK_EQ(residuals.size(), X_test.size());
        double err = std::sqrt(residuals[0]);
        int ind = std::max(ceil((err-options_.sigma_min)/options_. delta_sigma),
                           0.0);
        if(ind < nSigmas && ind >= 0)
          hypotheses[hyp_index].ks[ind] += 1;

      }
      ++report.num_chunks_checked;
    }

    // Set the index of the chunk at the first unevaluated point of the next
    // chunk.
    chunk_index = current_chunk_size;
    // Compute the number of hypotheses remaining to be used for the next
    // iteration. In the paper Nister proposes to decrease with a factor of
    // 2^-i. //

    int next_hypotheses_remaining =
        (num_hypotheses_remaining & 1) ? (num_hypotheses_remaining / 2)
                                       : (num_hypotheses_remaining / 2) - 1;
    //actually evaluate the hypotheses
    //int ind = current_chunk_size/options_.B-1;

    for(int hyp_index = 0; hyp_index < num_hypotheses_remaining;
        ++hyp_index )
    {
      std::vector<double> curEps;
      computeEps(curEps,hypotheses[hyp_index].ks,datum_index);
      typename LRTSupportMeasurer::Support best_support;
      for(int k = 0; k < curEps.size(); ++k)
      {
        const auto support = support_measurer.Evaluate(p_s[k], curEps[k]/*+taos[ind]*/,
                                                       sigmas[k], datum_index);
        if (support_measurer.Compare(support, best_support))
        {
          hypotheses[hyp_index].support = support;
          best_support = support;
        }

      }

    }

    if(num_hypotheses_remaining > 1)
    {
      // Sort the hypothesis remaining by their cost.
      std::nth_element(hypotheses.begin(), hypotheses.begin() +
                       next_hypotheses_remaining, hypotheses.begin() +
                       num_hypotheses_remaining);
      //    typename SupportMeasurer::Support mid_support =
      //        hypotheses[next_hypotheses_remaining-1].support;

      //    for(int i=next_hypotheses_remaining; i<num_hypotheses_remaining; ++i)
      //      if(!support_measurer.Compare(hypotheses[i].support,mid_support))
      //        next_hypotheses_remaining++;

      //    std::nth_element(hypotheses.begin(), hypotheses.begin() +
      //                     next_hypotheses_remaining, hypotheses.begin() +
      //                     num_hypotheses_remaining);

      num_hypotheses_remaining = next_hypotheses_remaining;
    }
  }

  // Get the best hypothesis.
  const auto best_hypothesis = std::min_element(hypotheses.begin(),
                                                hypotheses.begin()+
                                                num_hypotheses_remaining);


  report.support = best_hypothesis->support;
  report.model = best_hypothesis->model;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.
  double ins = 0;
  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), X.size());
  double max_residual = std::pow(report.support.sigma, 2);
  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] <= max_residual) {
      report.inlier_mask[i] = true;
      ins++;
    } else {
      report.inlier_mask[i] = false;
    }
  }
 // report.support.inRatio = ins/num_samples;
  return report;
}


}

#endif // PREEMPTIVE_LRTSAC_H

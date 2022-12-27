import pickle

import numpy as np
from mab_samplers.linear_cmab_sampler import LinearCMABSampler
from thompson_samplers.normal_inv_gamma_ts_mab import NormalInverseGammaThompsonSamplingMAB
from thompson_samplers.multiple_normal_inv_gamma_ts_cmab import MultipleNormalInverseGammaThompsonSamplinCMAB
from thompson_samplers.normal_linear_ts_cmab import NormalLinearThompsonSamplingCMAB

class LinearCMABSampler1(LinearCMABSampler):
  """Linear sampler implementing the model 
  E[R] ~ N(beta_0 + beta_1 * D + beta_2 * D * X) 
  """

  def sample_context(self, i=None):
    bit = np.random.uniform() < 0.5 
    if bit == 0:
      # arm0: E[R] = beta_0 * 1
      # arm1: E[R] = beta_0 * 1 + beta_1 * 1
      return [[1, 0, 0], [1, 1, 0]], 0
    else: 
      # arm0: E[R] = beta_0 * 1
      # arm1: E[R] = beta_0 * 1 + beta_1 * 1 + beta_2 * 1
      return [[1, 0, 0], [1, 1, 1]], 1


def main():
  # Set up (reward, context) sampler
  sigma = 0.5
  beta = [0.0, 0.3, -1.2]
  sampler = LinearCMABSampler1(num_arms=2, parameters={"beta": beta, "sigma": sigma})

  # Set up CMAB with bayesian linear regression and normal-gamma priors
  cmab_thompson_parameters = {
    "beta'": [0.0, 0.0, 0.0],
    "B": np.eye(3), 
    "a": 0.2, 
    "b": 0.2
  }

  CMAB = NormalLinearThompsonSamplingCMAB(
    num_arms=2,
    reward_sampler=sampler,
    thompson_parameters = cmab_thompson_parameters
  )


  # Set up regular MAB with normal-gamma priors for the same population. Ignores
  # the context
  mab_thompson_parameters = [
    {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1},
    {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1},    
  ]

  MAB = NormalInverseGammaThompsonSamplingMAB(
    num_arms=2,
    reward_sampler=sampler,
    thompson_parameters = mab_thompson_parameters
  )


  # Set up the new CMAB algorithm that treats each context group a separate 
  # population and applies regular MAB for each population.
  new_cmab_thompson_parameters = [
    [
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1},
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1}, 
    ],
    [
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1},
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1}, 
    ],
  ]

  NEW_CMAB = MultipleNormalInverseGammaThompsonSamplinCMAB(
    num_arms=2, 
    reward_sampler=sampler, 
    thompson_parameters=new_cmab_thompson_parameters
  )


  # Run simulation 1000 times and store the results
  simulations = []
  for i in range(1000):
    np.random.seed(i)

    # reset the priors
    CMAB.reset_thompson_parameters(cmab_thompson_parameters)
    MAB.reset_thompson_parameters(mab_thompson_parameters)
    NEW_CMAB.reset_thompson_parameters(new_cmab_thompson_parameters)

    # run the simulations
    cmab_trace = CMAB.run(iterations=1000)
    mab_trace = MAB.run(iterations=1000)
    uniform_trace = MAB.run(iterations=1000, policy="uniform")
    new_cmab_trace = NEW_CMAB.run(iterations=1000)

    simulations.append({
      "seed": i,
      "cmab_trace": cmab_trace,
      "mab_trace": mab_trace,
      "uniform_trace": uniform_trace,
      "new_cmab_trace": new_cmab_trace
    })

    if (i % 10 == 9):
      print(f"Saving simulations {i-9}-{i}")
      with open(f'simulations_object_{i-9}_{i}', 'wb') as f:
        pickle.dump(simulations, f)
        simulations = []



if __name__ == "__main__":
  main()
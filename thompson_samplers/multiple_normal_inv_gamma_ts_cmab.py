import numpy as np
from dataclasses import dataclass, field
from .thompson_sampling_mab import ThompsonSamplingMAB


@dataclass
class MultipleNormalInverseGammaThompsonSamplinCMAB(ThompsonSamplingMAB):
  """New Normal-inverse-gamma Thompson Sampling algorithm that takes into 
  accounts context

  Separate distribution is modeled for each context. In essence, this is 
  just an application of regular N-G Thompson for multiple populations
  """

  def _sample_reward_model_parameters(self):
    sigma_squared_per_context_per_arm = [
        [
          np.random.gamma(
              self.thompson_parameters[context][arm]["alpha"],
              self.thompson_parameters[context][arm]["beta"]
          ) ** (-1)
          for arm in range(self.num_arms)
        ] 
        for context in [0, 1]
    ]

    mu_per_context_per_arm = [
        [
          np.random.normal(
              self.thompson_parameters[context][arm]["mu'"],
              np.sqrt(
                  sigma_squared_per_context_per_arm[context][arm] / self.thompson_parameters[context][arm]["lambda"]
              )
          )
          for arm in range(self.num_arms)
        ]
        for context in [0, 1]
    ]

    return [
        [ 
          {
              "mu": mu_per_context_per_arm[context][arm],
              "sigma": np.sqrt(sigma_squared_per_context_per_arm[context][arm])
          }
          for arm in range(self.num_arms)
        ]
        for context in [0, 1]
    ]


  def _get_expected_arm_reward(self, arm, reward_model_parameters, context=None, context_i=None):
    return reward_model_parameters[context_i][arm]["mu"]


  def _update_thompson_parameters_from_data(self, old_data, new_data):
    chosen_arm = new_data["arm"]
    reward = new_data["reward"]
    context_i = new_data["context_i"]

    prior_mu = self.thompson_parameters[context_i][chosen_arm]["mu'"]
    prior_lambda = self.thompson_parameters[context_i][chosen_arm]["lambda"]
    prior_alpha = self.thompson_parameters[context_i][chosen_arm]["alpha"]
    prior_beta = self.thompson_parameters[context_i][chosen_arm]["beta"]
  
    self.thompson_parameters[context_i][chosen_arm] = {
        "mu'": (prior_lambda * prior_mu + reward) / (prior_lambda + 1),
        "lambda": prior_lambda + 1.0,
        "alpha": prior_alpha + 0.5, 
        "beta": prior_beta + ((prior_lambda) * ((reward-prior_mu) ** 2.0)) / (2.0 * (prior_lambda + 1.0))
    }
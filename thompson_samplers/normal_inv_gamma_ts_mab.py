import numpy as np
from dataclasses import dataclass, field
from .thompson_sampling_mab import ThompsonSamplingMAB


@dataclass
class NormalInverseGammaThompsonSamplingMAB(ThompsonSamplingMAB):
  """ A simple Normal-inverse-gamma Thompson Sampling algorithm implementation 
  self.thompson_parameters keeps track of self.num_arms sets of 
  normal-inverse-gamma distribution parameters 

  For each arm i:
    R_i ~ N(mu_i, sigma_i^2) is the reward model
    (mu_i, sigma_i^2) ~ N-inv-G(mu_i^', lambda_i, alpha_i, beta_i)
    where (mu_i^', lambda_i, alpha_i, beta_i) are the thompson parameters 
  """

  def _sample_reward_model_parameters(self):
    sigma_squared_per_arm = [
        np.random.gamma(
            self.thompson_parameters[arm]["alpha"],
            self.thompson_parameters[arm]["beta"]
        ) ** (-1)
        for arm in range(self.num_arms)
    ]

    mu_per_arm = [
        np.random.normal(
            self.thompson_parameters[arm]["mu'"],
            np.sqrt(
                sigma_squared_per_arm[arm] / self.thompson_parameters[arm]["lambda"]
            )
        )
        for arm in range(self.num_arms)
    ]

    return [ 
        {
            "mu": mu_per_arm[arm],
            "sigma": np.sqrt(sigma_squared_per_arm[arm])
        }
        for arm in range(self.num_arms)
    ]


  def _get_expected_arm_reward(self, arm, reward_model_parameters, context=None, context_i=None):
    return reward_model_parameters[arm]["mu"]


  def _update_thompson_parameters_from_data(self, new_data):
    chosen_arm = new_data["arm"]
    reward = new_data["reward"]

    prior_mu = self.thompson_parameters[chosen_arm]["mu'"]
    prior_lambda = self.thompson_parameters[chosen_arm]["lambda"]
    prior_alpha = self.thompson_parameters[chosen_arm]["alpha"]
    prior_beta = self.thompson_parameters[chosen_arm]["beta"]
  
    self.thompson_parameters[chosen_arm] = {
        "mu'": (prior_lambda * prior_mu + reward) / (prior_lambda + 1),
        "lambda": prior_lambda + 1.0,
        "alpha": prior_alpha + 0.5, 
        "beta": prior_beta + ((prior_lambda) * ((reward-prior_mu) ** 2.0)) / (2.0 * (prior_lambda + 1.0))
    }
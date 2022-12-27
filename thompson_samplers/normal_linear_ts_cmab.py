import numpy as np
from dataclasses import dataclass
from .thompson_sampling_mab import ThompsonSamplingMAB


@dataclass
class NormalLinearThompsonSamplingCMAB(ThompsonSamplingMAB):
  """ A Thompson Sampling algorithm implementation with linear contextual reward
  model with normal inverse gamma prior for the reward model parameters. 
  The formulas for the posterior taken from
  https://bookdown.org/aramir21/IntroductionBayesianEconometricsGuidedTour/linear-regression-the-conjugate-normal-normalinverse-gamma-model.html 

  For each arm i:
    R_i ~ N(beta^T * context_i, sigma^2) is the reward model
    
  beta | sigma ~ N_k(beta', sigma^2 B)
  sigma^2 ~ G^-1(a/2, b/2)

  The thompson parameters are beta', B, a and b

  k is the dimensionality of the context / the linear model

  The posterior thompson parameters from a single observation (r, c) are given by
  post_B = (B^{-1} + cc^T)^{-1}
  post_beta' = post_B(B^{-1}beta' + cy)
  post_a = a + 1
  post_b = b + r^2 + beta'^T B^{-1} beta' - post_beta'^T post_B^{-1} post_beta'
  """


  def _sample_reward_model_parameters(self):
    sigma_squared = np.random.gamma(
        self.thompson_parameters["a"] * 0.5,
        self.thompson_parameters["b"] * 0.5
    ) ** (-1.0)

    beta = np.random.multivariate_normal(
        self.thompson_parameters["beta'"], 
        sigma_squared * self.thompson_parameters["B"]
    )

    return {
        "beta": beta,
        "sigma_squared": sigma_squared
    }


  def _get_expected_arm_reward(self, arm, reward_model_parameters, context=None, context_i=None):
    return np.dot(reward_model_parameters["beta"], context)


  def _update_thompson_parameters_from_data(self, old_data, new_data):
    chosen_arm = new_data["arm"]
    reward = np.array(new_data["reward"]) # y
    context = np.array(new_data["context"]) # X

    prior_beta_p = self.thompson_parameters["beta'"]
    prior_B = self.thompson_parameters["B"]
    prior_a = self.thompson_parameters["a"]
    prior_b = self.thompson_parameters["b"]

    post_B = np.linalg.inv(np.linalg.inv(prior_B) + np.outer(context, context))
    post_beta_p = post_B @ (
        (np.linalg.inv(prior_B) @ prior_beta_p) + 
        context * reward
    )
    
    post_a = prior_a + 1.0
    post_b = (
        prior_b + 
        reward ** 2.0 + 
        prior_beta_p @ np.linalg.inv(prior_B) @ prior_beta_p -
        post_beta_p @ np.linalg.inv(post_B) @ post_beta_p
    )

    self.thompson_parameters = {
        "beta'": post_beta_p,
        "B": post_B,
        "a": post_a,
        "b": post_b
    }
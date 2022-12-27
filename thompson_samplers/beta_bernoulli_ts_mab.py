import numpy as np
from thompson_sampling_mab import ThompsonSamplingMAB


@dataclass
class BetaBernoulliThompsonSamplingMAB(ThompsonSamplingMAB):
  """ A simple Beta Bernoulli Thompson Sampling algorithm implementation 
  self.thompson_parameters keeps track of self.num_arms pairs of 
  beta distribution parameters 

  For each arm i:
    E[R] = sigma_i is the reward model
    sigma_i ~ Beta(a_i, b_i) where (a_i, b_i) are the thompson parameters 
  """

  def _sample_reward_model_parameters(self):
    return [ 
        np.random.beta(*self.thompson_parameters[arm]) 
        for arm in range(self.num_arms)
    ]


  def _get_expected_arm_reward(self, arm, reward_model_parameters, context=None, context_i=None):
    return reward_model_parameters[arm]


  def _update_thompson_parameters_from_data(self, old_data, new_data):
    chosen_arm = new_data["arm"]
    reward = new_data["reward"]


    self.thompson_parameters[chosen_arm] = [
        self.thompson_parameters[chosen_arm][0] + (reward == 1),
        self.thompson_parameters[chosen_arm][1] + (reward == 0),
    ]
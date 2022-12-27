from dataclasses import dataclass
import numpy as np
from .mab_sampler import MABSampler


@dataclass
class BernoulliMABSampler(MABSampler):
  """The parameters of this sampler are of the form
  [p_1, p_2, ...]
  A single bernoulli mean p_i is provided for each arm 
  """
  def sample(self, arm, context=None):
    return np.random.binomial(n=1, p=self.parameters[arm])

  def get_expected_reward(self, arm, context=None):
    return self.parameters[arm]
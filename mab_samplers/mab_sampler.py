from dataclasses import dataclass
import numpy as np
from typing import Any 


@dataclass
class MABSampler:
  """Base class for multi-armed-bandit samplers 
  Requires implementing sample(arm) and get_expected_reward(arm)
  Optionally can implement sample_context() for contextual 
  multi-armed bandit settings. 
  """

  num_arms: int 
  parameters: Any 
  with_context: bool = False

  def sample_context(self):
    pass

  def sample(self, arm, context=None):
    pass
  
  def get_expected_reward(self, arm, context=None):
    pass 

  def get_best_arm_expected_reward(self, contexts_per_arm=None):
    return np.max(
        [
          self.get_expected_reward(
            arm, contexts_per_arm[arm] if contexts_per_arm else None
          ) 
          for arm in range(self.num_arms)
        ]
    )
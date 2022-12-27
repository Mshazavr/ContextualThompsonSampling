from mab_sampler import MABSampler


@dataclass
class NormalMABSampler(MABSampler):
  """The parameters of this sampler are of the form
  [
    {
        "mu": mu,
        "sigma": sigma
    },
    ...
  ]
  One (mu, sigma) pair is provided for each arm
  """
  
  def sample(self, arm, context=None):
    return np.random.normal(
        loc=self.parameters[arm]['mu'], 
        scale=self.parameters[arm]['sigma']
    )
  
  def get_expected_reward(self, arm, context=None):
    return self.parameters[arm]['mu']
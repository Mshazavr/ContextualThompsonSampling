from mab_sampler import MABSampler


@dataclass
class LinearCMABSampler(MABSampler):
  """MAB sampler for contextual linear MAB problem 
  This class can't be used on its own, as the sample_context
  function still needs to be implemented

  The parameters of this sampler are of the form 
  {
    "beta": beta,
    "sigma": sigma
  }

  Note that for this setup, only one (beta, sigma) pair is provided. 
  The differences among arms are manifested thourgh the context part
  """
  with_context: bool = True

  def sample_context(self, i=None):
    pass

  def sample(self, arm, i=None, context=None):
    mu = np.dot(self.parameters["beta"], context)

    return np.random.normal(loc=mu, scale=self.parameters["sigma"])
  
  
  def get_expected_reward(self, arm, context=None):
    return np.dot(self.parameters["beta"], context)
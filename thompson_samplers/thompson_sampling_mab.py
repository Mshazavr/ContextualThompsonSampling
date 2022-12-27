from typing import Any
from mab_samplers import MABSampler

@dataclass
class ThompsonSamplingMAB:
  """ A class for implementing MAB algorithm with Thompson Sampling approach 
  It requires specifying 
    num_arms: the number of arms
    reward_sampler: an implementation of Sampler class, which has a sample method  
                  for sampling from the true data of a given arm
    thompson_parameters: the parameters defining the priors for the underlying 
                         reward model parameters
  
  An implementation of this class needs to implement 
    _sample_parameters: function for sampling reward model parameters using the 
                       thompson parameters of the class
    _get_expected_arm_reward: function for computing expected arm reward of a given arm 
                    with the given reward model parameters 
    _update_thompson_parameters_from_data: Update the thompson parameters given 
                                          the sampled data.
  """

  num_arms: int 
  reward_sampler: MABSampler
  thompson_parameters: Any = None
  observed_data: list = field(default_factory=list)


  def _sample_reward_model_parameters(self):
    pass


  def _get_expected_arm_reward(self, arm, reward_model_parameters, context=None, context_i=None):
    pass


  def _update_thompson_parameters_from_data(self, old_data, new_data):
    pass    

  
  def reset_thompson_parameters(self, thompson_parameters):
    self.thompson_parameters = thompson_parameters


  def run(self, iterations=1000, policy="thompson", verbose=False):
    """Runs Thompson sampling algorithm for given number of iterations

    Returns a trace object that contains infromation about each step of
    the algorithm (such as chosen_arm and regret)

    if policy is "uniform" instead of "thompson", then a simple uniform
    sampling will be run instead of the thompson sampling algorithm. This
    can be usseful for benchmarking purposes
    """
    assert policy in ["thompson", "uniform"]

    def select_arm(contexts_per_arm=None, context_i=None):
      if policy == "uniform":
        return np.random.randint(self.num_arms)

      reward_model_parameters = self._sample_reward_model_parameters()
      
      expected_rewards_given_parameters = [
          self._get_expected_arm_reward(
              arm=i, 
              reward_model_parameters=reward_model_parameters,
              context=contexts_per_arm[i] if contexts_per_arm else None,
              context_i=context_i if context_i is not None else None, 
          ) 
          for i in range(self.num_arms)
      ]

      return np.argmax(expected_rewards_given_parameters)

    trace = []
    
    # Main MAB loop
    for i in range(iterations):
      if verbose:
        print(f"Thompson Step: {i}")

      # Sample context
      if self.reward_sampler.with_context: 
        contexts_per_arm, context_i = self.reward_sampler.sample_context()
      else:
        contexts_per_arm, context_i = None, None

      # Select arm
      chosen_arm = select_arm(contexts_per_arm, context_i)
      context = contexts_per_arm[chosen_arm] if contexts_per_arm else None

      # Pull from the chosen arm
      observed_reward = self.reward_sampler.sample(
          arm=chosen_arm, 
          context=context
      )

      self.observed_data.append(
          {
              "arm": chosen_arm, 
              "reward": observed_reward, 
              "context": context,
              "context_i": context_i
           }
      )

      # Perform Bayesian Update
      self._update_thompson_parameters_from_data(
          self.observed_data[:-1], self.observed_data[-1]
      )

      # Record results in trace
      trace.append(
          {
              "step": i,
              "chosen_arm": chosen_arm, 
              "thompson_parameters": copy.deepcopy(self.thompson_parameters),
              "regret": (
                  self.reward_sampler.get_best_arm_expected_reward(contexts_per_arm) - 
                  self.reward_sampler.get_expected_reward(chosen_arm, context)
              ),
              #"obesrved_data": self.observed_data[-1]
          }
      )

    return trace
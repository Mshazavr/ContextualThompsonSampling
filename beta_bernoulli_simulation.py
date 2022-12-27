from mab_samplers.bernoulli_mab_sampler import BernoulliMABSampler
from thompson_samplers.beta_bernoulli_ts_mab import BetaBernoulliThompsonSamplingMAB


def main():
  sampler = BernoulliMABSampler(num_arms=2, parameters=[0.4, 0.6])
  MAB = BetaBernoulliThompsonSamplingMAB(
    num_arms=2, 
    reward_sampler=sampler, 
    thompson_parameters=[[1, 1], [1, 1]]
  )
  trace = MAB.run(iterations=1000)
  print (trace[-1])

if __name__ == "__main__":
  main()
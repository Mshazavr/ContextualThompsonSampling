from mab_samplers.normal_mab_sampler import NormalMABSampler
from thompson_samplers.normal_inv_gamma_ts_mab import NormalInverseGammaThompsonSamplingMAB

def main():
  sampler = NormalMABSampler(
    num_arms=2, 
    parameters=[
      {"mu": 0.3, "sigma": 1}, 
      {"mu": 0.6, "sigma": 1}
    ]
  )
  
  MAB = NormalInverseGammaThompsonSamplingMAB(
    num_arms=2, 
    reward_sampler=sampler, 
    thompson_parameters=[
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1},
      {"mu'": 0, 'lambda': 1.0, 'alpha': 0.1, 'beta': 0.1}, 
    ]
  )
  trace = MAB.run(iterations=1000)
  
  print(trace[-1])

if __name__ == "__main__":
  main()
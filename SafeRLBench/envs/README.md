# Environments

These classes implement simulation environments.

## Requirements

Environments have to inherit from
`SafeRLBench.EnvironmentBase`.

Further as documented in `base.py` they have to ensure
that the class requirements are met:

- Initialize parameters:  

  - state_space: Space object
  - action_space: Space object
  - horizon: Integer  
    - Note: If `_rollout` is overwritten this is
    optional.  


- Methods that have to be overwritten:
  - `_update(action)`: Compute a step
    - Parameter:
      - action:   element of action space
    - Return:
      - action:   element of action space
      - state:    element of state space
      - reward:   double
  - `_reset()`: Reset the environment to initial state  


- Methods that can be overwritten:

  - `_rollout(policy)`: Perform a rollout using policy
    - Parameter:
      - policy: callable
    - Return:
      - trace: list of (action, state, reward)-tuples

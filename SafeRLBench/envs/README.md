# Environments

Environments need to provide the following interface:

- `update(action)`:
  Computes a step
  - Parameter:
    - action:   element of action space
  - Return: (woip)
    - action:   element of action space
    - state:    element of state space
    - reward:   float
    - achieved: boolean

- `reset()`:
  Resets the environment

import numpy as np

__all__ = ['Rollout']

class Rollout(object):
    """
    Wrapper class performing rollouts on environment
    Requires environment to have _update(action) and _reset() methods
    """
    def __init__(self, environment, max_it=200):
        self.environment = environment
        self.max_it      = max_it

    def rollout(self, policy):
        """
        Performs rollout of policy on set environment
        """
        environment = self.environment
        state = environment.state
        trace = []

        environment.reset()

        for i in range(self.max_it):
            action = policy(state)
            action, state, reward, achieved = environment.update(action)
            trace.append((action, state, reward))
            if achieved: 
                break
        
        return(trace, i, achieved)

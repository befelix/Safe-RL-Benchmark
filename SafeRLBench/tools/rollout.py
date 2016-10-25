import numpy as np

__all__ = ['Rollout']

class Rollout(object):
    """
    Wrapper class performing rollouts on environment
    Requires environment to have _update(action) and _reset() methods
    """
    def __init__(self, environment, max_it=200, abort=True):
        self.environment = environment
        self.max_it      = max_it
        self.abort       = abort

    def rollout(self, policy):
        """
        Performs rollout of policy on set environment
        """
        environment = self.environment
        environment.reset()
        
        state = environment.state
        trace = []

        for i in range(self.max_it):
            action = policy(state)
            action, state, reward, achieved = environment.update(action)
            trace.append((action, np.copy(state), reward))
            if self.abort and achieved: 
                break
        
        return(trace, i, achieved)

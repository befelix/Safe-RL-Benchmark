import numpy as np

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

        for i in range(self.max_it):
            action = policy(state)
            state, reward, archieved = environment._update(action)
            trace.append((action, state, reward))
            if archieved: 
                break
        
        return(trace, i)

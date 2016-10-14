import numpy as np
from numpy.random import rand
from numpy.linalg import norm, solve

__all__ = ['LinearFDEstimator']

class LinearFDEstimator(object):
    def __init__(self, executer, environment, max_it=500, eps=0.0001, var=0.1,
                 parameter_domain=np.array([0,100]), rate=1):
        self.executer    = executer
        self.environment = environment
        self.state_dim   = environment.state.shape[0]
        self.par_dim     = self.state_dim+1

        self.parameter_domain = parameter_domain

        self.parameters = []

        self.rate   = rate
        self.eps    = eps
        self.max_it = max_it

    def _optimize(self, policy_scale):
        """           
        Parameters:
        policy_scale: array with dimension state_dim+1
        """
        pard      = self.parameter_domain
        parameter = rand(self.par_dim)*(pard[1]-pard[0]) + pard[0]

        converged = False

        print("Start Linear Finite Difference optimization:")
        print("Initial Parameters: "+str(parameter))

        for n in range(self.max_it):
            self.parameters.append(parameter)
            par_policy = lambda parameter: (lambda x: (parameter * policy_scale).dot(np.array([1,x[0],x[1]])))
            grad = self._estimate_gradient(par_policy, parameter)
            if n % 10 == 0:
                trace, i, achieved = self.executer.rollout(par_policy(parameter))
                cummulative_reward = sum([x[2] for x in trace])
                print("Run: "+str(n)+"  \tParameter: \t"+str(parameter)+"\tReward: "+str(cummulative_reward)+"\n\t\tGradient: \t"+str(grad))
            if norm(grad) < self.eps:
                converged = True
                break
                
            parameter += self.rate * grad

        return (parameter, converged)

    def _estimate_gradient(self, par_policy, parameter):
        executer = self.executer
        
        ## using forward differences
        trace, i, achieved = executer.rollout(par_policy(parameter))
        Jref = sum([x[2] for x in trace])/i
            
        dJ = np.zeros((2*self.par_dim))
        dV = np.zeros((2*self.par_dim,self.par_dim))

        for n in range(2*self.par_dim):
            variation = rand(self.par_dim) 
            variation /= norm(variation)
            trace_n, i_n, achieved = executer.rollout(par_policy(parameter + variation))
            Jn = sum([x[2] for x in trace])/i_n

            dJ[n] = Jn - Jref
            dV[n] = variation

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))
            
        return grad

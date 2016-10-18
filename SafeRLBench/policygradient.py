import numpy as np
from numpy.random import rand
from numpy.linalg import norm, solve

__all__ = ['LinearFDEstimator']

class LinearFDEstimator(object):
    def __init__(self, executer, environment, max_it=500, eps=0.01, var=0.1,
                 parameter_domain=np.array([0,100]), rate=-0.1):
        self.executer    = executer
        self.environment = environment
        self.state_dim   = environment.state.shape[0]
        self.par_dim     = self.state_dim+1

        self.parameter_domain = parameter_domain

        self.parameters = []

        self.rate   = rate
        self.eps    = eps
        self.max_it = max_it

        self.var = var

        self.best_reward = -float("inf")
        self.best_parameter = None
        self.best_goal = False

    def optimize(self, policy_scale):
        """           
        Parameters:
        policy_scale: array with dimension state_dim+1
        """
        par_policy = lambda par: (lambda x: (par * policy_scale).dot(np.array([1,x[0],x[1]])))
        
        parameter = self._initialize_parameters(par_policy)

        converged = False

        print("Start Linear Finite Difference optimization:")
        print("Initial Parameters: "+str(parameter))

        for n in range(self.max_it):
            self.parameters.append(parameter)
            grad, trace, achieved = self._estimate_gradient(par_policy, 
                                                            parameter,
                                                            2*self.par_dim)
            
            # store best result
            cummulative_reward = sum([x[2] for x in trace])

            if (cummulative_reward > self.best_reward):
                self.best_reward    = cummulative_reward
                self.best_parameter = parameter
                self.best_goal      = achieved

            # print once in a while for debugging
            if n % 10 == 0:
                print("Run: "+str(n)+"  \tParameter: \t"+str(parameter)
                        +"\tReward: "+str(cummulative_reward)
                        +"\n\t\tGradient: \t"+str(grad))

            # stop when gradient converges
            if norm(grad) < self.eps:
                converged = True
                break
                
            # update parameter
            parameter += self.rate * grad

        return (parameter, converged)

    def _estimate_gradient(self, par_policy, parameter, iterations):
        grad, trace, achieved = self._estimate_forward_gradient(par_policy, parameter, iterations)
        return(grad, trace, achieved)

    def _estimate_forward_gradient(self, par_policy, parameter, iterations):
        executer = self.executer
        
        ## using forward differences
        trace, i, achieved = executer.rollout(par_policy(parameter))
        Jref = sum([x[2] for x in trace])/i
            
        dJ = np.zeros((iterations))
        dV = np.zeros((iterations,self.par_dim))

        for n in range(iterations):
            variation = rand(self.par_dim) 
            variation *= self.var / norm(variation)
            trace_n, i_n, achieved = executer.rollout(par_policy(parameter + variation))
            Jn = sum([x[2] for x in trace])/i_n

            dJ[n] = Jref - Jn
            dV[n] = variation

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))
            
        return grad, trace, achieved

    def _estimate_central_gradient(self, par_policy, parameter, iterations):
        executer = self.executer

        trace, i, achieved = executer.rollout(par_policy(parameter))

        dJ = np.zeros((iterations))
        dV = np.zeros((iterations, self.par_dim))

        for n in range(iterations):
            variation = rand(self.par_dim)
            variation *= self.var / norm(variation)
            
            trace_n,     i_n,     _ = executer.rollout(par_policy(parameter+variation))
            trace_n_ref, i_n_ref, _ = executer.rollout(par_policy(parameter-variation))

            Jn     = sum([x[2] for x in trace_n]) / i_n
            Jn_ref = sum([x[2] for x in trace_n_ref]) / i_n_ref

            dJ[n] = Jn - Jn_ref
            dV[n] = variation

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))

        return grad, trace, achieved


    def _initialize_parameters(self, par_policy):

        pard = self.parameter_domain
    
        while True:
            parameter = rand(self.par_dim) * (pard[1] - pard[0]) + pard[0]

            grad, _, _ = self._estimate_gradient(par_policy, parameter, 2*self.par_dim)

            if (norm(grad) >= self.eps):
                return (parameter)

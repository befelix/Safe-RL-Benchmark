import numpy as np
from numpy.random import rand
from numpy.linalg import norm, solve

__all__ = ['LinearFDEstimator', 'GPOMDP']

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

    def optimize(self, policy):
        """           
        Parameters:
        """
        parameter = self._initialize_parameters(policy) 

        converged = False

        print("Start Linear Finite Difference optimization:")
        print("Initial Parameters: "+str(parameter))

        for n in range(self.max_it):
            self.parameters.append(np.copy(parameter))
            grad, trace, achieved = self._estimate_gradient(policy, parameter)
            
            # store best result
            cummulative_reward = sum([x[2] for x in trace])

            if (cummulative_reward > self.best_reward):
                self.best_reward    = cummulative_reward
                self.best_parameter = np.copy(parameter)
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

    def reset(self):
        self.parameters = []

        self.best_reward = -float("inf")
        self.best_parameter = None
        self.best_goal = False

    def _estimate_gradient(self, policy, parameter):
        grad, trace, achieved = self._estimate_central_gradient(policy, parameter)
        return(grad, trace, achieved)

    def _estimate_forward_gradient(self, policy, parameter):
        executer = self.executer
        
        ## using forward differences
        policy.setParameter(parameter)
        trace, i, achieved = executer.rollout(policy)
        Jref = sum([x[2] for x in trace])/i
            
        dJ = np.zeros((self.par_dim))
        dV = np.eye(self.par_dim)

        for n in range(self.par_dim):
            variation = dV[n] 
            
            policy.setParameter(parameter + variation)
            trace_n, i_n, achieved = executer.rollout(policy)
            
            Jn = sum([x[2] for x in trace])/i_n

            dJ[n] = Jref - Jn

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))
            
        return grad, trace, achieved

    def _estimate_central_gradient(self, policy, parameter):
        executer = self.executer

        policy.setParameter(parameter)
        trace, i, achieved = executer.rollout(policy)

        dJ = np.zeros((self.par_dim))
        dV = np.eye(self.par_dim)

        for n in range(self.par_dim):
            variation = dV[n]
            
            policy.setParameter(parameter+variation)
            trace_n,     i_n,     _ = executer.rollout(policy)
            
            policy.setParameter(parameter-variation)
            trace_n_ref, i_n_ref, _ = executer.rollout(policy)

            Jn     = sum([x[2] for x in trace_n]) / i_n
            Jn_ref = sum([x[2] for x in trace_n_ref]) / i_n_ref

            dJ[n] = Jn - Jn_ref

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))

        return grad, trace, achieved


    def _initialize_parameters(self, policy):

        pard = self.parameter_domain
    
        while True:
            parameter = rand(self.par_dim) * (pard[1] - pard[0]) + pard[0]

            grad, _, _ = self._estimate_gradient(policy, parameter)
           
            if (norm(grad) >= self.eps):
                return (parameter)


class GPOMDP(object):
    def __init__(self, executer, environment, max_it=200, eps=0.001,
                       parameter_domain=np.array([0,1]), rate=1):
        self.environment = environment
        self.executer    = executer
        self.state_dim   = environment.state.shape[0]
        self.par_dim     = self.state_dim+1
        
        self.parameter_domain = parameter_domain

        self.parameters = []

        self.rate   = rate
        self.eps    = eps
        self.max_it = max_it

        self.best_reward = -float("inf")
        self.best_parameter = None
        self.best_goal = False
        

    def optimize(self, policy):
        """           
        Parameters:
        """
        parameter = self._initialize_parameters(policy) 

        converged = False

        print("Start GPOMDP optimization:")
        print("Initial Parameters: "+str(parameter))

        for n in range(self.max_it):
            self.parameters.append(np.copy(parameter))
            grad, trace, achieved = self._estimate_gradient(policy, parameter)
            
            # store best result
            cummulative_reward = sum([x[2] for x in trace])

            if (cummulative_reward > self.best_reward):
                self.best_reward    = cummulative_reward
                self.best_parameter = np.copy(parameter)
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

    def reset(self):
        self.parameters = []

        self.best_reward = -float("inf")
        self.best_parameter = None
        self.best_goal = False

    def _estimate_gradient(self, policy, parameter):
        executer = self.executer        
        H = executer.max_it
        shape = policy.parameter_shape

        b_nom = np.zeros((H, shape))
        b_div = np.zeros((H,shape))
        b = np.zeros((H, shape))
        grad = np.zeros(shape)
        
        policy.setParameter(parameter)

        for n in range(self.max_it):
            trace, i, achieved = executer.rollout(policy)
            b_n = np.zeros((H,shape))

            for k, state in enumerate(trace):
                update = policy.log_grad(state[1], state[0]) 
                for j in range(k+1):
                    b_n[j] += update
            
            fac = n / (n+1)

            b_n = b_n**2
            b_div = fac * b_div + b_n  / (n+1)

            for k, state in enumerate(trace):
                b_nom[k] = fac * b_nom[k] + b_n[k] * state[2] / ((n+1) * (i+1))

            b = b_nom / b_div
            
            grad_update = np.zeros(shape)
            update = np.zeros(shape)
            for k, state in enumerate(trace):
                update += policy.log_grad(state[1], state[0])
                grad_update += update * (-b[k] + state[2] / (i+1))
            
            if (norm(grad_update) < self.eps):
                break
            grad += np.nan_to_num(grad_update)

        return grad / (n+1), trace, achieved
                    
    def _initialize_parameters(self, policy):

        pard = self.parameter_domain
    
        while True:
            parameter = rand(self.par_dim) * (pard[1] - pard[0]) + pard[0]

            grad, _, _ = self._estimate_gradient(policy, parameter)
            
            if (norm(grad) >= self.eps):
                return (parameter)



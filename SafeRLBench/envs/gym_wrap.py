from SafeRLBench import EnvironmentBase


class GymWrap(EnvironmentBase):

    def __init__(self, env, horizon, render=False):
        super(GymWrap, self).__init__(env.action_space, env.observation_space,
                                      horizon)
        self.environment = env
        self.render = render
        self.done = False

        self.state = env.reset()

    def _update(self, action):
        observation, reward, done, info = self.environment.step(action)
        self.state = observation
        self.done = done
        return action, observation, reward

    def _reset(self):
        self.environment.reset()
        self.done = False

    def _rollout(self, policy):
        state = self.environment.reset()
        trace = []
        for n in range(self.horizon):
            if self.render:
                self.environment.render()
            trace.append(self.update(policy(state)))
            if self.done:
                break
        return trace

"""Wrapper for OpenAI Gym."""

from SafeRLBench import EnvironmentBase


class GymWrap(EnvironmentBase):
    """Wrapper class for the OpenAI Gym.

    Attributes
    ----------
    env : gym environment
        Environment of the OpenAI Gym created by gym.make().
    horizon : integer
        Horizon for rollout.
    render : boolean
        Default False. If True simulation will be rendered during rollouts.
    """

    def __init__(self, env, horizon=100, render=False):
        """Initialize attributes."""
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
            state = self.state

            if self.done:
                break
        return trace

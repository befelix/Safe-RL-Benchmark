from SafeRLBench.envs import Quadrocopter
from SafeRLBench.policy import NonLinearQuadrocopterController
from mock import Mock

env = Quadrocopter(Mock(), Mock())
ctrl = NonLinearQuadrocopterController(reference=env.reference)

env.rollout(ctrl)

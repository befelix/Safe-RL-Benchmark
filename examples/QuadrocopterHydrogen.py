from SafeRLBench.envs import Quadrocopter
from SafeRLBench.policy import NonLinearQuadrocopterController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = Quadrocopter()
ctrl = NonLinearQuadrocopterController(reference=env.reference)

trace = env.rollout(ctrl)

states = [t[1] for t in trace]

x = [s[0] for s in states]
y = [s[1] for s in states]
z = [s[2] for s in states]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
plt.show()

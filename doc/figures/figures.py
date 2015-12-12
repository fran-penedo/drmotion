import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()

eps = 5.0
q = np.array([-15.0, 0.0])
q1 = np.array([0.0, 0.0])
l = 0.5
d = 1.0
v = q1 - q
m = np.abs(v[0]) * (- v[0] / np.abs(v[0]))
p = q + l * v

tx = eps**2 / m
ty1 = np.sqrt(eps**2 - eps**4 / m**2)
t1 = np.array([tx, ty1])
t2 = np.array([tx, -ty1])

bll = p - [0, (l * v[0] / (np.abs(tx - q[0])))* ty1]
bul = p + [0, (l * v[0] / (np.abs(tx - q[0])))* ty1]
blr = bll + d * v / np.linalg.norm(v)
bur = bul + d * v / np.linalg.norm(v)

patches = []

circle = mpatches.Circle(q1, eps, fill=False)
patches.append(circle)

obst = mpatches.Polygon(np.array([bll, bul, bur, blr]), True)
patches.append(obst)

ax.arrow(*(q.tolist() + v.tolist()), head_width=.5, head_length=0.5, ls='-', length_includes_head=True)

ax.plot(*zip(*[q, t1]), ls='--')
ax.plot(*zip(*[q, t2]), ls='--')


pcol = PatchCollection(patches, match_original=True)
ax.add_collection(pcol)

points = [q, q1, t1, t2, p]
ax.plot(*zip(*points), marker='o', ls='')

ax.annotate('$q$', xy=q, xytext=(q + [0, -.5]), fontsize=15)
ax.annotate('$q\'$', xy=q1, xytext=(q1 + [0, -.5]), fontsize=15)
ax.annotate('$p$', xy=p, xytext=(p + [-.5, -.5]), fontsize=15)
ax.annotate('$v$', xy=(q1 - [eps/2, 0]), xytext=(q1 - [eps/2, -.5]), fontsize=15)


ax.set_xlim([-20, 5])
ax.set_ylim([-10, 10])
plt.axis('equal')
plt.show()



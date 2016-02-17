from util import Tree
import numpy as np

class RRT(object):
    def __init__(self, constraints, obstacles, dx):
        self.constraints = constraints
        self.obstacles = obstacles
        self.dx = dx

    def build_tree(self, x_init, goal):
        t = Tree(x_init)
        cur = t

        while not cur.node in goal:
            x_random = self.pick_random()
            x_near_tree = nearest(t, x_random)
            x_near = x_near_tree.node
            x_new = x_near + self.dx * (x_random - x_near) / \
                np.linalg.norm(x_random - x_near)
            if self.isvalid(x_new):
                cur = Tree(x_new)
                x_near_tree.add_child(cur)

        return t, cur

    def pick_random(self):
        return np.array([np.random.uniform(c[0], c[1])
                         for c in self.constraints.constraints])

    def isvalid(self, x_new):
        return x_new in self.constraints and \
            all(x_new not in obs for obs in self.obstacles)

        return True


def nearest(t, x):
    if len(t.children) > 0:
        nearc = [nearest(c, x) for c in t.children]
        dists = [np.linalg.norm(x - nc.node) for nc in nearc]
        minc = np.argmin(dists)
        if dists[minc] < np.linalg.norm(x - t.node):
            return nearc[minc]
        else:
            return t
    else:
        return t


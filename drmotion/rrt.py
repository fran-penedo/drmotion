from util import Tree, line, contains
import numpy as np
import random

class RRT(object):
    def __init__(self, constraints, obstacles, dx):
        self.constraints = constraints
        self.obstacles = obstacles
        self.dx = dx

    def build_tree(self, x_init, goal, t_init=None, iters=-1):
        if t_init is None:
            t = Tree(x_init)
        else:
            t = t_init

        cur = t

        while not contains(goal, cur.node) and iters != 0:
            x_random = self.pick_random()
            x_near_tree = nearest(t, x_random)
            x_near = x_near_tree.node
            v = x_random - x_near
            vnorm = np.linalg.norm(v)
            if vnorm > self.dx:
                x_new = x_near + self.dx * v / vnorm
            else:
                x_new = x_random

            if isvalid(x_new, x_near, self.constraints, self.obstacles):
                cur = Tree(x_new)
                x_near_tree.add_child(cur)
                g = connect(x_new, goal, self.constraints, self.obstacles)
                if g is not None:
                    g_tree = Tree(g)
                    cur.add_child(g_tree)
                    cur = g_tree

            iters -= 1

        return t, cur

    def pick_random(self):
        if isinstance(self.constraints, list):
            return random_sample(random.choice(self.constraints))
        else:
            return random_sample(self.constraints)


def random_sample(box):
    return np.array([np.random.uniform(c[0], c[1])
                    for c in box.constraints])



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

def connect(x, goal, constraints, obstacles):
    if isinstance(goal, np.ndarray):
        for y in goal:
            if isvalid(y, x, constraints, obstacles):
                return y
    else:
        for y in goal.corners():
            if isvalid(y, x, constraints, obstacles):
                return y

def isvalid(x_new, x_near, constraints, obstacles):
    l = line(x_new, x_near)
    if isinstance(constraints, list):
        isinside = any(contains(c, x_new) for c in constraints)
    else:
        isinside = contains(constraints, x_new)
    return isinside and \
        all(not obs.contains(l) for obs in obstacles)

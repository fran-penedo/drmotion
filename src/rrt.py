from util import Tree, line
import numpy as np

class RRT(object):
    def __init__(self, constraints, obstacles, dx):
        self.constraints = constraints
        self.obstacles = obstacles
        self.dx = dx

    def build_tree(self, x_init, goal):
        t = Tree(x_init)
        cur = t

        while not goal.contains(cur.node):
            x_random = self.pick_random()
            x_near_tree = nearest(t, x_random)
            x_near = x_near_tree.node
            v = x_random - x_near
            vnorm = np.linalg.norm(v)
            if vnorm > self.dx:
                x_new = x_near + self.dx * v / vnorm
            else:
                x_new = x_random
            if self.isvalid(x_new, x_near):
                cur = Tree(x_new)
                x_near_tree.add_child(cur)
                print t.nodes()
            # else:
            #     print 'not valid'
            #     print x_new
            #     print x_near
            #     print t.nodes()

        return t, cur

    def pick_random(self):
        return np.array([np.random.uniform(c[0], c[1])
                         for c in self.constraints.constraints])

    def isvalid(self, x_new, x_near):
        l = line(x_new, x_near)
        return self.constraints.contains(x_new) and \
            all(not obs.contains(l) for obs in self.obstacles)



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


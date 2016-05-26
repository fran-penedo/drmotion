import numpy as np
from rrt import RRT
from util import Tree, cover, faces, extend, contains
from dreal import drh_connect_pair, DRHNoModel

class DRMotion(object):
    def __init__(self, constraints, obstacles, dx, eps, t_max):
        self.constraints = constraints
        self.obstacles = obstacles
        self.dx = dx
        self.eps = eps
        self.t_max = t_max
        self.iters = 50

    def build_tree(self, x_init, goal):
        t = Tree(x_init)
        cur = t

        while not contains(goal, np.array(cur.node)):
            print "DRM iteration"
            cur = self.explore(t, goal, self.iters)
            if contains(goal, np.array(cur.node)):
                break

            region = cover(np.array(t.nodes()), goal, self.eps)
            try:
                cur = self.connect_to_expansion(region, t)
            except DRMNotConnected:
                raise DRMNotConnected(t)

        return t, cur

    def connect_to_expansion(self, region, t):
        for f, d in faces(region):
            try:
                p = drh_connect_pair(f.aspoly(), extend(f, d, self.eps).aspoly(),
                                     self.constraints,
                                     self.obstacles, self.t_max, True)
                drm = DRMotion(self.constraints, self.obstacles, self.dx, self.eps,
                                self.t_max)
                # a_tree's root is p[0], last is in t
                a_tree, last = drm.build_tree(p[0], np.array(t.nodes()))
                last.make_root()
                t.find(last.node).add_children(last.children)
                b_tree = Tree(p[1])
                a_tree.add_child(b_tree)
                return b_tree
            except DRHNoModel:
                pass
            except DRMNotConnected:
                pass

        raise DRMNotConnected()

    def explore(self, t, goal, iters):
        print "Exploring"
        rrt = RRT(self.constraints, self.obstacles, self.dx)
        _, cur = rrt.build_tree(None, goal, t, iters)
        print "Finished exploring"
        return cur


class DRMNotConnected(Exception):
    def __init__(self, tree_progress):
        self.tree_progress = tree_progress


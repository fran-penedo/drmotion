import numpy as np
from rrt import RRT
import rrt
from util import Tree, cover, faces, extend, contains, Box
import util
from dreal import drh_connect_pair, DRHNoModel
import logging
logger = logging.getLogger("DRM")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
# logger.addHandler(handler)

logger.setLevel(logging.DEBUG)

class DRMotion(object):
    def __init__(self, constraints, obstacles, dx, eps, t_max, explored=None):
        self.constraints = constraints
        self.obstacles = obstacles
        self.dx = dx
        self.eps = eps
        self.t_max = t_max
        self.iters = 50
        if explored is None:
            self.explored = []
        else:
            self.explored = explored

    def build_tree(self, x_init, goal, t_init=None):
        if t_init is None:
            t = Tree(x_init)
        else:
            t = t_init
        cur = t

        while not contains(goal, np.array(cur.node)):
            # print "DRM iteration"
            cur = self.explore(t, goal, self.iters)
            if contains(goal, np.array(cur.node)):
                break

            if isinstance(goal, Box):
                exclude = goal.corners()
            else:
                exclude = goal
            if len(self.explored) > 0:
                exclude = [exclude]
                exclude.extend([x.nodes() for x in self.explored])
                exclude = np.vstack(exclude)

            region = cover(np.array(t.nodes()), exclude, self.eps)
            # util.plot_casestudy2(self.constraints, goal, self.obstacles, t, region)
            try:
                cur, explored_tree = self.connect_to_expansion(region, t, goal)
                if any(contains(goal, np.array(x))
                                for x in explored_tree.nodes()):
                       cur = explored_tree.find(next(x for x in explored_tree.nodes()
                                  if contains(goal, np.array(x))))
                       break

            except DRMNotConnected:
                raise DRMNotConnected(t)

        return t, cur

    def connect_to_expansion(self, region, t, goal):
        i = 0
        for f, d, b in faces(region):
            try:
                obs = self.obstacles + [box.aspoly() for box in region if box is not b]
                # print "Trying face " + str(i)
                p = drh_connect_pair(f.aspoly(), extend(f, d, self.eps).aspoly(),
                                     self.constraints,
                                     obs, self.t_max, True)
                util.plot_casestudy3(self.constraints, goal, self.obstacles, t, region, p)
                # print "Pair obtained"

                last = rrt.connect(p[0], np.array(t.nodes()),
                                   self.constraints, self.obstacles)
                if last is not None:
                    a_tree = Tree(p[0])
                    last = Tree(last)
                    a_tree.add_child(last)
                else:
                    a_tree = Tree(p[0])
                    for e in self.explored:
                        conn_explored = rrt.connect(p[0], np.array(e.nodes()),
                                            self.constraints, self.obstacles)
                        if conn_explored is not None:
                            ecopy = e.copy()
                            ecopy.find(conn_explored).add_child(a_tree)
                            a_tree.make_root()
                            self.explored.remove(e)
                            break

                    exp = [t] + self.explored

                    drm = DRMotion(self.constraints, self.obstacles, self.dx, self.eps,
                                    self.t_max, exp)
                    # a_tree's root is p[0], last is in t
                    a_tree, last = drm.build_tree(None, np.array(t.nodes()), a_tree)

                last.make_root()
                t.find(last.node).add_children(last.children)
                b_tree = Tree(p[1])
                a_tree.add_child(b_tree)
                return b_tree, last
            except DRHNoModel:
                pass
            except DRMNotConnected as e:
                b_tree = Tree(p[1])
                a_tree.add_child(b_tree)
                self.explored.insert(0, e.tree_progress)

            i += 1

        raise DRMNotConnected(t)

    def explore(self, t, goal, iters):
        # print "Exploring"
        rrt = RRT(self.constraints, self.obstacles, self.dx)
        _, cur = rrt.build_tree(None, goal, t, iters)
        # print "Finished exploring"
        return cur


class DRMNotConnected(Exception):
    def __init__(self, tree_progress):
        self.tree_progress = tree_progress


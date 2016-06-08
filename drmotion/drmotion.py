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

def build_tree(region, obstacles, x_init, goal, dx, eps, t_max, iters=50):
    return expand_tree(region, obstacles, Tree(x_init), goal, dx, eps, t_max,
                       [], iters)

def expand_tree(region, obstacles, t, goal, dx, eps, t_max, explored=None,
               iters=50):
    cur = t

    while not contains(goal, cur.node):
        cur = explore(region, obstacles, t, goal, dx, iters)
        if contains(goal, cur.node):
            break

        if isinstance(goal, Box):
            exclude = goal.corners()
        else:
            exclude = goal
        if len(explored) > 0:
            exclude = np.vstack([exclude] + [x.nodes() for x in explored])

        exp_region = cover(np.array(t.nodes()), exclude, eps)
        # util.plot_casestudy2(region, goal, obstacles, t, region)
        try:
            cur, explored_tree = connect_to_expansion(
                exp_region, region, obstacles, t, goal, dx, eps, t_max,
                explored, iters)
            if any(contains(goal, x) for x in explored_tree.nodes()):
                    cur = explored_tree.find(
                        next(x for x in explored_tree.nodes()
                            if contains(goal, x)))
                    break

        except DRMNotConnected:
            raise DRMNotConnected(t)

    return t, cur

def connect_to_expansion(exp_region, region, obstacles, t, goal, dx, eps,
                         t_max, explored, iters):
    for face, direction, box in faces(exp_region):
        try:
            obs = obstacles + \
                [b.aspoly() for b in exp_region if b is not box]
            a, b = drh_connect_pair(
                face.aspoly(), extend(face, direction, eps).aspoly(),
                region, obs, t_max, True)
            util.plot_casestudy3(region, goal, obstacles, t, exp_region, np.vstack([a,b]))

            last = rrt.connect(a, np.array(t.nodes()), region, obstacles)
            a_tree = Tree(a)
            if last is not None:
                last = Tree(last)
                a_tree.add_child(last)
            else:
                connect_to_explored(a_tree, explored, region, obstacles)
                # a_tree's root is a, last is in t
                a_tree, last = expand_tree(
                    region, obstacles, a_tree, np.vstack(t.nodes()), dx, eps, t_max,
                    [t] + explored, iters)

            last.make_root()
            t.find(last.node).add_children(last.children)
            b_tree = Tree(b)
            a_tree.add_child(b_tree)
            return b_tree, last
        except DRHNoModel:
            pass
        except DRMNotConnected as e:
            b_tree = Tree(b)
            a_tree.add_child(b_tree)
            explored.insert(0, e.tree_progress)

    raise DRMNotConnected(t)

def connect_to_explored(a_tree, explored, region, obstacles):
    for e in explored:
        conn_explored = rrt.connect(a_tree.node, np.array(e.nodes()),
                            region, obstacles)
        if conn_explored is not None:
            ecopy = e.copy()
            ecopy.find(conn_explored).add_child(a_tree)
            a_tree.make_root()
            explored.remove(e)
            break

def explore(region, obstacles, t, goal, dx, iters):
    # print "Exploring"
    rrt = RRT(region, obstacles, dx)
    _, cur = rrt.build_tree(None, goal, t, iters)
    # print "Finished exploring"
    return cur


class DRMNotConnected(Exception):
    def __init__(self, tree_progress):
        self.tree_progress = tree_progress


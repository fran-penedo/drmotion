from rrt import *
from util import Box, Polytope
import numpy as np
import nose.tools as nt

def rrt_test():
    cons = Box(np.array([[0, 10], [0, 10]]))
    x_init = np.array([0, 0])
    goal = Box(np.array([[9, 9.5], [9, 9.5]]))
    rrt = RRT(cons, [], 1)
    t, end = rrt.build_tree(x_init, goal)
    nt.assert_true(end.node in goal)


def rrt_obs_test():
    cons = Box(np.array([[0, 10], [0, 10]]))
    x_init = np.array([0, 0])
    goal = Box(np.array([[9, 9.5], [9, 9.5]]))
    obstacles = [Polytope(np.array([[1, 1,1], [1, 1,2.5], [1, 2.5,2.5], [1, 2.5,1]]), False),
                 Polytope(np.array([[1, 4,6], [1, 5,6], [1, 5,7], [1, 4,7]]), False)]
    rrt = RRT(cons, obstacles, 1)
    t, end = rrt.build_tree(x_init, goal)
    nt.assert_true(end.node in goal)
    nt.assert_true(np.all([[n not in obs for obs in obstacles]
                           for n in t.nodes()]))



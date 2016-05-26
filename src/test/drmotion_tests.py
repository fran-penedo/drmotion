from drmotion import *
import numpy as np
from util import Box, Polytope

def drm_test():
    cons = Box(np.array([[0, 10], [0, 10]]))
    x_init = np.array([1, 1])
    goal = Box(np.array([[9, 9.5], [1, 1.5]]))
    obstacles = [Polytope(np.array([[1, 5,0], [1, 5,9.4], [1, 6,9.4], [1, 6,0]]), False)]

    drm = DRMotion(cons, obstacles, 1, 0.5, 1)
    t, cur = drm.build_tree(x_init, goal)

drm_test.slow = 0

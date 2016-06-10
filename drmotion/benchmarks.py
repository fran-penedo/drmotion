from drmotion import *
import numpy as np
from util import Box, Polytope, grid_obsts, gen_grid
from timeit import default_timer as timer
import logging
logger = logging.getLogger("DRM")

def bench1():
    print "Bench 1"
    its = 30
    cons = Box(np.array([[0, 10], [0, 10]]))
    x_init = np.array([1, 1])
    goal = Box(np.array([[9, 9.5], [1, 1.5]]))
    obstacles = [Polytope(np.array([[1, 5,0], [1, 5,9.4], [1, 6,9.4], [1, 6,0]]), False)]

    drm_nodes = []
    rrt_nodes = []
    drm_times = []
    rrt_times = []
    for i in range(its):
        print "it {0}".format(i)
        start = timer()
        drm = DRMotion(cons, obstacles, 1, 0.5, 1)
        t, cur = drm.build_tree(x_init, goal)
        end = timer()
        drm_nodes.append(len(t.nodes()))
        drm_times.append(end - start)
        start = timer()
        rrt = RRT(cons, obstacles, 1)
        t, cur = rrt.build_tree(x_init, goal)
        rrt_nodes.append(len(t.nodes()))
        end = timer()
        rrt_times.append(end - start)

    print "drm nodes: max {0} min {1} avg {2}".format(
        max(drm_nodes), min(drm_nodes), sum(drm_nodes) / float(its))
    print "drm times: max {0} min {1} avg {2}".format(
        max(drm_times), min(drm_times), sum(drm_times) / float(its))
    print "rrt nodes: max {0} min {1} avg {2}".format(
        max(rrt_nodes), min(rrt_nodes), sum(rrt_nodes) / float(its))
    print "rrt times: max {0} min {1} avg {2}".format(
        max(rrt_times), min(rrt_times), sum(rrt_times) / float(its))

def bench2():
    print "Bench 2"
    its = 30
    cons = Box(np.array([[0, 10], [0, 10]]))
    x_init = np.array([1, 1])
    goal = Box(np.array([[9, 9.5], [1, 1.5]]))
    obstacles = [Polytope(np.array([[1, 5,0], [1, 5,10], [1, 5.3,10], [1, 5.3,0]]), False)]

    drm_nodes = []
    drm_times = []
    for i in range(its):
        print "it {0}".format(i)
        start = timer()
        try:
            drm = DRMotion(cons, obstacles, 1, 0.5, 1)
            t, cur = drm.build_tree(x_init, goal)
        except DRMNotConnected as e:
            drm_nodes.append(len(e.tree_progress.nodes()))
            pass
        end = timer()
        drm_times.append(end - start)

    print "drm nodes: max {0} min {1} avg {2}".format(
        max(drm_nodes), min(drm_nodes), sum(drm_nodes) / float(its))
    print "drm times: max {0} min {1} avg {2}".format(
        max(drm_times), min(drm_times), sum(drm_times) / float(its))

def bench3():
    logger.info("Benchmark 3: Random obstacles in grid")
    region = Box(np.array([[0, 10], [0, 10]]))
    obstacles = [o.aspoly() for o in grid_obsts(gen_grid(10, 0.2))]
    x_init = np.array([0.5, 0.5])
    goal = Box(np.array([[9, 10], [9, 10]]))

    t, cur = build_tree(region, obstacles, x_init, goal, 1, 0.5, 1)


if __name__ == "__main__":
    bench3()

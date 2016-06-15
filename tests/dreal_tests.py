from drmotion.dreal import *
import numpy as np
import nose.tools as nt
import logging
logger = logging.getLogger("DRM")

def c_minmax_test():
    v = "x"
    mm = np.array([1, 2])
    c = c_minmax(mm, v)
    nt.assert_equals(c, "(and (x >= 1) (x <= 2))")

def c_shape_box_test():
    v = "x"
    b = Box(np.array([[1, 2], [2, 3]]))
    c = c_shape(b, v)
    nt.assert_equal(c, "(and (and (x_0 >= 1) (x_0 <= 2)) (and (x_1 >= 2) (x_1 <= 3)))")

def drh_check_sat_test():
    sat = "tests/drh_sat.drh"
    unsat = "tests/drh_unsat.drh"

    with open(sat, 'r') as f:
        drh = f.read()
        res, _ = drh_check_sat(drh)
        nt.assert_true(res)

    with open(unsat, 'r') as f:
        drh = f.read()
        res, _ = drh_check_sat(drh)
        nt.assert_false(res)

@nt.nottest
def drh_connect_test():
    region = Box(np.array([[0, 10], [0, 10]]))
    init = Box(np.array([[5, 5], [2, 8]]))
    goal = Box(np.array([[6, 6], [2, 8]]))
    t_max = 4
    obst1 = Ellipsoid2D(1.0, 2.1, [5.5, 6])
    obst2 = Ellipsoid2D(1.0, 2.1, [5.5, 1])
    obst3 = Ellipsoid2D(1.0, 2.1, [5.5, 3])

    drh = drh_connect(init, goal, region, [obst1, obst2], t_max)
    res, out = drh_check_sat(drh)

    nt.assert_true(res)

    drh = drh_connect(init, goal, region, [obst1, obst3], t_max)
    res, out = drh_check_sat(drh)

    logger.debug(drh)
    logger.debug(out)

    # nt.assert_false(res)

def drh_connect_dec_test():
    region = Box(np.array([[0, 10], [0, 10]]))
    init = Box(np.array([[5, 5], [2, 8]])).aspoly()
    goal = Box(np.array([[6, 6], [2, 8]])).aspoly()
    t_max = 4
    obst1 = Box(np.array([[5, 6], [4, 6]])).aspoly()

    drh = drh_connect(init, goal, region, [obst1], t_max, "x", True)
    res, out = drh_check_sat(drh, k=10)

    # logger.debug(drh)
    # logger.debug(out)
    nt.assert_true(res)

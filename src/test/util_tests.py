from util import *
import numpy as np
import nose.tools as nt
import cddwrap as cdd

def nodes_test():
    t = Tree(1)
    t.add_child(Tree(2))
    t.add_child(Tree(3))
    t.children[0].add_child(Tree(0))
    nt.assert_set_equal(set(t.nodes()), set(range(4)))

def box_test():
    cons = np.array([[-1, 1], [2, 3]])
    b = Box(cons)
    nt.assert_true(b.contains(np.array([0, 2.5])))
    nt.assert_false(b.contains(np.array([0, 0])))
    nt.assert_false(b.contains(np.array([0, 2.5, 0])))

    x = np.array([[0, 2.5], [0, 0]])
    np.testing.assert_array_equal(b.contains(x), np.array([True, False]))

    x = np.array([[1,1], [2,2]])
    b = Box(x)
    nt.assert_true(b.contains(np.array([1,2])))

def polytope_test():
    p = Polytope([[1, 0, 0], [1, 1, 0], [1, 0, 1]], False)
    p1 = Polytope([[1, 0.5, 0.5], [1, 1.5, 1.5]], False)
    p2 = Polytope([[1, 1.2, 1.2], [1, 1.5, 1.5]], False)

    nt.assert_true(p.contains(p1))
    nt.assert_false(p.contains(p2))

def line_test():
    p = Polytope([[1, 0, 0], [1, 1, 0], [1, 0, 1]], False)
    p1 = line(np.array([0.5, 0.5]), np.array([1.5, 1.5]))
    p2 = line(np.array([1.2, 1.2]), np.array([1.5, 1.5]))

    nt.assert_true(p.contains(p1))
    nt.assert_false(p.contains(p2))

def ellipsoid2d_test():
    e = Ellipsoid2D(2.0, 3.0, [1.0, 2.0])
    x = np.array([1.0, 2.0])
    y = np.array([10.0, 10.0])

    nt.assert_true(e.contains(x))
    nt.assert_false(e.contains(y))
    np.testing.assert_array_equal(e.contains(np.vstack([x, y])),
                                  np.array([True, False]))

    # Can't check this tangent due to floating point precision errors
    # l1 = line(np.array([0, 5]), np.array([10,5]))
    l2 = line(np.array([0, 4]), np.array([10,5]))
    l3 = line(np.array([0, 6]), np.array([10,6]))

    # nt.assert_true(e.contains(l1))
    nt.assert_true(e.contains(l2))
    nt.assert_false(e.contains(l3))


def cover_test():
    n = 2
    contain = np.random.random((100, n))
    exclude = np.random.random((50, n))
    epsilon = 0.1

    boxes = cover(contain, exclude, epsilon)
    plot_boxes(boxes, contain, exclude)
    nt.assert_false(any(np.any(
        np.vstack([box.contains(exclude) for box in boxes]), axis=0)))
    nt.assert_true(all(np.any(
        np.vstack([box.contains(contain) for box in boxes]), axis=0)))

def extend_test():
    cons = np.array([[-1, 1], [2, 3], [1, 1]])
    b = Box(cons)
    d = np.array([0, 0, -1])
    e = 2
    be = extend(b, d, e)

    np.testing.assert_array_equal(be.constraints,
                                  np.array([[-1 - e/np.sqrt(3), 1 + e/np.sqrt(3)],
                                            [2 - e/np.sqrt(3), 3 + e/np.sqrt(3)],
                                            [1 - e/np.sqrt(3), 1 - e/np.sqrt(3)]]))

def conv_test():
    a = line(np.r_[0, 0], np.r_[0, 1])
    b = line(np.r_[1, 0], np.r_[1, 1])
    c = conv([a,b])
    nt.assert_equal(len(cdd.vrep_pts(c)), 4)

def exterior_test():
    region = conv_pts(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    obst = conv_pts(np.array([[.2, .2], [.4, .4], [.4, .2], [.2, .4]]))
    ext = exterior(obst, region)
    print len(ext)

def cdecomp_test():
    region = conv_pts(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    obst = conv_pts(np.array([[.2, .2], [.4, .4], [.4, .2], [.2, .4]]))
    free = cdecomp(region, [obst])
    nt.assert_equal(len(free), 4)

    obst1 = conv_pts(np.array([[.3, .3], [.5, .5], [.5, .3], [.3, .5]]))
    free = cdecomp(region, [obst, obst1])
    nt.assert_equal(len(free), 8)


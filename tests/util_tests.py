from drmotion.util import *
import numpy as np
import nose.tools as nt
import drmotion.cddwrap as cdd
import logging
logger = logging.getLogger("DRM")

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
    logger.debug("Not covered")
    for x in contain:
        covered = False
        for b in boxes:
            if b.contains(x):
                covered = True
                break
        if not covered:
            logger.debug(x)
    # plot_boxes(boxes, contain, exclude)
    nt.assert_false(any(np.any(
        np.vstack([box.contains(exclude) for box in boxes]), axis=0)))
    nt.assert_true(all(np.any(
        np.vstack([box.contains(contain) for box in boxes]), axis=0)))

def cover2_test():
    n = 2
    contain = np.random.random((100, n))
    # exclude = np.random.random((50, n))
    exclude = np.array([[0.4, 0.4],
                        [0.4, 0.45],
                        [0.4, 0.5],
                        [0.4, 0.55],
                        [0.4, 0.6],
                        [0.45, 0.6],
                        [0.5, 0.6],
                        [0.55, 0.6],
                        [0.6, 0.6],
                        [0.6, 0.55],
                        [0.6, 0.5],
                        [0.6, 0.45],
                        [0.6, 0.4],
                        [0.55, 0.4],
                        [0.5, 0.4],
                        [0.45, 0.4]
                        ])
    epsilon = 0.1

    boxes = cover(contain, exclude, epsilon)
    # plot_boxes(boxes, contain, exclude)
    nt.assert_false(any(np.any(
        np.vstack([box.contains(exclude) for box in boxes]), axis=0)))
    nt.assert_true(all(np.any(
        np.vstack([box.contains(contain) for box in boxes]), axis=0)))

def cover3_tests():
    n = 2
    epsilon = 0.1
    contain = np.array([[ 0.5       ,  0.5       ],
       [ 0.94744622,  1.39431084],
       [ 0.90613957,  2.39345736],
       [ 1.04455347,  3.38383183],
       [ 2.04011684,  3.47792533],
       [ 1.9496294 ,  4.47382292],
       [ 1.65800586,  5.4303561 ],
       [ 1.60558361,  6.4289811 ],
       [ 2.41555396,  7.01545193],
       [ 1.82727726,  7.82411164],
       [ 0.83900369,  7.671418  ],
       [ 0.62600687,  7.31020401],
       [ 1.97154287,  8.10118609],
       [ 2.83674288,  8.60261299],
       [ 3.69284694,  9.11941647],
       [ 4.7448    ,  7.00140809],
       [ 5.09835   ,  7.00140809],
       [ 5.08851893,  7.30123588],
       [ 4.39728879,  7.24856661],
       [ 3.90055765,  9.48060168],
       [ 3.75194497,  9.45170629],
       [ 2.74882117,  8.25941134],
       [ 2.57092151,  7.94427887],
       [ 2.34133237,  8.30332381],
       [ 1.42192026,  8.90576206],
       [ 2.20845538,  9.1257628 ],
       [ 1.64749373,  9.3645565 ],
       [ 2.13134068,  9.95269653],
       [ 0.52706546,  9.35211943],
       [ 1.42958835,  8.0673349 ],
       [ 2.52395647,  6.92817505],
       [ 0.80958065,  6.35891335],
       [ 1.51417317,  4.37116487],
       [ 1.84862699,  4.368818  ],
       [ 3.02958775,  3.62265719],
       [ 3.77727206,  4.28671154],
       [ 3.42810158,  5.22377076],
       [ 4.26064152,  5.77773579],
       [ 5.19126072,  5.41174685],
       [ 5.88725429,  5.89690091],
       [ 3.79174204,  3.21061486],
       [ 4.74479945,  2.9078252 ],
       [ 5.62941997,  2.44151358],
       [ 6.46049137,  1.88534784],
       [ 7.30695218,  2.41779888],
       [ 7.62016203,  3.36748282],
       [ 7.91232134,  4.32385248],
       [ 8.86900279,  4.61498924],
       [ 8.78717211,  5.61163548],
       [ 9.2729091 ,  6.4857405 ],
       [ 9.40976816,  7.47633103],
       [ 8.5947178 ,  8.05572115],
       [ 8.16895698,  7.69824023],
       [ 9.02183253,  8.95991859],
       [ 9.25650507,  6.6106104 ],
       [ 9.26437268,  5.43454337],
       [ 7.92813394,  5.70584109],
       [ 7.01635855,  6.1165303 ],
       [ 7.75900754,  5.80778185],
       [ 8.09212072,  5.76076951],
       [ 8.3977007 ,  5.84045098],
       [ 9.02236584,  5.84890392],
       [ 8.82265134,  4.65978221],
       [ 8.902775  ,  5.0923492 ],
       [ 8.869     ,  1.00121811],
       [ 9.22256   ,  1.00140405],
       [ 9.38070152,  1.52525039],
       [ 8.0226411 ,  0.46860511],
       [ 8.78936087,  1.04932736],
       [ 7.55033253,  3.85330991],
       [ 7.30583787,  4.38669934],
       [ 8.56104081,  3.30225849],
       [ 8.79252488,  3.76363934],
       [ 7.15443277,  2.09198785],
       [ 7.85643685,  2.54526923],
       [ 6.10767015,  0.99269462],
       [ 6.71301552,  0.19673169],
       [ 6.34209794,  0.17719685],
       [ 6.23611924,  1.20525523],
       [ 5.53867286,  2.17882707],
       [ 5.92671869,  2.70748591],
       [ 6.19496107,  2.5766948 ],
       [ 6.35251256,  3.36655456],
       [ 3.16911511,  3.12555925],
       [ 2.62120961,  2.78380192],
       [ 2.86464523,  1.81388486],
       [ 2.89208077,  1.2843174 ],
       [ 2.32671146,  0.61179139],
       [ 2.98804494,  1.11914196],
       [ 3.62617313,  1.78381449],
       [ 3.95830802,  0.84058259],
       [ 4.87462378,  0.45394299],
       [ 3.9797776 ,  1.24001384],
       [ 4.45668924,  1.97604044],
       [ 3.98029467,  1.84019502],
       [ 2.3859547 ,  3.29226546],
       [ 0.34813061,  4.10146348],
       [ 0.71358136,  4.62630178],
       [ 0.81397672,  4.3832048 ],
       [ 0.6947915 ,  5.49037949],
       [ 0.68526054,  5.87109046],
       [ 0.48432293,  3.29395332],
       [ 1.30786052,  2.13044212],
       [ 1.4766589 ,  2.75365559],
       [ 1.691766  ,  2.10941908],
       [ 0.7128332 ,  1.42772157],
       [ 0.18357025,  1.88472612],
       [ 0.37880043,  0.97722761],
       [ 0.73715747,  0.34879418]])
    exclude = np.array([[ 9,  9],
       [ 9, 10],
       [10,  9],
       [10, 10]], dtype=float)

    boxes = cover(contain, exclude, epsilon)
    # plot_boxes(boxes, contain, exclude)
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
    logger.debug(len(ext))

def cdecomp_test():
    region = conv_pts(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    obst = conv_pts(np.array([[.2, .2], [.4, .4], [.4, .2], [.2, .4]]))
    free = cdecomp(region, [obst])
    nt.assert_equal(len(free), 4)

    obst1 = conv_pts(np.array([[.3, .3], [.5, .5], [.5, .3], [.3, .5]]))
    free = cdecomp(region, [obst, obst1])
    nt.assert_equal(len(free), 8)


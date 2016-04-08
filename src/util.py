import numpy as np
import cddwrap as cdd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Tree(object):
    def __init__(self, node=None):
        self.children = []
        self.node = node
        self.parent = None

    def add_child(self, x):
        self.children.append(x)
        x.parent = self

    def nodes(self):
        return [self.node] + [n for nodes in [c.nodes() for c in self.children]
                              for n in nodes]


class Box(object):
    def __init__(self, constraints):
        self.constraints = constraints
        self.n = constraints.shape[0]

    def contains(self, x):
        if x.shape == (self.constraints.shape[0],):
            return np.all(x >= self.constraints[:,0]) and \
                np.all(x <= self.constraints[:,1])
        elif len(x.shape) > 1 and x.shape[1] == self.constraints.shape[0]:
            return np.logical_and(np.all(x >= self.constraints[:,0], axis=1),
                                  np.all(x <= self.constraints[:,1], axis=1))
        else:
            return False


class Polytope(cdd.CDDMatrix):

    def contains(self, x):
        if isinstance(x, Polytope):
            return not cdd.pempty(cdd.pinters(self, x))
        elif isinstance(x, np.ndarray):
            return not cdd.pempty(
                cdd.pinters(self, Polytope([np.insert(x, 0, 1)], False)))
        else:
            raise Exception("Not implemented")


def line(a, b):
    return Polytope([np.insert(x, 0, 1) for x in [a, b]], False)


class Ellipsoid2D(object):
    def __init__(self, a, b, v):
        self.a2 = a*a
        self.b2 = b*b
        self.v = v
        self.A = np.diag([1/self.a2, 1/self.b2])

    def contains(self, x):
        if isinstance(x, Polytope):
            vrep = cdd.vrep_pts(x)
            if vrep.shape == (2, 2):
                z = vrep[0] - vrep[1]
                w = vrep[1] - self.v
                print vrep
                print z
                print w
                print self.A
                return (z.dot(self.A).dot(w))**2 >= \
                    (z.dot(self.A).dot(z))*(w.dot(self.A).dot(w) - 1)
            else:
                raise Exception("Not implemented")
        elif x.shape == (2,):
            return (x[0] - self.v[0])**2 / self.a2 + \
                (x[1] - self.v[1])**2 / self.b2 <= 1
        elif len(x.shape) > 1 and x.shape[1] == 2:
            return (x[:,0] - self.v[0])**2 / self.a2 + \
                (x[:,1] - self.v[1])**2 / self.b2 <= 1
        else:
            return False


def cover(contain, exclude, epsilon):
    if len(contain) == 0:
        return []
    elif len(contain) == 1:
        return [Box(np.vstack([contain, contain]).T)]

    mins = np.min(contain, axis=0)
    maxs = np.max(contain, axis=0)
    cons = np.vstack([mins, maxs]).T
    box = Box(cons.copy())

    exclude = exclude[box.contains(exclude)]
    if len(exclude) == 0:
        return [box]

    rmins = np.min(exclude, axis=0)
    rmaxs = np.max(exclude, axis=0)
    rcons = np.vstack([rmins, rmaxs]).T

    if np.all(cons - rcons < epsilon):
        dsplit = np.argmax(cons[:,1] - cons[:,0])
        theta = (cons[dsplit, 1] + cons[dsplit, 0]) / 2
        return cover(contain[contain[:,dsplit] < theta],
                     exclude[exclude[:,dsplit] < theta], epsilon) + \
                cover(contain[contain[:,dsplit] >= theta],
                     exclude[exclude[:,dsplit] >= theta], epsilon)

    n = contain.shape[1]
    boxes = []

    for i in range(n):
        dmin = min([x for x in contain[:, i] if x > rmaxs[i]])
        cons[i] = np.array([dmin, maxs[i]])
        boxes.append(Box(cons.copy()))
        dmax = max([x for x in contain[:, i] if x < rmins[i]])
        cons[i] = np.array([mins[i], dmax])
        boxes.append(Box(cons.copy()))
        cons[i] = np.array([dmax, dmin])

    rbox = Box(rcons)
    ncontain = contain[rbox.contains(contain)]
    nexclude = exclude[rbox.contains(exclude)]
    nboxes = cover(ncontain, nexclude, epsilon)

    return boxes + nboxes


def extend(face, d, epsilon):
    cons = face.constraints.copy()
    ds = d.copy()
    ds.shape = (ds.shape[0], 1)
    dim = cons.shape[0]
    s = epsilon / np.sqrt(dim)
    # move face outwards
    cons = cons + ds * s
    # extend to corners
    v = (np.ones(cons.shape) - np.abs(ds)) * np.array([-s, s])
    cons = cons + v

    return Box(cons)


def plot_boxes(boxes, include, exclude):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(include[:,0], include[:,1], 'bo')
    ax.plot(exclude[:,0], exclude[:,1], 'ro')
    for box in boxes:
        cs = box.constraints
        x, y = cs[:,0]
        w, h = cs[:,1] - cs[:,0]
        ax.add_patch(patches.Rectangle((x,y), w, h, facecolor="green", alpha=.5))

    plt.show()

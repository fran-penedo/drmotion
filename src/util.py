import numpy as np
import cddwrap as cdd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools as it

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

    def aspoly(self):
        m = np.empty((self.n * 2, self.n + 1))
        m[0::2, 1:] = np.identity(self.n)
        m[1::2, 1:] = -np.identity(self.n)
        m[0::2, 0] = -self.constraints[:, 0]
        m[1::2, 0] = self.constraints[:, 1]
        return Polytope(m)


class Polytope(cdd.CDDMatrix):

    @staticmethod
    def fromcdd(m):
        x = Polytope([])
        x._m = m._m
        return x

    def contains(self, x):
        if isinstance(x, Polytope):
            return not cdd.pempty(cdd.pinters(self, x))
        elif isinstance(x, np.ndarray):
            return not cdd.pempty(
                cdd.pinters(self, Polytope([np.insert(x, 0, 1)], False)))
        else:
            raise Exception("Not implemented")

    @property
    def n(self):
        return self.col_size - 1


def inters(*args):
    x = cdd.pinters(*args)
    return Polytope.fromcdd(x)

def inters_to_union(p, ps):
    x = cdd.pinters_to_union(p, ps)
    return [Polytope.fromcdd(a) for a in x]

def line(a, b):
    return Polytope([np.insert(x, 0, 1) for x in [a, b]], False)

def conv_pts(m):
    return Polytope(np.insert(m, 0, 1, axis=1), False)

def conv(pols):
    return conv_pts(np.vstack([cdd.vrep_pts(p) for p in pols]))

def exterior(p, region):
    it = constr_it(len(p))
    next(it)
    exts = inters_to_union(region,
                                [Polytope(np.asarray(
                                    np.multiply(p, np.matrix(c).T)))
                                    for c in it])
    return [ext for ext in exts if len(ext.lin_set) == 0]

def constr_it(n):
    i = 0
    m = 2**n
    while i < m:
        b = binp(i, m.bit_length() - 1)
        yield np.array([-x if x == 1 else 1 for x in b])
        i += 1

def bin(s):
    return [s] if s <= 1 else bin(s >> 1) + [s & 1]

def binp(s, p):
    b = bin(s)
    return [0 for i in range(p - len(b))] + b

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

def faces(region):
    for box in region:
        for i in range(region.n):
            cons_up = box.constraints.copy()
            cons_up[i,0] = cons[i,1]
            yield Box(cons_up)
            cons_down = box.constraints.copy()
            cons_down[i,1] = cons[i,0]
            yield Box(cons_down)

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


def cdecomp(region, obsts):
    robsts = inters_to_union(region, obsts)

    vs = np.vstack([cdd.vrep_pts(obs) for obs in robsts + [region]])
    vs = vs[vs[:,0].argsort()]
    vs = vs[~np.isclose(np.r_[1, np.diff(vs, axis=0)[:,0]], 0)]

    lines = [inters(region, Polytope(np.array([[-v[0], 1, 0],
                                                    [v[0], -1, 0]])))
             for v in vs]
    free = []

    for i in range(len(lines) - 1):
        cil = conv(lines[i:i+2])
        cobsts = inters_to_union(cil, obsts)
        exts = [exterior(obs, cil) for obs in cobsts if len(obs.lin_set) == 0]
        if len(exts) == 0:
            free.append(cil)
        elif len(exts) == 1:
            free.extend(exts[0])
        else:
            frees = [inters(*tup) for tup in it.product(*exts)]
            free.extend(x for x in frees if not cdd.pempty(x))

    return free

def adj_matrix(pols):
    n = len(pols)
    m = dict()
    for i in range(n):
        for j in range(i, n):
            if i != j:
                ints = inters(pols[i], pols[j])
                if not cdd.pempty(ints):
                    m[i,j] = ints
                    m[j,i] = ints
    return m

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

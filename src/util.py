import numpy as np
import cddwrap as cdd

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

    def __contains__(self, x):
        if x.shape == (self.constraints.shape[0],):
            return np.all(x >= self.constraints[:,0]) and \
                np.all(x <= self.constraints[:,1])
        else:
            return False


class Polytope(cdd.CDDMatrix):

    def __contains__(self, x):
        if isinstance(x, Polytope):
            return not cdd.pempty(cdd.pinters(self, x))
        elif isinstance(x, np.ndarray):
            return not cdd.pempty(
                cdd.pinters(self, Polytope([np.insert(x, 0, 1)], False)))
        else:
            raise Exception("Not implemented")

def line(a, b):
    return Polytope([np.insert(x, 0, 1) for x in [a, b]], False)

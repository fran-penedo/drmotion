import cdd
import numpy as np

class CDDMatrix(object):

    def __init__(self, rows, ineq=True):
        if len(rows) > 0:
            m = cdd.Matrix(rows)
            if not ineq:
                m.rep_type = cdd.RepType.GENERATOR
                m = cdd.Polyhedron(m).get_inequalities()
            m.rep_type = cdd.RepType.INEQUALITY
            m.canonicalize()
            self._m = m

    def __len__(self):
        return len(self._m)

    def __getitem__(self, key):
        return self._m.__getitem__(key)

    def __setitem__(self, key, value):
        return self._m.__setitem__(key, value)

    def __delitem__(self, key):
        return self._m.__delitem__(key)

    def canonicalize(self):
        return self._m.canonicalize()

    @property
    def col_size(self):
        return self._m.col_size

    @property
    def row_size(self):
        return self._m.row_size

    @property
    def obj_type(self):
        return self._m.obj_type

    @obj_type.setter
    def obj_type(self, value):
        self._m.obj_type = value

    @property
    def rep_type(self):
        return self._m.rep_type

    @property
    def lin_set(self):
        return self._m.lin_set

    @rep_type.setter
    def rep_type(self, value):
        self._m.rep_type = value

    def extend(self, b):
        lin = []
        notlin = []
        for i in range(b.row_size):
            if i in b.lin_set:
                lin.append(b[i])
            else:
                notlin.append(b[i])

        self._extend(lin, True)
        self._extend(notlin, False)

    def _extend(self, rows, linear):
        if len(rows) > 0:
            self._m.extend(rows, linear)

    def copy(self):
        m = CDDMatrix([])
        m._m = self._m.copy()
        return m

    def __deepcopy__(self, memo):
        return self.copy()

    def __str__(self):
        return self._m.__str__()


def vrep(m):
    return CDDMatrix(cdd.Polyhedron(m._m).get_generators())

def vrep_pts(m):
    return np.array([v[1:] for v in vrep(m)])

def pempty(m):
    m.obj_type = cdd.LPObjType.MAX
    m.obj_func = [0 for i in range(m.col_size)]
    lp = cdd.LinProg(m._m)
    lp.solve()
    return lp.status == cdd.LPStatusType.INCONSISTENT

def pinters(a, b):
    return extend(a, b)

def extend(a, b):
    x = a.copy()
    x.extend(b)
    return x

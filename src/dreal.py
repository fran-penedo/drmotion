from util import Box, Polytope, Ellipsoid2D, cdecomp, conv, adj_matrix
import numpy as np
import cddwrap as cdd
from multimethods import multimethod
from cStringIO import StringIO
from subprocess import Popen, PIPE
import re
import os
from os import path
import tempfile

LIB_DIR = path.join(path.dirname(__file__), '../lib')
DREACH = path.join(LIB_DIR, 'dReal', 'bin', 'dReach')
DREAL_LIBS = path.join(LIB_DIR, 'dReal', 'lib')
DREACH_OPT_K = "-k "
DREACH_OPT_MODEL = "--model"

def c_minmax(mm, v):
    return "(and ({v} >= {min}) ({v} <= {max}))".format(
        v=v, min=mm[0], max=mm[1])

@multimethod(Box, str)
def c_shape(box, v):
    return "(and {0})".format(" ".join(
        [c_minmax(mm, "{0}_{1}".format(v, i))
         for i, mm in enumerate(box.constraints)]))

@multimethod(Polytope, str)
def c_shape(pol, v):
    return c_cddmatrix(pol, v)

@multimethod(cdd.CDDMatrix, str)
def c_shape(pol, v):
    return c_cddmatrix(pol, v)

def c_cddmatrix(pol, v):
    return "(and {0})".format(" ".join(
        ["({1:f} {0} ({2:f} + {3}))".format(
                "=" if i in pol.lin_set else "<=",
                0, eq[0], c_linear(eq, v))
            for i, eq in enumerate(pol)]))

def c_linear(eq, v):
    return " + ".join(["({0}_{1} * {2:f})".format(v, i - 1, eq[i])
                     for i in range(1, len(eq))])

@multimethod(Ellipsoid2D, str)
def c_shape(el, v):
    return ("(({0}_{1} - {v[0]:f})^2 / {a2:f} + " +
            "({0}_{2} - {v[1]:f})^2 / {b2:f} >= 1)").format(
                v, 0, 1, v=el.v, a2=el.a2, b2=el.b2)

def c_id_reset(v, n):
    return "(and {0})".format(" ".join(
        ["({0}_{1}' = {0}_{1})".format(v, i) for i in range(n)] +
        ["(z' = z)"]))

def drh_connect(init, goal, region, obsts, t_max, x_label="x", decomp=False):
    out = StringIO()

    for i in range(region.n):
        print >>out, "[{0[0]}, {0[1]}] {1}_{2};".format(
            region.constraints[i], x_label, i)
    print >>out, "[0, {0}] time;".format(t_max)
    print >>out, "[0, 3.15] z;"

    if decomp:
        if region.n > 2:
            raise Exception()
        r = conv([init, goal])
        dec = [init] + cdecomp(r, obsts) + [goal]
        jumps = adj_matrix(dec)
        for i, invt in enumerate(dec):
            print >>out, drh_mode(i, x_label, invt, jumps)

    else:
        #FIXME obsts
        print >>out, drh_mode(0, x_label, obsts, dict())

    print >>out, "init:"
    print >>out, "@1 {0};".format(c_shape(init, x_label))

    print >>out, "goal:"
    print >>out, "@{1} {0};".format(c_shape(goal, x_label),
                                    1 if not decomp else len(dec))

    s = out.getvalue()
    out.close()
    return s

def drh_mode(l, x_label, invt, jumps):
    out = StringIO()
    print >>out, "{" + " mode {0};".format(l + 1)

    print >>out, "invt:"
    print >>out, c_shape(invt, x_label) + ";"

    print >>out, "flow:"
    for i in range(invt.n):
        print >>out, "d/dt[{0}_{1}] = {2}(z);".format(x_label, i,
                                                    "cos" if i == 0 else "sin")
    print >>out, "d/dt[z] = 0;"

    print >>out, "jump:"
    if len(jumps) > 0:
        for k in jumps:
            if k[0] == l:
                print >>out, "{0} ==> @{1} {2};".format(
                    c_shape(jumps[k], x_label), k[1] + 1,
                    c_id_reset(x_label, invt.n))

    print >>out, " }"

    s = out.getvalue()
    out.close()
    return s


def drh_check_sat(drh, k=0):
    t = tempfile.NamedTemporaryFile(suffix=".drh", delete=False)
    t.write(drh)
    t.close()
    process = [DREACH, DREACH_OPT_K + str(k), t.name, DREACH_OPT_MODEL]
    ps = Popen(process, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, LD_LIBRARY_PATH=DREAL_LIBS))
    out, err = ps.communicate()
    outlines = out.splitlines()
    if outlines[-1].endswith("SAT"):
        return True, out
    elif outlines[-1].endswith("unsat."):
        return False, out
    else:
        print drh
        print out
        print err
        raise Exception()

def drh_connect_pair(init, goal, region, obsts, t_max, decomp=False):
    label = "x"
    drh = drh_connect(init, goal, region, obsts, t_max, label, decomp)
    sat, model = drh_check_sat(drh, 10 if decomp else 0)
    if sat:
        xlines = [l for l in model.splitlines() if l.startswith(label + "_")]
        xlinesgrp = [[l for l in xlines if l.startswith(label + "_" + str(i))]
                     for i in range(region.n)]
        return np.vstack([[parse_model_value(xlinesgrp[i][j])
                           for i in range(region.n)]
                          for j in [0, -1]])
    else:
        raise DRHNoModel()

def parse_model_value(line):
    fp = r'[-+]?\d*\.\d+|\d+'
    m = re.match(r'.*= \[(' + fp + r'), (' + fp + r')\]', line)
    return (float(m.group(1)) + float(m.group(2))) / 2.0


class DRHNoModel(Exception):
    def __init__(self):
        Exception.__init__(self)



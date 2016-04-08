from util import Box, Polytope, Ellipsoid2D
from multimethods import multimethod
from cStringIO import StringIO
from subprocess import Popen, PIPE
import os
from os import path
import tempfile

LIB_DIR = path.join(path.dirname(__file__), '../lib')
DREACH = path.join(LIB_DIR, 'dReal', 'bin', 'dReach')
DREAL_LIBS = path.join(LIB_DIR, 'dReal', 'lib')
DREACH_OPT_K0 = "-k 0"
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
    return "(and {0})".format(" ".join(
        ["({0} {1:f} (+ {2:f} {3}))".format(
                "=" if i in pol.lin_set else "<=",
                0, eq[0], c_linear(eq, v))
            for i, eq in enumerate(pol)]))

def c_linear(eq, v):
    return " ".join(["(* {0}_{1} {2:f})".format(v, i - 1, eq[i])
                     for i in range(1, len(eq))])

@multimethod(Ellipsoid2D, str)
def c_shape(el, v):
    return ("(({0}_{1} - {v[0]:f})^2 / {a2:f} + " +
            "({0}_{2} - {v[1]:f})^2 / {b2:f} >= 1)").format(
                v, 0, 1, v=el.v, a2=el.a2, b2=el.b2)

def drh_connect(init, goal, region, obsts, t_max):
    out = StringIO()
    x_label = "x"

    for i in range(region.n):
        print >>out, "[{0[0]}, {0[1]}] {1}_{2};".format(
            region.constraints[i], x_label, i)
    print >>out, "[0, {0}] time;".format(t_max)
    print >>out, "[0, 3.15] z;"

    print >>out, "{ mode 1;"

    print >>out, "invt:"
    for obst in obsts:
        print >>out, c_shape(obst, x_label) + ";"

    print >>out, "flow:"
    for i in range(region.n):
        print >>out, "d/dt[{0}_{1}] = {2}(z);".format(x_label, i,
                                                     "cos" if i == 0 else "sin")
    print >>out, "d/dt[z] = 0;"

    print >>out, "jump:"

    print >>out, " }"

    print >>out, "init:"
    print >>out, "@1 {0};".format(c_shape(init, x_label))

    print >>out, "goal:"
    print >>out, "@1 {0};".format(c_shape(goal, x_label))

    s = out.getvalue()
    out.close()
    return s

def drh_check_sat(drh):
    t = tempfile.NamedTemporaryFile(suffix=".drh", delete=False)
    t.write(drh)
    t.close()
    process = [DREACH, DREACH_OPT_K0, t.name, DREACH_OPT_MODEL]
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




def dreal_find_p(smt):
    check, out = _dreal_check_sat(smt, verbose=True)
    if check:
        r = re.compile("p([0-9]+) : \[ ENTIRE \] = \[(%s), (%s)\]" % (FP_REGEXP, FP_REGEXP))
        p_tuples = sorted([(int(i), (float(a) + float(b)) / 2)
                           for i, a, b in r.findall(out)])
        return zip(*p_tuples)[1]
    else:
        return None


def dreal_check_sat(smt):
    return _dreal_check_sat(smt)[0]


def _dreal_check_sat(smt, verbose=False):
    t = tempfile.NamedTemporaryFile(suffix=".smt2", delete=False)
    t.write(smt)
    t.close()
    process = [DREAL]
    if verbose:
        process.append("--model")
    process.append(t.name)
    ps = Popen(process, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, LD_LIBRARY_PATH=DREAL_LIBS))
    out, err = ps.communicate()
    outlines = out.splitlines()
    if outlines[0].startswith("delta-sat") or \
            outlines[-1].startswith("delta-sat") or \
            outlines[0].startswith("sat"):
        return True, out
    elif outlines[0].startswith("unsat") or outlines[-1].startswith("unsat"):
        return False, out
    else:
        print smt
        print out
        print err
        raise Exception()


def dreal_connect_smt(Xl1, Pl1, Xl2, PExcl=None, excl_slack=0):
    n = len(Xl1[0]) - 1
    if PExcl is None:
        PExcl = []
    out = StringIO()
    print >>out, "(set-logic QF_NRA)"
    for i in range(n):
        print >>out, "(declare-fun x%d () Real)" % i
        print >>out, "(declare-fun xn%d () Real)" % i

    for i in range(n * n + n):
        print >>out, "(declare-fun p%d () Real)" % i

    print >>out, "(assert %s)" % dreal_poly(Xl1, "x")
    print >>out, "(assert %s)" % dreal_poly(Xl2, "xn", 0.01)

    if len(Pl1) > 0:
        print >>out, "(assert %s)" % dreal_poly(Pl1, "p")

    if len(PExcl) > 0:
        print >>out, "(assert (not %s))" % dreal_poly(PExcl, "p", -excl_slack)

    for i in range(n):
        print >>out, "(assert (= xn%d (+ %s)))" % \
            (i,
             " ".join(["(* p%d x%d)" % (n * i + j, j) for j in range(n)] +
                      ["p%d" % (n * n + i)]))

    print >>out, "(check-sat)"
    print >>out, "(exit)"

    s = out.getvalue()
    out.close()
    return s



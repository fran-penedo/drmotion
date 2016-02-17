from util import *
import numpy as np
import nose.tools as nt

def nodes_test():
    t = Tree(1)
    t.add_child(Tree(2))
    t.add_child(Tree(3))
    t.children[0].add_child(Tree(0))
    nt.assert_set_equal(set(t.nodes()), set(range(4)))

def box_test():
    cons = np.array([[-1, 1], [2, 3]])
    b = Box(cons)
    nt.assert_true(np.array([0, 2.5]) in b)
    nt.assert_false(np.array([0, 0]) in b)
    nt.assert_false(np.array([0, 2.5, 0]) in b)





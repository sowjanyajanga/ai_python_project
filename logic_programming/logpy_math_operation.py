from logpy import run, var, fact
import logpy.assoccomm as la

# DOES NOT WORK

# Define mathematical operations
add = 'addition'
mul = 'multiplication'
from logpy import run, var, fact

# Declare that these operations are commutative
# using the facts system
fact(la.commutative, mul)
fact(la.commutative, add)
fact(la.associative, mul)
fact(la.associative, add)

# Define some variables
a, b, c = var('a'), var('b'), var('c')

# expression_orig = 3x (-2) + (1 + 2 x 3) x (-1)
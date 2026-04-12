"""Named state index constants for the triple pendulum simulation.

Provides integer indices and slices for clean, readable state vector access
without magic numbers. Mirrors the packing order in parameters/packing.py.
"""


class QIndex:
    """Configuration vector q indices (dim 4)."""
    CART = 0
    THETA1 = 1
    THETA2 = 2
    THETA3 = 3
    ANGLES = slice(1, 4)
    ALL = slice(0, 4)
    DIM = 4


class DQIndex:
    """Velocity vector dq indices (dim 4)."""
    CART = 0
    OMEGA1 = 1
    OMEGA2 = 2
    OMEGA3 = 3
    ANGULAR = slice(1, 4)
    ALL = slice(0, 4)
    DIM = 4


class XIndex:
    """Full state vector x = [q, dq] indices (dim 8)."""
    Q = slice(0, 4)
    DQ = slice(4, 8)
    CART_POS = 0
    THETA1 = 1
    THETA2 = 2
    THETA3 = 3
    CART_VEL = 4
    OMEGA1 = 5
    OMEGA2 = 6
    OMEGA3 = 7
    DIM = 8


class PIndex:
    """Parameter vector p indices (dim 13). Mirrors packing.py IDX_* constants."""
    MT = 0
    GX1 = 1; GX2 = 2; GX3 = 3
    A1 = 4; A2 = 5; A3 = 6
    B1 = 7; B2 = 8; B3 = 9
    GG1 = 10; GG2 = 11; GG3 = 12
    DIM = 13

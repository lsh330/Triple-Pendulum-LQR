from parameters.physical import PhysicalParams


def compute_derived(p: PhysicalParams) -> dict:
    return {
        "Mt": p.mc + p.m1 + p.m2 + p.m3,
        "gx1": (p.m1 / 2 + p.m2 + p.m3) * p.L1,
        "gx2": (p.m2 / 2 + p.m3) * p.L2,
        "gx3": (p.m3 / 2) * p.L3,
        "a1": (p.m1 / 3 + p.m2 + p.m3) * p.L1 ** 2,
        "a2": (p.m2 / 3 + p.m3) * p.L2 ** 2,
        "a3": (p.m3 / 3) * p.L3 ** 2,
        "b1": (p.m2 / 2 + p.m3) * p.L1 * p.L2,
        "b2": (p.m3 / 2) * p.L1 * p.L3,
        "b3": (p.m3 / 2) * p.L2 * p.L3,
        "gg1": (p.m1 / 2 + p.m2 + p.m3) * p.g * p.L1,
        "gg2": (p.m2 / 2 + p.m3) * p.g * p.L2,
        "gg3": (p.m3 / 2) * p.g * p.L3,
    }

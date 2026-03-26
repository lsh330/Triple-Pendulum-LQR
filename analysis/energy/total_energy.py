"""Compute kinetic, potential and total energy."""

from analysis.energy.kinetic_energy import compute_kinetic_energy
from analysis.energy.potential_energy import compute_potential_energy


def compute_energy(cfg, q, dq, phi1, phi2, phi3) -> dict:
    """Return dict with keys KE, PE, TE."""
    KE = compute_kinetic_energy(cfg, q, dq, phi1, phi2, phi3)
    PE = compute_potential_energy(cfg, phi1, phi2, phi3)
    TE = KE + PE
    return dict(KE=KE, PE=PE, TE=TE)

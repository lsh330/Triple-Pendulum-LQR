"""Transition graph between equilibrium configurations.

Each configuration is encoded as a 3-bit integer (D=0, U=1):
    DDD=000=0, DDU=001=1, DUD=010=2, DUU=011=3
    UDD=100=4, UDU=101=5, UUD=110=6, UUU=111=7

Two configurations are adjacent iff their Hamming distance is 1
(exactly one link changes orientation). BFS on this graph yields
a minimal-hop transition path.
"""

from collections import deque

from parameters.equilibrium import config_index  # noqa: F401 (re-exported)


# 8-node Hamming graph: 3-bit representation of each config
_CONFIG_BITS = {
    "DDD": 0b000,
    "DDU": 0b001,
    "DUD": 0b010,
    "DUU": 0b011,
    "UDD": 0b100,
    "UDU": 0b101,
    "UUD": 0b110,
    "UUU": 0b111,
}

# Reverse lookup: bit-pattern -> name
_BITS_CONFIG = {v: k for k, v in _CONFIG_BITS.items()}


def hamming_distance(config_a: str, config_b: str) -> int:
    """Return the Hamming distance between two configurations.

    Counts the number of link orientations that differ between the
    two configurations (0..3).

    Parameters
    ----------
    config_a, config_b : str
        Configuration names, e.g. "DDD", "UUU".

    Returns
    -------
    int
        Number of differing bits (0..3).
    """
    bits_a = _CONFIG_BITS[config_a]
    bits_b = _CONFIG_BITS[config_b]
    xor = bits_a ^ bits_b
    return bin(xor).count("1")


def adjacent_configs(config_name: str) -> list:
    """Return configurations reachable by flipping exactly one link orientation.

    Each node in the Hamming graph has exactly 3 neighbours.

    Parameters
    ----------
    config_name : str
        Source configuration name.

    Returns
    -------
    list of str
        Neighbouring configuration names (always length 3).
    """
    bits = _CONFIG_BITS[config_name]
    neighbours = []
    for bit_pos in range(3):
        flipped = bits ^ (1 << bit_pos)
        neighbours.append(_BITS_CONFIG[flipped])
    return neighbours


def plan_transition(source: str, target: str) -> list:
    """Plan shortest-hop path from *source* to *target* using BFS.

    Returns a list of configuration names including both endpoints.
    The length of the list equals ``hamming_distance(source, target) + 1``.

    Examples
    --------
    >>> plan_transition("DDD", "UUU")
    ['DDD', 'UDD', 'UUD', 'UUU']   # one valid answer (BFS order may vary)

    Parameters
    ----------
    source, target : str
        Start and end configuration names.

    Returns
    -------
    list of str
        Minimal-hop path, e.g. ``["DDD", "UDD", "UUD", "UUU"]``.

    Raises
    ------
    ValueError
        If either name is unknown (should not happen for valid 3-char codes).
    """
    if source not in _CONFIG_BITS:
        raise ValueError(f"Unknown configuration '{source}'")
    if target not in _CONFIG_BITS:
        raise ValueError(f"Unknown configuration '{target}'")
    if source == target:
        return [source]

    visited = {source}
    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        for neighbour in adjacent_configs(current):
            if neighbour == target:
                return path + [neighbour]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    # Should never be reached for a connected 3-bit Hamming graph
    raise ValueError(f"No path found from '{source}' to '{target}'")


def plan_transition_indices(source: str, target: str) -> list:
    """Same as :func:`plan_transition` but returns integer indices (0–7).

    Parameters
    ----------
    source, target : str
        Start and end configuration names.

    Returns
    -------
    list of int
        Integer indices for each step in the path.
    """
    path = plan_transition(source, target)
    return [config_index(name) for name in path]

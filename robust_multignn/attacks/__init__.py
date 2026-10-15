from typing import Callable

from torch import Tensor

from robust_multignn.attacks.aml import prbcd_attack_aml, uniform_attack_aml
from robust_multignn.attacks.pgd import pgd_attack
from robust_multignn.attacks.prbcd import prbcd_attack
from robust_multignn.attacks.uniform import uniform_attack

Attack = Callable[..., tuple[Tensor]]


def get_attack(attack_name: str) -> Attack:
    """Factory method to retrieve the attack function based on the provided attack name.

    Parameters
    ----------
    attack_name : str
        Name of the attack function to retrieve. Supported attack types are:
            - uniform: Perturbs the graph structure by randomly sampling new target
            nodes for a subset of edges uniformly at random.
            - pgd: Perturbs the graph structure (incidence matrix) using Projected
            Gradient Descent.
            - prbcd: Perturbs the graph structure using a Proximal Randomized Block
            Coordinate Descent method.
            - uniform_aml / pgd_aml / prbcd_aml: AML variants of the above that
            operate on edge-level mini-batches (LinkNeighborLoader) and restrict
            the perturbation budget to test-period edges (graph.split_mask).

    Returns
    -------
    Attack
        The attack function corresponding to the given attack name.

    Raises
    ------
    ValueError
        If the provided attack name is not supported.
    """
    if attack_name == "uniform":
        return uniform_attack
    if attack_name == "pgd":
        return pgd_attack
    if attack_name == "prbcd":
        return prbcd_attack
    if attack_name == "prbcd_aml":
        return prbcd_attack_aml
    if attack_name == "uniform_aml":
        return uniform_attack_aml
    else:
        raise ValueError(f"Unsupported attack type: {attack_name}")


__all__ = [
    "uniform_attack",
    "pgd_attack",
    "prbcd_attack",
    "prbcd_attack_aml",
    "uniform_attack_aml",
    "get_attack",
]

import torch


def make_undirected(inc: torch.Tensor) -> torch.Tensor:
    """Converts a directed incidence matrix to an undirected one by taking the absolute value of its entries.

    Parameters
    ----------
    inc : torch.Tensor
        [n_nodes, n_edges] A directed incidence matrix.

    Returns
    -------
    torch.Tensor
        [n_nodes, n_edges] An undirected incidence matrix.
    """
    if inc.is_sparse:
        inc = inc.coalesce()
        return torch.sparse_coo_tensor(
            inc.indices(),
            inc.values().abs(),
            inc.shape,
            device=inc.device,
        ).coalesce()
    else:
        return inc.abs()


def compute_degree_and_norm(inc: torch.Tensor) -> torch.Tensor:
    """Computes the degree and normalization factor for a given incidence matrix.

    Parameters
    ----------
    inc : torch.Tensor
        [n_nodes, n_edges] Incidence matrix of the graph.

    Returns
    -------
    torch.Tensor
        [n_nodes, 1] Normalization factors for each node.
    """
    # Compute degree of each node from the incidence matrix
    if inc.is_sparse:
        inc = inc.coalesce()
        idx, vals = inc.indices(), inc.values()
        deg = torch.zeros(inc.shape[0], device=vals.device, dtype=vals.dtype)
        deg.scatter_add_(0, idx[0], vals.abs())
    else:
        deg = inc.abs().sum(dim=1)

    # Compute normalization factor (D^-0.5) for each node, handling zero degrees
    deg_inv_sqrt = deg.pow(-0.5)
    return torch.nan_to_num(deg_inv_sqrt, posinf=0.0).view(-1, 1)


def create_inc_masks(
    inc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds filters for the incidence matrix.

    Parameters
    ----------
    inc : torch.Tensor
        [n_nodes, n_edges] Incidence matrix of the graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        sources, targets, and candidate mask tensors.
    """
    # Get number of nodes and prepare row indices for filtering
    num_nodes = inc.shape[0]
    rows = torch.arange(num_nodes, device=inc.device).view(-1, 1)

    # Get sources and targets of each edge from the incidence matrix
    sources = inc.argmin(dim=0)
    targets = inc.argmax(dim=0)

    # Candidate mask for valid perturbation targets (exclude sources and targets of each edge)
    cand_mask = ((rows != sources) & (rows != targets)).T

    return sources, targets, cand_mask


def validate_inc_pert(
    inc_pert: torch.Tensor,
    inc: torch.Tensor,
    num_pert: int,
) -> None:
    """Validates that a perturbed incidence matrix is valid.

    Parameters
    ----------
    inc_pert : torch.Tensor
        [n_nodes, n_edges] Perturbed incidence matrix.
    inc : torch.Tensor
        [n_nodes, n_edges] Original incidence matrix.
    num_pert : int
        Number of perturbations expected.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    # Prepare indices for validation checks
    sources = inc.argmin(dim=0)
    cols = torch.arange(inc.shape[1], device=inc.device)

    # Validate incidence and perturbation properties
    assert inc_pert.shape == inc.shape, "Shape should be unchanged"
    assert torch.all(inc_pert[sources, cols] == -1), "Sources should be -1"
    assert torch.all(inc_pert.sum(dim=0) == 0), "Edge columns should sum to zero"
    assert torch.sum(inc_pert != inc) == 2 * num_pert, (
        "Exactly num_pert edges should be perturbed"
    )


def tanh_margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute tanh margin loss for multi-class classification.

    Parameters
    ----------
    logits : torch.Tensor
        [n, c] Class logits.
    labels : torch.Tensor
        [n] Ground-truth class indices.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    num_samples = logits.size(0)
    sample_idx = torch.arange(logits.size(0))
    sorted = logits.argsort(-1, descending=True)
    target_class_mask = sorted.eq(labels.unsqueeze(1))
    best_non_target_class = sorted[~target_class_mask].view(num_samples, -1)[:, 0]
    margin = logits[sample_idx, labels] - logits[sample_idx, best_non_target_class]
    return torch.tanh(-margin).mean()

from typing import Any, Dict, Optional, TypeAlias, Union

import torch
from torch import Tensor

# from robust_multignn.models.line_graph_gnn import LineGraphGNNModel
from robust_multignn.models.line_graph_gnn_scatter import LineGraphGNNModel
from robust_multignn.models.multigin import MultiGIN
from robust_multignn.models.multigin_scatter import MultiGIN as MultiGINScatter
from robust_multignn.models.multipna_scatter import (
    MultiPNA,
    compute_pna_degree_histograms,
)

Model: TypeAlias = Union[LineGraphGNNModel, MultiGIN, MultiGINScatter, MultiPNA]


def add_graph_dependent_model_params(
    model_type: str,
    model_params: Dict[str, Any],
    graph: Any,
) -> Dict[str, Any]:
    """Attach parameters that must be computed from the full training graph."""
    params = dict(model_params)
    if model_type == "multipna_scatter":
        params.update(
            compute_pna_degree_histograms(graph.edge_index, int(graph.num_nodes))
        )
    return params


def create_model(
    model_type: str,
    model_params: Dict[str, Any],
    train_inds: Optional[Tensor] = None,
    val_inds: Optional[Tensor] = None,
    test_inds: Optional[Tensor] = None,
) -> Model:
    """Factory method to create a model instance based on the provided hyperparameters.

    Parameters
    ----------
    model_type : str
        The type of the model to be created. Supported model types are:
            - line_graph_gnn: Line Graph GNN for edge-level prediction
            - multigin: MultiGIN for node-level prediction
            - multigin_scatter: scatter-based MultiGIN for edge-level prediction
              (AML); requires the seed-index tensors below
            - multipna_scatter: scatter-based PNA for AML edge-level prediction;
              also requires full-training-graph degree histograms in model_params
    model_params : Dict[str, Any]
        A dictionary containing the model hyperparameters.
    train_inds : Tensor, optional
        Local edge indices of training-period seeds (required for ``line_graph_gnn``).
    val_inds : Tensor, optional
        Local edge indices of validation-period seeds (required for ``line_graph_gnn``).
    test_inds : Tensor, optional
        Local edge indices of test-period seeds (required for ``line_graph_gnn``).

    Returns
    -------
    Model
        An instance of the specified model type initialized with provided parameters.

    Raises
    ------
    ValueError
        If the specified model type is not supported.
    """
    if model_type == "line_graph_gnn":
        if train_inds is None or val_inds is None or test_inds is None:
            raise ValueError(
                "train_inds, val_inds, and test_inds are required for line_graph_gnn"
            )
        return LineGraphGNNModel(
            **model_params,
            train_inds=train_inds,
            val_inds=val_inds,
            test_inds=test_inds,
        )
    elif model_type == "multigin":
        return MultiGIN(**model_params)
    elif model_type == "multigin_scatter":
        if train_inds is None or val_inds is None or test_inds is None:
            raise ValueError(
                "train_inds, val_inds, and test_inds are required for multigin_scatter"
            )
        return MultiGINScatter(
            **model_params,
            train_inds=train_inds,
            val_inds=val_inds,
            test_inds=test_inds,
        )
    elif model_type == "multipna_scatter":
        if train_inds is None or val_inds is None or test_inds is None:
            raise ValueError(
                "train_inds, val_inds, and test_inds are required for "
                "multipna_scatter"
            )
        return MultiPNA(
            **model_params,
            train_inds=train_inds,
            val_inds=val_inds,
            test_inds=test_inds,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "LineGraphGNNModel",
    "MultiGIN",
    "MultiGINScatter",
    "MultiPNA",
    "add_graph_dependent_model_params",
    "compute_pna_degree_histograms",
    "create_model",
    "Model",
]

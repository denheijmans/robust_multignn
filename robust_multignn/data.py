from typing import Any, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import OptTensor

_ports_cpp: Any = None


def _get_ports_cpp() -> Any:
    global _ports_cpp
    if _ports_cpp is None:
        import robust_multignn_ports_cpp
        _ports_cpp = robust_multignn_ports_cpp
    return _ports_cpp


def compute_batch_ports(
    edge_index: Tensor,
    time: "Tensor | None",
    num_nodes: int,
) -> "tuple[Tensor, Tensor]":
    """Compute in/out port IDs for all edges in a batch using the C++ extension.

    Port IDs reflect the local neighborhood ordering within the batch, so they
    are correct even after graph perturbations that shift timestamp ordering.
    """
    M = edge_index.size(1)
    device = edge_index.device

    edge_src = edge_index[0].cpu().numpy().astype(np.int64, copy=False)
    edge_dst = edge_index[1].cpu().numpy().astype(np.int64, copy=False)
    t = (
        time.cpu().numpy().astype(np.float64, copy=False)
        if time is not None
        else np.zeros(M, dtype=np.float64)
    )

    cpp = _get_ports_cpp()
    in_ports_np, out_ports_np = cpp.assign_ports(edge_src, edge_dst, t, int(num_nodes))

    return torch.from_numpy(in_ports_np).to(device), torch.from_numpy(out_ports_np).to(device)


class GraphData(Data):
    """GraphData is a homogeneous graph data structure that extends PyG's Data class
    with additional properties for source and target incidence matrices, as well as
    in/out ports for edges. It also includes a method to add ego node identifiers to
    the node features.

    Parameters
    ----------
    x : OptTensor, optional
        Node feature matrix with shape :obj:`[num_nodes, num_node_features]`,
        by default :obj:`None`.
    edge_index : OptTensor, optional
        Graph connectivity in COO format with shape :obj:`[2, num_edges]`,
        by default :obj:`None`.
    edge_attr : OptTensor, optional
        Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`,
        by default :obj:`None`.
    y : OptTensor, optional
        Target labels with shape :obj:`[num_nodes]` or :obj:`[num_edges]`,
        by default :obj:`None`.
    """

    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, **kwargs)

    def _build_partial_inc(self, node_ids: Tensor) -> Tensor:
        """Builds a partial incidence matrix for the given node IDs.

        Parameters
        ----------
        node_ids : Tensor
            Node IDs for which to build the partial incidence matrix
            of shape :obj:`[num_edges]`.

        Returns
        -------
        Tensor
            Partial incidence matrix in sparse COO format
            with shape :obj:`[num_nodes, num_edges]`.
        """
        device = node_ids.device
        columns = torch.arange(self.num_edges, device=device)
        indices = torch.stack([node_ids, columns], dim=0)
        values = torch.ones(self.num_edges, dtype=torch.float32, device=device)
        size = (self.num_nodes, self.num_edges)
        return torch.sparse_coo_tensor(indices, values, size).coalesce()

    @property
    def src_edge_inc(self) -> Tensor:
        """Builds source-edge incidence matrix

        Returns
        -------
        Tensor
            Partial incidence matrix holding only source node IDs for each edge
            in sparse COO format with shape :obj:`[num_nodes, num_edges]`.
        """
        assert self.edge_index is not None
        return self._build_partial_inc(self.edge_index[0])

    @property
    def tgt_edge_inc(self) -> Tensor:
        """Builds target-edge incidence matrix

        Returns
        -------
        Tensor
            Partial incidence matrix holding only target node IDs for each edge
            in sparse COO format with shape :obj:`[num_nodes, num_edges]`.
        """
        assert self.edge_index is not None
        return self._build_partial_inc(self.edge_index[1])

    def add_ports(self, max_ports: int):
        """Adds port numbers to each edge based on its neighbors. This method computes
        the port numbers for both incoming and outgoing edges and stores them as
        properties of the GraphData object.

        Parameters
        ----------
        max_ports : int
            The maximum number of ports to assign to any node.
        """
        # Compute port numbers for incoming and outgoing edges
        adj_list_in, adj_list_out = to_adj_lists_with_times(self)
        self.in_ports = ports(self.edge_index, adj_list_in)
        self.out_ports = ports(self.edge_index.flipud(), adj_list_out)

        # Wrap port numbers into the configured port vocabulary.
        self.in_ports %= max_ports
        self.out_ports %= max_ports


# Define type alias for batches of GraphData (Python 3.9 compatible)
GraphDataBatch = Union[GraphData, Batch]


def ports(edge_index: Tensor, adj_list: dict) -> Tensor:
    """Assign port numbers to each edge based on its neighbors.

    Parameters
    ----------
    edge_index : Tensor
        Graph connectivity in COO format with shape :obj:`[2, num_edges]`.
    adj_list : dict
        Adjacency list mapping each node to its neighbors with timestamps.

    Returns
    -------
    Tensor
        Port numbers for each edge with shape :obj:`[num_edges]`.
    """
    num_edges = edge_index.size(1)
    ports = torch.zeros(num_edges, device=edge_index.device, dtype=torch.long)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1:
            continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:, [0]], return_index=True, axis=0)
        nbs_unique = a[np.sort(idx)][:, 0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u, v)] = i
    for i in range(num_edges):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        ports[i] = ports_dict[(u, v)]
    return ports


def to_adj_lists_with_times(data: GraphData) -> tuple[dict, dict]:
    """Converts edge index and timestamps to adjacency lists for incoming
    and outgoing edges.

    Parameters
    ----------
    data : GraphData
        Input graph data containing edge index and optional timestamps.

    Returns
    -------
    tuple[dict, dict]
        Adjacency lists for incoming and outgoing edges with timestamps.
    """
    timestamps = (
        data.time.view(-1, 1)
        if data.time is not None
        else torch.zeros(
            (data.num_edges, 1),
            device=data.edge_index.device,
            dtype=torch.float32,
        )
    )
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(data.num_nodes)])
    adj_list_in = dict([(i, []) for i in range(data.num_nodes)])
    for u, v, t in edges:
        u, v, t = int(u), int(v), float(t)
        adj_list_out[u].append((v, t))
        adj_list_in[v].append((u, t))
    return adj_list_in, adj_list_out


def to_adj_lists(edge_index: Tensor, num_nodes: int) -> tuple[dict, dict]:
    """Converts edge index to adjacency lists for incoming and outgoing edges.

    Parameters
    ----------
    edge_index : Tensor
        Graph connectivity in COO format with shape :obj:`[2, num_edges]`.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[dict, dict]
        Adjacency lists for incoming and outgoing edges.
    """
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u, v in edge_index.T:
        u, v = int(u), int(v)
        adj_list_out[u].append(v)
        adj_list_in[v].append(u)
    return adj_list_in, adj_list_out


def build_neighbor_loader(
    data: GraphData,
    batch_size: int,
    num_neighbors: list[int],
    shuffle: bool,
) -> NeighborLoader:
    """Builds neighbor sampling data loader using both forward and reverse edges.

    Parameters
    ----------
    data : GraphData
        The input graph data to build the neighbor loader from.
    batch_size : int
        The size of each batch to sample.
    num_neighbors : list[int]
        The number of neighbors to sample at each hop.
    shuffle : bool
        Whether to shuffle the data.

    Returns
    -------
    NeighborLoader
        The built neighbor sampling data loader.
    """
    # Convert to HeteroData with reverse edges
    hetero_data = data.to_heterogeneous(
        node_type_names=["n"],
        edge_type_names=[("n", "fwd", "n")],
    )
    hetero_data[("n", "rev", "n")].edge_index = data.edge_index.flipud()

    # Transform sampled batches back to homogeneous GraphData
    def to_homogeneous_batch(hetero_batch: HeteroData) -> GraphData:
        node_kwargs = dict(hetero_batch["n"].items())
        edge_kwargs = dict(hetero_batch[("n", "fwd", "n")].items())
        kwargs = {**node_kwargs, **edge_kwargs}
        return GraphData(**kwargs)

    # Build neighbor sampling data loader
    return NeighborLoader(
        hetero_data,
        num_neighbors={
            ("n", "fwd", "n"): num_neighbors,
            ("n", "rev", "n"): num_neighbors,
        },
        input_nodes=("n", torch.arange(data.num_nodes)),
        transform=to_homogeneous_batch,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )


def build_hetero_data(data: GraphData) -> HeteroData:
    """Convert a homogeneous :class:`GraphData` to a bidirectional
    :class:`HeteroData` suitable for use with :class:`~torch_geometric.loader.LinkNeighborLoader`.

    Two edge types are registered on the single node type ``"n"``:

    * ``("n", "fwd", "n")`` — original directed edges, carrying all edge
      attributes (``edge_attr``, ``y``, ``timestamps``).
    * ``("n", "rev", "n")`` — same edges with src/dst flipped, carrying
      ``timestamps`` so the sampler can follow backward neighbours.

    Parameters
    ----------
    data : GraphData
        Source homogeneous graph.

    Returns
    -------
    HeteroData
        Bidirectional heterogeneous graph.
    """
    fwd = ("n", "fwd", "n")
    rev = ("n", "rev", "n")

    hetero_data = HeteroData()
    hetero_data["n"].x = data.x

    hetero_data[fwd].edge_index = data.edge_index
    hetero_data[fwd].edge_attr = data.edge_attr
    if data.get("y") is not None:
        hetero_data[fwd].y = data.y
    if data.get("timestamps") is not None:
        hetero_data[fwd].timestamps = data.timestamps
    if data.get("split_mask") is not None:
        hetero_data[fwd].split_mask = data.split_mask
    if data.get("simp_edge_batch") is not None:
        hetero_data[fwd].simp_edge_batch = data.simp_edge_batch

    hetero_data[rev].edge_index = data.edge_index.flipud()

    return hetero_data


class AddEgoIdsForLinkNeighbor(BaseTransform):
    r"""Append one-hot ego identifiers keyed by seed edge to a link batch.

    Each seed edge (a column of ``edge_label_index``) is assigned its own column
    in the ego block, and *both endpoints* of that seed edge are marked with a 1
    in it. This tags, per seed edge, the node pair being classified -- letting
    the message-passing layers distinguish the endpoints of one seed edge from
    those of another -- while keeping the block width bounded by the number of
    seed edges (``<= batch_size``) rather than the number of seed nodes.

    A node shared by several seed edges is marked in each of their columns;
    non-seed nodes stay all-zero.

    Parameters
    ----------
    ego_dim : int
        Width of the ego block. Must be ``>=`` the number of seed edges per batch
        (i.e. ``>= batch_size``), otherwise a :class:`ValueError` is raised.
    """

    def __init__(self, ego_dim: int):
        self.ego_dim = ego_dim

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        is_hetero = isinstance(data, HeteroData)
        if is_hetero:
            store = data["n"]
            edge_label_index = data["n", "fwd", "n"].edge_label_index
        else:
            store = data
            edge_label_index = data.edge_label_index

        x = store.x
        device = x.device
        num_seed_edges = edge_label_index.size(1)

        if num_seed_edges > self.ego_dim:
            raise ValueError(
                f"ego_dim ({self.ego_dim}) must be >= the number of seed edges "
                f"per batch ({num_seed_edges}); set ego_dim >= batch_size."
            )

        ego = torch.zeros((x.size(0), self.ego_dim), device=device, dtype=x.dtype)
        cols = torch.arange(num_seed_edges, device=device)
        ego[edge_label_index[0], cols] = 1.0  # source endpoint of each seed edge
        ego[edge_label_index[1], cols] = 1.0  # target endpoint of each seed edge
        store.x = torch.cat([x, ego], dim=1)
        return data


class AddSingleEgoIdForLinkNeighbor(BaseTransform):
    r"""Append a single 0/1 ego indicator marking a link batch's seed nodes.

    Adds exactly **one** feature column, set to 1 for every node that is an
    endpoint of a seed edge (``edge_label_index``) and 0 otherwise. This is the
    standard Multi-GNN / AML ego identifier: constant width (1) regardless of
    batch size, so it avoids the memory blow-up of the per-seed-edge one-hot in
    :class:`AddEgoIdsForLinkNeighbor`. The consuming model must size its node
    encoder for ``in_dim + 1`` (i.e. configure ``ego_dim = 1``).
    """

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        is_hetero = isinstance(data, HeteroData)
        if is_hetero:
            store = data["n"]
            edge_label_index = data["n", "fwd", "n"].edge_label_index
        else:
            store = data
            edge_label_index = data.edge_label_index

        x = store.x
        ids = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        seed_nodes = torch.unique(edge_label_index.reshape(-1)).to(x.device)
        ids[seed_nodes] = 1.0
        store.x = torch.cat([x, ids], dim=1)
        return data


def build_edge_neighbor_loader(
    data: GraphData,
    num_neighbors: list = None,
    batch_size: int = 512,
    shuffle: bool = False,
    bidirectional: bool = True,
    ego_dim: int = 0,
    seed_mask: Tensor | None = None,
    **kwargs,
) -> LinkNeighborLoader:
    """Build a :class:`~torch_geometric.loader.LinkNeighborLoader` for
    edge-level classification tasks (e.g. AML fraud detection).

    Parameters
    ----------
    data : GraphData
        The full (possibly cumulative) graph for context.
    num_neighbors : list
        Number of neighbours to sample per hop.
    batch_size : int
        Number of target edges per mini-batch.
    shuffle : bool
        Whether to shuffle the target edges each epoch.
    bidirectional : bool
        If ``True`` (default), converts ``data`` to a bidirectional
        :class:`HeteroData` with forward and reverse edge types so the
        sampler fetches neighbours in both directions. Each sampled batch is
        collapsed back to a homogeneous :class:`GraphData` via a transform.
        If ``False``, ``data`` is passed directly to
        :class:`~torch_geometric.loader.LinkNeighborLoader` as a plain
        homogeneous graph; no transform is applied.
    ego_dim : int
        If ``> 0``, append a single 0/1 ego indicator to each batch's node
        features via :class:`AddSingleEgoIdForLinkNeighbor` (marks the endpoints
        of the batch's seed edges). The indicator is always width 1, so the
        consuming model should set ``ego_dim = 1``. ``0`` (default) disables it,
        which is the right setting for models that do not consume ego ids.
    seed_mask : Tensor, optional
        Boolean edge mask selecting the seed (label) edges. Defaults to
        ``data.split_mask``. A subset of the split (e.g. only the illicit
        edges an attacker controls) yields a loader whose ``input_id`` indexes
        that subset, so the model must be given the matching seed indices
        (``mask_to_index(seed_mask)``) under the split name it is called with.

    Returns
    -------
    LinkNeighborLoader
        The edge-level neighbour-sampling data loader.
    """
    kwargs.pop("eval_inds", None)
    eval_inds = data.split_mask if seed_mask is None else seed_mask

    ego_transform = AddSingleEgoIdForLinkNeighbor() if ego_dim else None

    if bidirectional:
        fwd = ("n", "fwd", "n")
        rev = ("n", "rev", "n")

        hetero_data = build_hetero_data(data)

        if eval_inds is not None:
            edge_label_index = (fwd, data.edge_index[:, eval_inds])
            edge_label = data.y[eval_inds]
        else:
            edge_label_index = (fwd, data.edge_index)
            edge_label = data.y

        # Collapse each hetero batch back to homogeneous GraphData;
        # edge_label and input_id are automatically placed on the fwd edge type
        # by LinkNeighborLoader and will be included via edge_kwargs
        def to_homogeneous_batch(hetero_batch: HeteroData) -> GraphData:
            node_kwargs = dict(hetero_batch["n"].items())
            edge_kwargs = dict(hetero_batch[fwd].items())
            batch = GraphData(**{**node_kwargs, **edge_kwargs})
            return ego_transform(batch) if ego_transform is not None else batch

        return LinkNeighborLoader(
            hetero_data,
            num_neighbors={fwd: num_neighbors, rev: num_neighbors},
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=to_homogeneous_batch,
            **kwargs,
        )
    else:
        if eval_inds is not None:
            edge_label_index = data.edge_index[:, eval_inds]
            edge_label = data.y[eval_inds]
        else:
            edge_label_index = data.edge_index
            edge_label = data.y

        return LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=ego_transform,
            **kwargs,
        )


def augment_with_ego_ids(
    x: Tensor,
    batch_size: int,
    ego_dim: int,
) -> Tensor:
    """Augments node features with ego node identifiers for each batch.

    Parameters
    ----------
    x : Tensor
        Node feature matrix with shape :obj:`[num_nodes, num_node_features]`.
    batch_size : int
        Number of nodes in each batch.
    ego_dim : int
        Dimensionality of the ego node identifiers.

    Returns
    -------
    Tensor
        Augmented node feature matrix
        with shape :obj:`[num_nodes, num_node_features + ego_dim]`.
    """
    num_nodes = x.size(0)
    ego_ids = torch.zeros((num_nodes, ego_dim), device=x.device, dtype=torch.float32)
    for i in range(batch_size):
        ego_ids[i, i % ego_dim] = 1.0
    return torch.cat((x, ego_ids), dim=1)

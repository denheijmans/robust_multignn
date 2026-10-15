import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm


def _mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )


class MultiGINLayer(nn.Module):
    """Single GINe layer using incidence matrices; H and Z must already be in hidden_dim space."""

    def __init__(
        self,
        hidden_dim: int,
        eps: float,
        train_eps: bool,
        reverse_mp: bool,
        edge_updates: bool,
    ):
        super().__init__()
        self.reverse_mp = reverse_mp
        self.edge_updates = edge_updates

        if train_eps:
            self.eps_fwd = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps_fwd", torch.tensor(eps))

        self.mlp_fwd = _mlp(hidden_dim)
        self.bn = BatchNorm(hidden_dim)

        if reverse_mp:
            if train_eps:
                self.eps_rev = nn.Parameter(torch.tensor(eps))
            else:
                self.register_buffer("eps_rev", torch.tensor(eps))
            self.mlp_rev = _mlp(hidden_dim)

        if edge_updates:
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, H: Tensor, Z: Tensor, B_src: Tensor, B_dst: Tensor):
        """Compute one GINe message-passing step.

        Args:
            H: Node features of shape (V, hidden_dim).
            Z: Edge features of shape (E, hidden_dim).
            B_src: Source incidence matrix (V, E), sparse COO.
            B_dst: Destination incidence matrix (V, E), sparse COO.

        Returns:
            Tuple (H_new, Z_new).
        """
        m_fwd = F.relu(torch.sparse.mm(B_src.T, H) + Z)
        H_fwd = self.mlp_fwd((1 + self.eps_fwd) * H + torch.sparse.mm(B_dst, m_fwd))

        if self.reverse_mp:
            m_rev = F.relu(torch.sparse.mm(B_dst.T, H) + Z)
            H_rev = self.mlp_rev((1 + self.eps_rev) * H + torch.sparse.mm(B_src, m_rev))
            H_combined = (H_fwd + H_rev) / 2
        else:
            H_combined = H_fwd

        H_new = (H + F.relu(self.bn(H_combined))) / 2

        if self.edge_updates:
            edge_input = torch.cat(
                [torch.sparse.mm(B_src.T, H_new), torch.sparse.mm(B_dst.T, H_new), Z],
                dim=-1,
            )
            Z_new = Z + self.edge_mlp(edge_input) / 2
        else:
            Z_new = Z

        return H_new, Z_new


class MultiGIN(nn.Module):
    """Multi-layer GINe model using incidence matrices; faithful to Egressy et al. (AAAI 2024)."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        eps: float,
        train_eps: bool,
        out_dim: int,
        ego_dim: int,
        port_vocab_size: int,
        port_emb_dim: int,
        dropout: float,
        reverse_mp: bool,
        edge_updates: bool,
    ):
        super().__init__()
        self.ego_dim = ego_dim
        self.port_vocab_size = port_vocab_size
        use_ports = bool(port_vocab_size) and bool(port_emb_dim)
        self.use_ports = use_ports
        self.dropout = dropout

        self.port_emb = None
        if use_ports:
            self.port_emb = nn.Embedding(port_vocab_size, port_emb_dim)

        self.node_emb = nn.Linear(in_dim + ego_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim + (2 * port_emb_dim if use_ports else 0), hidden_dim)
        self.layers = nn.ModuleList(
            [
                MultiGINLayer(hidden_dim, eps, train_eps, reverse_mp, edge_updates)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: Tensor,
        src_edge_inc: Tensor,
        tgt_edge_inc: Tensor,
        edge_attr: Tensor,
        in_ports: Tensor,
        out_ports: Tensor,
    ) -> Tensor:
        """Run a forward pass and return per-node class logits.

        Args:
            x: Node features of shape (V, in_node_dim + ego_dim).
            src_edge_inc: Source incidence matrix (V, E), sparse COO.
            tgt_edge_inc: Destination incidence matrix (V, E), sparse COO.
            edge_attr: Edge features of shape (E, in_edge_dim).
            in_ports: Incoming port ids of shape (E,).
            out_ports: Outgoing port ids of shape (E,).

        Returns:
            Logit tensor of shape (V, out_dim).
        """
        if self.use_ports:
            in_emb = self.port_emb(in_ports)
            out_emb = self.port_emb(out_ports)
            parts = ([edge_attr] if edge_attr is not None else []) + [in_emb, out_emb]
            edge_attr = torch.cat(parts, dim=-1)
        elif edge_attr is None:
            edge_attr = x.new_zeros(src_edge_inc.size(1), 0)

        H = self.node_emb(x)
        Z = self.edge_emb(edge_attr)

        for layer in self.layers:
            H, Z = layer(H, Z, src_edge_inc, tgt_edge_inc)
            H = F.dropout(H, p=self.dropout, training=self.training)

        return self.classifier(H)

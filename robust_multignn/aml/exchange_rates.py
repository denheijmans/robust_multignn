"""Exchange rate lookup for AML datasets.

``format_dataset`` (see :mod:`robust_multignn.aml.utils`) encodes currency
names as integers via a per-dataset insertion-order dictionary, so the same
integer can mean different currencies across datasets. To apply a currency
exchange rate table to the encoded ``formatted_transactions.csv`` columns, we
need each dataset's own name -> integer mapping to translate the
dataset-independent rate table into dataset-specific integer pairs.

Kept decoupled from ``AMLDataLoader``'s ``data.pt`` cache on purpose: the
rate lookup is composed *after* loading cached GraphData, so regenerating the
rate table doesn't invalidate that cache.
"""
from __future__ import annotations

import os.path as osp
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from robust_multignn.aml.dataloader import _CSV_NAMES
from robust_multignn.data import GraphData

DEFAULT_RATE_TABLE_PATH = Path("analysis/tables/exchange_rates/exchange_rates_universal.csv")


def build_currency_name_mapping(raw_trans_path: str) -> Dict[str, int]:
    """Replicate ``format_dataset``'s currency name -> int mapping, read-only.

    Reproduces the same insertion order as the row-by-row loop in
    ``robust_multignn.aml.utils.format_dataset`` ("Receiving Currency" then
    "Payment Currency" per row), but only reads the raw CSV. Safe to call
    independently since the mapping is a deterministic function of the raw
    CSV's row order, which ``format_dataset`` never modifies.

    Interleaves the two columns and takes ``pd.unique`` (order-preserving)
    instead of ``iterrows()``, which is too slow on multi-million row CSVs.
    """
    raw = pd.read_csv(raw_trans_path, usecols=["Receiving Currency", "Payment Currency"])

    n = len(raw)
    interleaved = np.empty(2 * n, dtype=object)
    interleaved[0::2] = raw["Receiving Currency"].to_numpy()
    interleaved[1::2] = raw["Payment Currency"].to_numpy()

    first_seen_order = pd.unique(interleaved)
    return {name: i for i, name in enumerate(first_seen_order)}


def load_universal_rate_table(
    rate_table_path: str | Path = DEFAULT_RATE_TABLE_PATH,
) -> pd.DataFrame:
    """Load the universal exchange rate table, building it if missing.

    See ``analysis/scripts/build_aml_universal_exchange_rate_table.py``.
    """
    rate_table_path = Path(rate_table_path)
    if not rate_table_path.exists():
        from analysis.scripts.build_aml_universal_exchange_rate_table import (
            DEFAULT_DATA_DIR,
            compute_pooled_rates,
            load_cross_currency_rows,
        )

        rate_table_path.parent.mkdir(parents=True, exist_ok=True)
        cross = load_cross_currency_rows(DEFAULT_DATA_DIR)
        rates = compute_pooled_rates(cross)
        rates.to_csv(rate_table_path, index=False)

    return pd.read_csv(rate_table_path)


def build_dataset_rate_lookup(
    name: str,
    data_dir: str,
    rate_table_path: str | Path = DEFAULT_RATE_TABLE_PATH,
) -> Dict[Tuple[int, int], float]:
    """Build a ``(sent_currency_int, received_currency_int) -> rate`` dict for
    one AML dataset, e.g. ``"Small_HI"``.

    The rate converts an amount in ``sent_currency_int`` to
    ``received_currency_int`` (``amount_received = rate * amount_sent``),
    matching the universal rate table's ``median_rate`` convention
    (``Payment Currency -> Receiving Currency``).
    """
    root_dir = osp.join(data_dir, name)
    raw_trans_path = osp.join(root_dir, _CSV_NAMES[name])

    currency_to_int = build_currency_name_mapping(raw_trans_path)
    rates = load_universal_rate_table(rate_table_path)

    lookup: Dict[Tuple[int, int], float] = {}
    for _, row in rates.iterrows():
        sent_name = row["Payment Currency"]
        received_name = row["Receiving Currency"]
        if sent_name not in currency_to_int or received_name not in currency_to_int:
            continue  # not observed in this dataset
        key = (currency_to_int[sent_name], currency_to_int[received_name])
        lookup[key] = row["median_rate"]

    return lookup


def build_dataset_rate_matrix(
    name: str,
    data_dir: str,
    rate_table_path: str | Path = DEFAULT_RATE_TABLE_PATH,
) -> Tensor:
    """Dense ``[num_currencies, num_currencies]`` version of
    :func:`build_dataset_rate_lookup`, for vectorized per-edge lookups
    (``rate_matrix[sent_currency, received_currency]``).

    The diagonal is filled with ``1`` (same-currency conversion is a no-op),
    since the lookup itself only covers genuinely cross-currency pairs.
    Off-diagonal entries for pairs never observed in the raw data stay ``0``;
    callers should treat that as "no rate available", not a real rate of zero.
    """
    lookup = build_dataset_rate_lookup(name, data_dir, rate_table_path)
    num_currencies = max(idx for pair in lookup for idx in pair) + 1

    rate_matrix = torch.zeros((num_currencies, num_currencies), dtype=torch.float32)
    rate_matrix.fill_diagonal_(1.0)
    for (sent, received), rate in lookup.items():
        rate_matrix[sent, received] = rate

    return rate_matrix


# edge_attr column layout built in AMLDataLoader._process().
_EDGE_ATTR_COLUMNS = ["Timestamp", "Amount Received", "Received Currency", "Payment Format"]
_AMOUNT_COL = _EDGE_ATTR_COLUMNS.index("Amount Received")
_CURRENCY_COL = _EDGE_ATTR_COLUMNS.index("Received Currency")


def convert_rewired_edge_currencies(
    edge_attr: Tensor,
    currency: Tensor,
    pert_edge_idx: Tensor,
    new_targets: Tensor,
    node_currency_support: Tensor,
    rate_matrix: Tensor,
    edge_attr_mean: Tensor,
    edge_attr_std: Tensor,
    generator: torch.Generator | None = None,
) -> Tuple[Tensor, Tensor]:
    """Make rewired edges look plausible at their new target node.

    A rewired edge is a cheap detection signal if its target has never
    received in that currency (e.g. a Euro-only account suddenly receiving
    USD). To hide this:

    * If ``node_currency_support[new_target, received_currency]`` is set, the
      target already handles that currency, so the edge is left untouched.
    * Otherwise, a currency is drawn uniformly from the ones the target does
      handle, and ``Amount Received`` is converted into it.
    * A target with no incoming edges (empty support set) has nothing to be
      inconsistent with, so its edges are left unchanged.

    This applies to every rewired edge, not just ones that were originally
    cross-currency, since the detection signal only cares whether the target
    handles the currency.

    edge_attr has no ``Amount Sent`` column, so the new amount is derived
    from the implied sent amount (``amount_received / old_rate``) rather than
    a stored one: ``new_amount_received = amount_received * (new_rate /
    old_rate)``. This also handles originally same-currency edges correctly,
    since ``old_rate`` is then the diagonal (1.0). Currency pairs absent from
    ``rate_matrix`` (rate 0, never observed together) are left unconverted.

    Operates out-of-place, returning new ``(edge_attr, currency)`` tensors
    with only the qualifying ``pert_edge_idx`` rows changed.

    Parameters
    ----------
    edge_attr : Tensor
        ``[num_edges, 4]`` normalized edge features, column order
        ``[Timestamp, Amount Received, Received Currency, Payment Format]``.
    currency : Tensor
        ``[num_edges, 2]`` exact integer ``[Sent Currency, Received Currency]``.
    pert_edge_idx : Tensor
        Indices of the edges being rewired in this candidate sample.
    new_targets : Tensor
        New target node id for each entry in ``pert_edge_idx`` (same order).
    node_currency_support : Tensor
        ``[num_nodes, num_currencies]`` bool; ``True`` where that node has
        received in that currency.
    rate_matrix : Tensor
        ``[num_currencies, num_currencies]`` dense rate lookup, ``rate[a, b]``
        converts an amount in currency ``a`` to currency ``b`` (``0`` = unknown).
    edge_attr_mean, edge_attr_std : Tensor
        Per-split z-norm stats used to build ``edge_attr``, needed to
        de-normalize / re-normalize ``Amount Received``.
    generator : torch.Generator, optional
        Seeded RNG for the uniform draw over supported currencies, so an
        attack run stays reproducible.
    """
    edge_attr = edge_attr.clone()
    currency = currency.clone()

    sent_currency = currency[pert_edge_idx, 0]
    old_received_currency = currency[pert_edge_idx, 1]

    support = node_currency_support[new_targets]  # [n, num_currencies]
    already_supported = support.gather(1, old_received_currency.unsqueeze(1)).squeeze(1)
    has_any_support = support.any(dim=1)

    # edges whose currency the new target doesn't handle, but has history to contradict it with
    needs_change = (~already_supported) & has_any_support
    if not needs_change.any():
        return edge_attr, currency

    chosen_currency = old_received_currency.clone()
    chosen_currency[needs_change] = torch.multinomial(
        support[needs_change].to(edge_attr.dtype),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    old_rate = rate_matrix[sent_currency, old_received_currency]
    new_rate = rate_matrix[sent_currency, chosen_currency]
    has_both_rates = (old_rate != 0) & (new_rate != 0)

    convert_mask = needs_change & has_both_rates
    if not convert_mask.any():
        return edge_attr, currency

    rows = pert_edge_idx[convert_mask]
    rate_ratio = (new_rate[convert_mask] / old_rate[convert_mask]).to(edge_attr.dtype)

    mean_amount = edge_attr_mean[0, _AMOUNT_COL]
    std_amount = edge_attr_std[0, _AMOUNT_COL]
    mean_currency = edge_attr_mean[0, _CURRENCY_COL]
    std_currency = edge_attr_std[0, _CURRENCY_COL]

    raw_amount = edge_attr[rows, _AMOUNT_COL] * std_amount + mean_amount
    new_raw_amount = raw_amount * rate_ratio
    edge_attr[rows, _AMOUNT_COL] = (new_raw_amount - mean_amount) / std_amount

    new_currency_ids = chosen_currency[convert_mask].to(edge_attr.dtype)
    edge_attr[rows, _CURRENCY_COL] = (new_currency_ids - mean_currency) / std_currency

    currency[rows, 1] = chosen_currency[convert_mask]

    return edge_attr, currency

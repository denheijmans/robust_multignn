# Adapted from https://github.com/IBM/Multi-GNN (data_loading.py)
# (Apache License 2.0); modified for this project.
"""AML (Anti-Money-Laundering) dataset loading utilities.

Loads the IBM AML transaction datasets (Small/Medium/Large × LI/HI variants),
formats them into PyG-compatible GraphData objects with temporal cumulative
splits, and returns a GraphDataset alongside per-split evaluation index masks.
"""
from typing import Tuple

import itertools
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from robust_multignn.data import GraphData
from robust_multignn.aml.utils import index_to_mask, find_parallel_edges, z_norm, format_dataset

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

AML_DATASET_NAMES = [
    "Small_LI",
    "Small_HI",
    "Medium_LI",
    "Medium_HI",
    "Large_LI",
    "Large_HI",
]

_CSV_NAMES = {
    "Small_LI": "LI-Small_Trans.csv",
    "Small_HI": "HI-Small_Trans.csv",
    "Medium_LI": "LI-Medium_Trans.csv",
    "Medium_HI": "HI-Medium_Trans.csv",
    "Large_LI": "LI-Large_Trans.csv",
    "Large_HI": "HI-Large_Trans.csv",
}


def _currency_support_per_node(df_edges: pd.DataFrame, num_nodes: int) -> Tensor:
    """Which currencies each node has ever received in, as a boolean
    ``[num_nodes, num_currencies]`` matrix.

    A set rather than a single "dominant" currency because accounts really do
    handle more than one: in Small_HI, 8,195 accounts receive in 2+ currencies
    (up to 10). Collapsing those to one currency would make a rewired edge
    carrying a currency the account genuinely accepts look like an anomaly,
    and trigger a conversion that manufactures an inconsistency the data does
    not have.

    Only the received side is used: it is what edge_attr's "Received Currency"
    column has to be consistent with for a (possibly rewired) edge to look
    plausible at its target node.

    Nodes that never appear as a destination (e.g. source-only accounts) get
    an all-False row; callers must handle that case explicitly rather than
    sampling from an empty support set.
    """
    num_currencies = int(df_edges["Received Currency"].max()) + 1

    support = np.zeros((num_nodes, num_currencies), dtype=bool)
    support[
        df_edges["to_id"].to_numpy(),
        df_edges["Received Currency"].to_numpy(),
    ] = True
    return torch.from_numpy(support)


class AMLDataLoader:
    """Loads and preprocesses one of the IBM AML benchmark datasets.

    Parameters
    ----------
    name : str
        Dataset variant, one of :data:`AML_DATASET_NAMES` (e.g. ``"Small_HI"``).
    data_dir : str
        Root directory that contains a sub-folder named ``name`` with the
        raw CSV file.
    """

    def __init__(self, name: str, data_dir: str):
        if name not in AML_DATASET_NAMES:
            raise ValueError(
                f"Unknown AML dataset '{name}'. Choose from {AML_DATASET_NAMES}."
            )
        self.name = name
        self.root_dir = osp.join(data_dir, name)

    def get_data(
        self,
    ) -> Tuple[GraphData, GraphData, GraphData, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(tr_data, val_data, te_data, tr_inds, val_inds, te_inds)``.

        Loads from ``data.pt`` cache when available; otherwise processes the
        raw CSV and saves the cache.  The ``*_inds`` tensors are the original
        temporal edge indices and are valid as local indices within their
        respective cumulative data objects.
        """
        cache_path = osp.join(self.root_dir, "data.pt")
        if osp.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            return (
                cached["train"],
                cached["val"],
                cached["test"]
            )
        return self._process()
    

    def _process(self) -> Tuple[GraphData, GraphData, GraphData, Tensor, Tensor, Tensor]:

        transaction_file = osp.join(self.root_dir, "formatted_transactions.csv")
        
        # Check if the formatted dataset already exists to avoid reprocessing
        if not osp.exists(transaction_file):
            format_dataset(osp.join(self.root_dir, _CSV_NAMES.get(self.name)))
        
        df_edges = pd.read_csv(transaction_file)
        df_edges = df_edges.sort_values(by='Timestamp')

        print(f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
        y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

        print(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        print(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
        print(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        node_features = ['Feature']

        print(f'Edge features being used: {edge_features}')
        print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
        edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

        # Unnormalized currency, kept alongside the (z-scored) copy inside
        # edge_attr: column 0 is the sent currency, column 1 the received
        # currency, matching formatted_transactions.csv.
        currency = torch.LongTensor(
            df_edges.loc[:, ['Sent Currency', 'Received Currency']].to_numpy()
        )

        # Set of currencies each node has received in, used to check whether a
        # rewired edge's currency is plausible at its new target node.
        node_currency_support = _currency_support_per_node(df_edges, max_n_id)

        simp_edge_batch = find_parallel_edges(edge_index)

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        print(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

        #data splitting
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i,j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v/split_totals_sum for v in split_totals]
                split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
                score = max(split_error) #- (split_totals_sum/total) + 1
                split_scores[(i,j)] = score
            else:
                continue

        i,j = min(split_scores, key=split_scores.get)
        #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
        print(f'Calculate split: {split}')

        #Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately
                
        train_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        test_inds = torch.cat(split_inds[2])
        e_train = train_inds
        e_val = torch.cat([train_inds, val_inds])
        e_test = torch.cat([train_inds, val_inds, test_inds])

        self.simp_edge_batch_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr, edge_attr_mean, edge_attr_std = z_norm(edge_attr[e_mask])
            masked_y = y[e_mask]
            masked_timestamps = timestamps[e_mask]
            normed_x, x_mean, x_std = z_norm(x)

            data = GraphData(
                x=normed_x,
                num_nodes=int(x.shape[0]),
                edge_index=masked_edge_index,
                edge_attr=masked_edge_attr,
                edge_attr_mean=edge_attr_mean,
                edge_attr_std=edge_attr_std,
                y=masked_y,
                timestamps=masked_timestamps,
                split_mask=index_to_mask(inds, size=masked_edge_index.shape[1]),
                simp_edge_batch = simp_edge_batch[e_mask],
                currency=currency[e_mask],
                node_currency_support=node_currency_support,
            )

            self.data_dict[split] = data
        
        torch.save(self.data_dict, osp.join(self.root_dir, "data.pt"))

        return self.data_dict['train'], self.data_dict['val'], self.data_dict['test']



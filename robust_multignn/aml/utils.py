# Adapted from https://github.com/IBM/Multi-GNN (format_kaggle_files.py)
# (Apache License 2.0); modified for this project.
from typing import Optional

import os
from datetime import datetime

import pandas as pd
import torch
from torch import Tensor


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional): The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:
        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:
        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)


def criminal_accounts(edge_index: Tensor, y: Tensor) -> Tensor:
    """Accounts that sent or received at least one illicit transaction.

    Under the criminal threat model these are the accounts the launderers
    control. Taken over every edge passed in -- for a cumulative AML graph
    that is all periods, i.e. a white-box attacker that knows its own
    accounts, including ones first used in the test period.

    Returns the sorted unique node ids.
    """
    return torch.unique(edge_index[:, y.bool()])


def illicit_edges(edge_inds: Tensor, y: Tensor) -> Tensor:
    """The illicit edges among ``edge_inds`` (order preserved)."""
    return edge_inds[y[edge_inds].bool()]


def find_parallel_edges(edge_index):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0

    for edge in edge_index.T:
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
            
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)
    return simplified_edge_batch


def z_norm(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalise a 2-D tensor column-wise, skipping zero-std columns.

    Returns ``(normalized, mean, std)``. The mean/std are returned (not just
    the normalized tensor) so callers can persist them and later convert a
    raw value into this same normalized space -- e.g. an attack that
    recomputes ``Amount Received`` for a rewired edge needs to normalize it
    consistently with the rest of the (unperturbed) split.
    """
    mean = data.mean(0).unsqueeze(0)
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.ones(1, dtype=torch.float32), std)
    return (data - mean) / std, mean, std


def format_dataset(inPath: str) -> None:
    """Turn text-attributed AML dataset into a numerically encoded CSV.

    Matches the original processing logic row-by-row:
    - Encodes categorical columns (currencies, payment format, accounts) as
      integer IDs using insertion-order dictionaries.
    - Converts timestamps to seconds elapsed since the start of the first day.
    - Writes the formatted CSV, then re-reads and sorts by Timestamp in-place.
    """
    outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

    raw = pd.read_csv(inPath, dtype=str)

    currency = dict()
    paymentFormat = dict()
    bankAcc = dict()
    account = dict()

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = (
        "EdgeID,from_id,to_id,Timestamp,"
        "Amount Sent,Sent Currency,Amount Received,Received Currency,"
        "Payment Format,Is Laundering\n"
    )

    firstTs = -1

    with open(outPath, "w") as writer:
        writer.write(header)
        for i in range(len(raw)):
            row = raw.iloc[i]
            datetime_object = datetime.strptime(row["Timestamp"], "%Y/%m/%d %H:%M")
            ts = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year
            hour = datetime_object.hour
            minute = datetime_object.minute

            if firstTs == -1:
                startTime = datetime(year, month, day)
                firstTs = startTime.timestamp() - 10

            ts = ts - firstTs

            cur1 = get_dict_val(row["Receiving Currency"], currency)
            cur2 = get_dict_val(row["Payment Currency"], currency)

            fmt = get_dict_val(row["Payment Format"], paymentFormat)

            fromAccIdStr = row["From Bank"] + row.iloc[2]
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = row["To Bank"] + row.iloc[4]
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(row["Amount Received"])
            amountPaidOrig = float(row["Amount Paid"])

            isl = int(row["Is Laundering"])

            line = "%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n" % (
                i,
                fromId,
                toId,
                ts,
                amountPaidOrig,
                cur2,
                amountReceivedOrig,
                cur1,
                fmt,
                isl,
            )
            writer.write(line)

    # Re-read and sort by Timestamp (column index 3), then overwrite
    formatted = pd.read_csv(outPath)
    formatted = formatted.sort_values(by="Timestamp")
    formatted.to_csv(outPath, index=False)

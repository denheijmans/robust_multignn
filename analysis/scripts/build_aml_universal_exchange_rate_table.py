"""Build one universal AML exchange rate lookup table, pooled across datasets.

The per-dataset tables from build_aml_exchange_rate_tables.py agree across
small-li, small-hi, medium-li, medium-hi for essentially every currency pair
(see check_aml_exchange_rate_consistency.py) -- fiat-to-fiat rates match to
within ~1%, so there is no benefit to keeping four separate tables. This
script pools the cross-currency transactions from all four raw CSVs and
computes one median rate per (Payment Currency, Receiving Currency) pair.

Currency pairs are keyed by name (not the per-dataset integer codes in
formatted_transactions.csv), so pooling is safe: "US Dollar" means the same
thing in every dataset even though its encoded integer does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data")

RAW_TRANS_PATHS = {
    "small-li": "Small_LI/LI-Small_Trans.csv",
    "small-hi": "Small_HI/HI-Small_Trans.csv",
    "medium-li": "Medium_LI/LI-Medium_Trans.csv",
    "medium-hi": "Medium_HI/HI-Medium_Trans.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one universal exchange rate lookup table, pooled across all AML datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="root directory containing Small_LI/, Small_HI/, Medium_LI/, Medium_HI/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/tables/exchange_rates/exchange_rates_universal.csv"),
        help="destination CSV for the pooled lookup table",
    )
    return parser.parse_args()


def load_cross_currency_rows(data_dir: Path) -> pd.DataFrame:
    frames = []
    for name, rel_path in RAW_TRANS_PATHS.items():
        df = pd.read_csv(data_dir / rel_path)
        cross = df[df["Payment Currency"] != df["Receiving Currency"]].copy()
        cross["dataset"] = name
        frames.append(cross)
    return pd.concat(frames, ignore_index=True)


def compute_pooled_rates(cross: pd.DataFrame) -> pd.DataFrame:
    cross = cross.copy()
    cross["rate"] = cross["Amount Received"] / cross["Amount Paid"]
    rates = cross.groupby(["Payment Currency", "Receiving Currency"])["rate"].agg(
        median_rate="median", mean_rate="mean", std_rate="std", n="count"
    )
    rates = rates.reset_index()

    # Number of datasets each pair was observed in, so a "universal" rate
    # backed by a single dataset can be told apart from one confirmed by all four.
    n_datasets = cross.groupby(["Payment Currency", "Receiving Currency"])["dataset"].nunique()
    rates = rates.merge(
        n_datasets.rename("n_datasets"),
        on=["Payment Currency", "Receiving Currency"],
    )
    return rates


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cross = load_cross_currency_rows(args.data_dir)
    rates = compute_pooled_rates(cross)
    rates = rates.sort_values("n", ascending=False)

    rates.to_csv(args.output, index=False)

    n_bitcoin = ((rates["Payment Currency"] == "Bitcoin") | (rates["Receiving Currency"] == "Bitcoin")).sum()
    print(f"Universal table: {len(rates)} currency pairs ({n_bitcoin} involving Bitcoin) -> {args.output}")
    print(rates.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

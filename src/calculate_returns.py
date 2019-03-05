#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate returns and labels."""

import argparse

import pandas as pd

from utils import calculate_class, calculate_returns


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument(
        "--indir", help="Dataset directory.",
        default="../data/dowjones/all_stocks_2006-01-01_to_2018-01-01.csv")

    parser.add_argument('--outdir', help='Model directory.',
                        default="../model/dowjones/sample.csv")
    args = parser.parse_args()
    dataset = pd.read_csv(args.indir, index_col='Date',
                          parse_dates=['Date'])
    returns = calculate_returns(dataset)
    labels = calculate_class(returns)
    # returns = (returns - returns.mean()) / returns.std()
    print(f"Returns shape: {returns.shape}")
    print(f"Labels shape: {labels.shape}")
    returns.to_csv("../data/dowjones_calculated/returns.csv")
    labels.to_csv("../data/dowjones_calculated/labels.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

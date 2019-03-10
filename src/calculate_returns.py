#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate returns and labels."""

import argparse

import pandas as pd

from utils import (calculate_absolute_class, calculate_class,
                   calculate_log_returns, calculate_returns)


def main():
    """Run main program."""
    index = "dowjones"
    index = "frankfurt"
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument(
        "--indir", help="Dataset directory.",
        default="../data/dowjones/all_stocks_2006-01-01_to_2018-01-01.csv")

    parser.add_argument('--outdir', help='Model directory.',
                        default="../model/dowjones/sample.csv")
    # args = parser.parse_args()
    # dataset = pd.read_csv(args.indir,
    #                       index_col='Date',
    #                       parse_dates=['Date'])
    dataset = pd.read_csv(f"../data/frankfurt_calculated/stocks.csv",
                          index_col='Date',
                          parse_dates=['Date'])
    returns = calculate_returns(dataset)
    log_returns = calculate_log_returns(dataset)
    labels = calculate_class(returns)
    absolute_labels = calculate_absolute_class(returns)
    log_labels = calculate_class(log_returns)
    absolute_log_labels = calculate_absolute_class(log_returns)
    # returns = (returns - returns.mean()) / returns.std()
    print(f"Returns shape: {returns.shape}")
    print(f"Labels shape: {labels.shape}")
    returns.to_csv(f"../data/{index}_calculated/returns1.csv")
    labels.to_csv(f"../data/{index}_calculated/labels1.csv")
    absolute_labels.to_csv(f"../data/{index}_calculated/absolute_labels1.csv")
    log_returns.to_csv(f"../data/{index}_calculated/log_returns1.csv")
    log_labels.to_csv(f"../data/{index}_calculated/log_labels1.csv")
    absolute_log_labels.to_csv(
        f"../data/{index}_calculated/absolute_log_labels1.csv")

    print("Done.")
    return 0


if __name__ == "__main__":
    main()

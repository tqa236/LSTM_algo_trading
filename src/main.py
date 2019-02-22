#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a LSTM model."""

import os
import argparse
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calculate_class(returns):
    """Find the class for each LSTM sequence based on the median returns."""
    median_returns = returns.median(axis=1)
    labels = returns.iloc[:, :-
                          1].apply(lambda x: np.where(x >= median_returns, 1, 0),
                                   axis=0)
    return labels


def calculate_returns(stocks):
    """Calculate the returns of all indices."""
    stocks = stocks[["High", "Name"]]
    stocks = stocks.pivot_table(
        values='High', index=stocks.index, columns='Name', aggfunc='first')
    returns = (stocks - stocks.shift(1))/stocks.shift(1)
    returns = returns.dropna()
    returns = (returns - returns.mean())/returns.std()

    return returns


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--indir", help="Dataset directory.",
                        default="../data/dowjones/all_stocks_2006-01-01_to_2018-01-01.csv")

    parser.add_argument('--outdir', help='Model directory.',
                        default="../model/dowjones/sample.csv")
    args = parser.parse_args()
    dataset = pd.read_csv(args.indir, index_col='Date',
                          parse_dates=['Date'])
    returns = calculate_returns(dataset)
    returns.to_csv("../model/dowjones/returns.csv")
    labels = calculate_class(returns)
    labels.to_csv("../model/dowjones/labels.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

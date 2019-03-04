#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a random strategy to pick the stocks."""

import argparse

import numpy as np
import pandas as pd


def generate_random_strategy(returns):
    """Generate a random probability tha"t a stock will beat the market."""
    probabilities = returns
    probabilities = probabilities.apply(
        lambda x: [np.random.rand() for i in x],
        axis=1, result_type='broadcast')
    return probabilities


def long_short_postion(probabilities, k):
    """
    Decide the stock position based on the probability that it will beat
    the market.
    """
    positions = probabilities
    short = np.argpartition(positions, k)[:k]
    neutral = np.argpartition(positions, len(
        positions) - k)[:(len(positions) - k)]
    positions[:] = 1
    positions[neutral] = 0
    positions[short] = -1
    return positions


def calculate_class(returns):
    """Find the class for each LSTM sequence based on the median returns."""
    median_returns = returns.median(axis=1)
    labels = returns.iloc[:, :].apply(lambda x: np.where
                                      (x >= median_returns, 1, 0), axis=0)
    return labels


def calculate_returns(stocks):
    """Calculate the real returns of all indices without normalization."""
    stocks = stocks[["Close", "Name"]]
    stocks = stocks.pivot_table(
        values='Close', index=stocks.index, columns='Name', aggfunc='first')
    returns = (stocks - stocks.shift(1)) / stocks.shift(1)
    returns = returns.dropna()
    return returns


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
    probabilities = generate_random_strategy(returns)
    k = 10
    positions = probabilities
    positions.apply(lambda x: long_short_postion(
        x, k), axis=1, result_type='broadcast')
    random_returns = returns.mul(positions)
    random_returns = random_returns[750:3000]
    no_rebalance = (random_returns + 1).product().sum() / (2 * k)
    rebalance = (1 + random_returns.sum(axis=1) / (2 * k)).product()
    print(f"Return without rebalance: {no_rebalance}")
    print(f"Return with rebalance: {rebalance}")
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

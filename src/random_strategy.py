#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a random strategy to pick the stocks."""

import argparse

import numpy as np
import pandas as pd

from utils import calculate_returns, long_short_postion


def generate_random_strategy(returns):
    """Generate a random probability tha"t a stock will beat the market."""
    probabilities = returns
    probabilities = probabilities.apply(
        lambda x: [np.random.rand() for i in x],
        axis=1, result_type='broadcast')
    return probabilities


def calculate_random_returns(returns, k=10, start=750, end=3000):
    """Calculate the returns of a random trading strategy."""
    probabilities = generate_random_strategy(returns)
    positions = probabilities
    positions.apply(lambda x: long_short_postion(
        x, k), axis=1, result_type='broadcast')
    random_returns = returns.mul(positions)
    random_returns = random_returns[start:end]
    no_rebalance = (random_returns + 1).product().sum() / (2 * k)
    rebalance = (1 + random_returns.sum(axis=1) / (2 * k)).product()
    return [no_rebalance, rebalance]


def random_trading(returns, k=10, start=750, end=3000, times=1):
    """Make a list of random trading returns."""
    return [calculate_random_returns(
        returns, k, start, end) for i in range(times)]


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument(
        "--indir", help="Dataset directory.",
        default="../data/dowjones/all_stocks_2006-01-01_to_2018-01-01.csv")

    parser.add_argument('--outdir', help='Model directory.',
                        default="../data/dowjones_calculated/rebalance.csv")
    args = parser.parse_args()
    dataset = pd.read_csv(args.indir, index_col='Date',
                          parse_dates=['Date'])
    returns = calculate_returns(dataset)
    results = random_trading(returns, times=100000)
    pd.DataFrame(data=results).to_csv(
        "../data/dowjones_calculated/random_trading.csv",
        sep=',', index=False, header=["No Rebalance", "Rebalance"])
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

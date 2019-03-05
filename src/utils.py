#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a random strategy to pick the stocks."""

import numpy as np


def generate_random_strategy(returns):
    """Generate a random probability tha"t a stock will beat the market."""
    probabilities = returns
    probabilities = probabilities.apply(
        lambda x: [np.random.rand() for i in x],
        axis=1, result_type='broadcast')
    return probabilities


def long_short_postion(probabilities, k):
    """
    Make a simple long short strategy.

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


def calculate_returns(stocks, normalize=False):
    """Calculate the real returns of all indices without normalization."""
    stocks = stocks[["Close", "Name"]]
    stocks = stocks.pivot_table(
        values='Close', index=stocks.index, columns='Name', aggfunc='first')
    returns = (stocks - stocks.shift(1)) / stocks.shift(1)
    returns = returns.dropna()
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

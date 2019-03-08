#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Every function that need to be used more than one time."""

import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator


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


def calculate_absolute_class(returns):
    """Predict the stock will go up or down."""
    labels = returns.iloc[:, :].apply(lambda x: np.where
                                      (x >= 0, 1, 0), axis=0)
    return labels


def calculate_returns(stocks):
    """Calculate the real returns of all indices without normalization."""
    stocks = stocks[["Close", "Name"]]
    stocks = stocks.pivot_table(
        values='Close', index=stocks.index, columns='Name', aggfunc='first')
    returns = (stocks - stocks.shift(1)) / stocks.shift(1)
    returns = returns.dropna()
    return returns


def calculate_log_returns(stocks):
    """Calculate the log returns of all indices without normalization."""
    stocks = stocks[["Close", "Name"]]
    stocks = stocks.pivot_table(
        values='Close', index=stocks.index, columns='Name', aggfunc='first')
    returns = np.log(stocks) - np.log(stocks.shift(1))
    returns = returns.dropna()
    return returns


def normalize_data(df):
    """normalize a dataframe."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    df = df.sub(mean, axis=0)
    df = df.div(std, axis=0)
    df = df.values
    return df


def generate_time_series_sample(data, target, timestep):
    """Generate samples of a time series with a certain length."""
    generator = TimeseriesGenerator(data, target,
                                    length=timestep, sampling_rate=1,
                                    batch_size=(data.shape[0] - timestep))
    return generator[0][0], generator[0][1]

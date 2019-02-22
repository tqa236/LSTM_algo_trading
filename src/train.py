#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a LSTM model."""

import os
import argparse
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_model(X_train, y_train, epochs=5, batch_size=32):
    """Train an LSTM model."""
    # The LSTM architecture
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Second LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Third LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Fourth LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    # The output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    # Fitting to the training set
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return regressor


def train(returns, labels, ticker="ABBA", length=240):
    """Train an LSTM model."""
    series = returns[ticker]
    label = labels[ticker]
    X = [series[i:i+length] for i in range(len(series) - length)]
    y = label[ticker][length:]
    X, y = np.array(X), np.array(y)
    SPLIT = int(0.6 * len(X))

    X_train = X[:SPLIT]
    y_train = y[:SPLIT]
    X_test = X[SPLIT:]
    y_test = y[SPLIT:]
    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--returns", help="Dataset directory.",
                        default="../model/dowjones/returns.csv")
    parser.add_argument("--labels", help="Dataset directory.",
                        default="../model/dowjones/returns.csv")
    parser.add_argument('--outdir', help='Model directory.',
                        default="../model/dowjones/sample.csv")
    args = parser.parse_args()
    returns = pd.read_csv(args.returns, index_col='Date',
                          parse_dates=['Date'])
    labels = pd.read_csv(args.returns, index_col='Date',
                         parse_dates=['Date'])
    returns.to_csv("../model/dowjones/sample1.csv")
    labels.to_csv("../model/dowjones/sample2.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

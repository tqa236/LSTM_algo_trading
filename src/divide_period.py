#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Divide the data into period."""

import argparse
import pickle

import pandas as pd


def divide_period(returns, labels, train_length=750, test_length=250,
                  timesteps=240):
    """Divide the data into period."""
    num_period = int((len(labels) - train_length) / test_length)
    trains = [(returns[test_length * i: train_length + test_length * i],
               labels[test_length * i: train_length + test_length * i])
              for i in range(num_period)]
    tests = [(returns[train_length - timesteps + test_length * i:
                      train_length + test_length * (i + 1)],
              labels[train_length - timesteps + test_length * i:
                     train_length + test_length * (i + 1)])
             for i in range(num_period)]

    return (trains, tests)


def main():
    """Run main program."""
    train_length = 2500
    test_length = 250
    timesteps = 240
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--returns", help="Dataset directory.",
                        default="../data/dowjones_calculated/returns1.csv")
    parser.add_argument(
        "--labels", help="Dataset directory.",
        default="../data/dowjones_calculated/absolute_labels1.csv")
    parser.add_argument('--outdir', help='Model directory.',
                        default=f"../data/dowjones_calculated/absolute_periods"
                        f"{train_length}_{test_length}_{timesteps}.txt")

    args = parser.parse_args()
    returns = pd.read_csv(args.returns, index_col='Date',
                          parse_dates=['Date'])
    labels = pd.read_csv(args.labels, index_col='Date',
                         parse_dates=['Date'])

    periods = divide_period(
        returns, labels, train_length, test_length, timesteps)
    print("Training set")
    print(f"Returns shape for 1 period: {periods[0][0][0].shape}")
    print(f"Labels shape for 1 period: {periods[0][0][1].shape}")
    print("Test set")
    print(f"Returns shape for 1 period: {periods[1][0][0].shape}")
    print(f"Labels shape for 1 period: {periods[1][0][1].shape}")

    with open(args.outdir, "wb") as file:
        pickle.dump(periods, file)
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

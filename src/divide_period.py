#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Divide the data into period."""

import argparse
import pickle

import pandas as pd


def divide_period(returns, labels, train_length=750, test_length=250):
    """Divide the data into period."""
    num_period = int((len(labels) - train_length) / test_length)
    trains = [(returns[250 * i: 750 + 250 * i],
               labels[250 * i: 750 + 250 * i]) for i in range(num_period)]
    tests = [(returns[750 + 250 * i: 750 + 250 * (i + 1)],
              labels[750 + 250 * i: 750 + 250 * (i + 1)])
             for i in range(num_period)]

    return (trains, tests)


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--returns", help="Dataset directory.",
                        default="../data/dowjones_calculated/returns.csv")
    parser.add_argument("--labels", help="Dataset directory.",
                        default="../data/dowjones_calculated/labels.csv")
    parser.add_argument('--outdir', help='Model directory.',
                        default="../data/dowjones_calculated/periods.txt")

    args = parser.parse_args()
    returns = pd.read_csv(args.returns, index_col='Date',
                          parse_dates=['Date'])
    labels = pd.read_csv(args.labels, index_col='Date',
                         parse_dates=['Date'])
    periods = divide_period(returns, labels)
    print("Training set")
    print(f"Returns shape for 1 period: {periods[0][0][0].shape}")
    print(f"Labels shape for 1 period: {periods[0][0][1].shape}")
    print("Test set")
    print(f"Returns shape for 1 period: {periods[1][0][0].shape}")
    print(f"Labels shape for 1 period: {periods[1][0][1].shape}")

    with open(args.outdir, "wb") as file:
        pickle.dump(periods, file)
    # with open("test.txt", "rb") as fp:   # Unpickling
    #     b = pickle.load(fp)
    print("Done.")
    return 0


if __name__ == "__main__":
    main()

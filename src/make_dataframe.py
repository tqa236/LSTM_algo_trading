#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate returns and labels."""

import argparse

import pandas as pd


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument(
        "--indir", help="Dataset directory.",
        default="../data/frankfurt/FSE_metadata.csv")

    parser.add_argument('--outdir', help='Model directory.',
                        default="../data/frankfurt_calculated/stocks.csv")
    args = parser.parse_args()
    tickers = pd.read_csv(args.indir)
    choose_from = tickers["from_date"] < "2001-01-01"
    choose_to = tickers["to_date"] > "2018-01-01"
    tickers = tickers[choose_from & choose_to]
    stock = pd.read_csv('../data/frankfurt/stocks/AAD_X.csv',
                        index_col='Date', parse_dates=['Date'])
    stocks = pd.DataFrame(index=stock.index)
    stocks = stocks.loc['2001-01-01':'2018-01-01']
    for ticker in tickers.code:
        stock = pd.read_csv(f'../data/frankfurt/stocks/{ticker}.csv',
                            index_col='Date', parse_dates=['Date'])
        stocks[ticker] = stock["Close"].loc['2001-01-01':'2018-01-01']
    stocks = stocks.dropna(axis=1)
    print(f"Stocks shape: {stocks.shape}")
    stocks.to_csv(args.outdir)

    print("Done.")
    return 0


if __name__ == "__main__":
    main()

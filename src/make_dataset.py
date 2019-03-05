#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make dataset."""
import pandas as pd

import quandl


def download_data(metadata_df, api=None):
    """Download data from Quandl."""
    for ticker in metadata_df["code"]:
        print(ticker)
        # try:
        symbol = "FSE/" + ticker
        quandl.ApiConfig.api_key = api
        mydata = quandl.get(symbol)
        mydata.to_csv("../data/frankfurt/stocks_tmp/" + ticker + ".csv")
        # except:
        #     pass


def main():
    """Run main program."""
    metadata_df = pd.read_csv("../data/frankfurt/FSE_metadata.csv")
    api = pd.read_csv("../data/personal_data/quandl_API.txt",
                      header=None)[0][0]
    download_data(metadata_df, api)
    return 0


if __name__ == "__main__":
    main()

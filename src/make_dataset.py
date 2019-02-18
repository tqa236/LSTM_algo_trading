#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import quandl


def download_data(metadata_df, API=None):
    for ticker in metadata_df["code"]:
        print(ticker)
        try:
            symbol = "FSE/" + ticker
            quandl.ApiConfig.api_key = API
            mydata = quandl.get(symbol)
            mydata.to_csv("../data/frankfurt/stocks_tmp/" + ticker + ".csv")
        except:
            pass


def main():
    """ Main program """
    metadata_df = pd.read_csv("../data/frankfurt/FSE_metadata.csv")
    API = pd.read_csv("../data/personal_data/quandl_API.txt", header=None)[0][0]
    download_data(metadata_df, API)
    return 0


if __name__ == "__main__":
    main()

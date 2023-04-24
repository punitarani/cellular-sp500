"""
data.py
"""

import datetime as dt

import pandas as pd
import pandas_datareader.data as web


def get_sp500_tickers() -> list[str]:
    """Get a list of S&P 500 tickers"""

    # Fetching the S&P 500 components list from wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]

    # Sort the list of tickers alphabetically
    df.sort_values(by=['Symbol'], inplace=True)

    # Ensure there are 503 tickers
    assert len(df) == 503, "There should be 503 tickers in the S&P 500"

    return list(df.Symbol)


def get_stock_data(ticker) -> pd.DataFrame:
    """Get historical OHLC data for a given ticker from 1990 to 2022 using stooq"""
    start_date = dt.datetime(1990, 1, 1)
    end_date = dt.datetime(2022, 12, 31)

    # Fetching the stock data
    data = web.DataReader(ticker, 'stooq', start_date, end_date)

    # Sort the data by date in ascending order
    data.sort_index(inplace=True)

    return data


if __name__ == "__main__":
    # Get the list of S&P 500 tickers
    print("S&P500 Companies: ", get_sp500_tickers())

    # Get the historical stock data for Apple
    aapl = get_stock_data('AAPL')
    print(aapl.head())

    # Save the data to a parquet file
    aapl.to_parquet('data/aapl.parquet')

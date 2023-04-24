"""
data.py
"""

import datetime as dt
import time
from multiprocessing import Pool

import pandas as pd
import pandas_datareader as web
import pandas_datareader.data as web
from tqdm import tqdm


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

    retries = 3
    delay = 1
    for retry in range(retries):
        try:
            # Fetching the stock data
            data = web.DataReader(ticker, 'stooq', start_date, end_date)

            # Sort the data by date in ascending order
            data.sort_index(inplace=True)

            return data
        except Exception as error:
            if retry < retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise error


def save_stock_data_as_parquet(ticker: str) -> None:
    """Save historical stock data as parquet file for a given ticker"""
    data = get_stock_data(ticker)
    data.to_parquet(f"data/{ticker}.parquet")


def process_ticker(ticker) -> tuple:
    """Process a ticker and return the ticker and a boolean indicating if the ticker was processed successfully"""
    try:
        save_stock_data_as_parquet(ticker)
        return ticker, True
    except Exception as e:
        return ticker, False, str(e)


def get_and_save_sp500_stock_data() -> None:
    """Fetch historical stock data for S&P 500 companies and save as parquet files"""
    tickers = get_sp500_tickers()
    downloaded_tickers = []
    failed_tickers = []

    # Use multiprocessing to speed up the process
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap_unordered(process_ticker, tickers), total=len(tickers)))

    for result in results:
        if result[1]:
            downloaded_tickers.append(result[0])
        else:
            failed_tickers.append(result[0])

    print("\nDownloaded tickers:")
    print(downloaded_tickers)

    print("\nFailed tickers:")
    print(failed_tickers)


if __name__ == "__main__":
    # Get the list of S&P 500 tickers
    print("S&P500 Companies: ", get_sp500_tickers())

    # Download and save the stock data for S&P 500 companies
    get_and_save_sp500_stock_data()

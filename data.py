"""
data.py
"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Fetching the stock data
    data = web.DataReader(ticker, 'stooq', start_date, end_date)

    # Sort the data by date in ascending order
    data.sort_index(inplace=True)

    return data


def save_stock_data_as_parquet(ticker: str) -> None:
    """Save historical stock data as parquet file for a given ticker"""
    data = get_stock_data(ticker)
    data.to_parquet(f"data/{ticker}.parquet")


def get_and_save_sp500_stock_data() -> None:
    """Fetch historical stock data for S&P 500 companies and save as parquet files"""
    tickers = get_sp500_tickers()
    downloaded_tickers = []
    failed_tickers = []

    # Use multithreading to speed up the process
    with ThreadPoolExecutor() as executor:
        # Wrap the futures list with tqdm to track progress
        futures = {executor.submit(save_stock_data_as_parquet, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(futures)):
            ticker = futures[future]
            try:
                future.result()
                downloaded_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"Error downloading {ticker}: {e}")

    print("\nDownloaded tickers:")
    print(downloaded_tickers)

    print("\nFailed tickers:")
    print(failed_tickers)


if __name__ == "__main__":
    # Get the list of S&P 500 tickers
    print("S&P500 Companies: ", get_sp500_tickers())

    # Download and save the stock data for S&P 500 companies
    get_and_save_sp500_stock_data()

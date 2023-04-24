"""
data.py
"""

import datetime as dt
import os
import time
from multiprocessing import Pool

import pandas as pd
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

    tickers = list(df.Symbol)

    # Update tickers to replace `.` with `-`
    return [ticker.replace('.', '-') for ticker in tickers]


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


def get_and_save_sp500_stock_data() -> None:
    """Fetch historical stock data for S&P 500 companies and save as parquet files"""
    tickers = get_sp500_tickers()
    downloaded_tickers = []
    failed_tickers = []

    def process_ticker(ticker) -> tuple:
        """Process a ticker and return the ticker and a boolean indicating if the ticker was processed successfully"""
        try:
            data = get_stock_data(ticker)
            data.to_parquet(f"data/{ticker}.parquet")
            return ticker, True
        except Exception as e:
            return ticker, False, str(e)

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


def validate_data() -> list[str]:
    """Validate data in each parquet file and return a list of invalid files"""

    # Define the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    tickers = get_tickers()
    invalid_tickers = []

    for ticker in tqdm(tickers, desc="Validating .parquet files"):
        data = load_data(ticker)

        # Check if all required columns are present
        if not all(column in data.columns for column in required_columns):
            invalid_tickers.append(ticker)

    return invalid_tickers


def get_tickers() -> list[str]:
    """Get a list of tickers from the filenames in data/ excluding the .parquet extension"""
    tickers = []
    data_directory = "data/"

    for file in os.listdir(data_directory):
        if file.endswith(".parquet"):
            ticker = file[:-8]  # Remove the .parquet extension
            tickers.append(ticker)

    return tickers


def load_data(ticker: str) -> pd.DataFrame:
    """Load the parquet file into a DataFrame with necessary processing"""
    file_path = f"data/{ticker}.parquet"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for ticker {ticker}")

    data = pd.read_parquet(file_path)

    # Perform any necessary processing here (e.g., renaming columns, handling missing data, etc.)

    return data


def generate_daily_change_df() -> pd.DataFrame:
    """Generate a DataFrame with tickers as columns, dates as rows, and daily percentage change as values"""
    tickers = get_tickers()
    daily_change_dfs = []

    for ticker in tqdm(tickers, desc="Processing tickers"):
        data = load_data(ticker)
        col_name = ticker

        # Calculate the daily percentage change
        data[col_name] = data['Close'].pct_change() * 100

        # Keep only the daily change column and the date index
        daily_change_dfs.append(data[[col_name]])

    # Merge DataFrames on the date index
    daily_change_df = pd.concat(daily_change_dfs, axis=1, join='outer')

    # Handle missing data (e.g., fill with zeros or interpolate)
    daily_change_df.fillna(0, inplace=True)

    return daily_change_df


if __name__ == "__main__":
    # Get the list of S&P 500 tickers
    print("S&P500 Companies: ", get_sp500_tickers())

    # Download and save the stock data for S&P 500 companies
    get_and_save_sp500_stock_data()
    validate_data()

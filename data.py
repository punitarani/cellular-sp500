"""
data.py
"""

import datetime as dt
import json
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


def process_ticker_stock_data(ticker) -> tuple:
    """Process a ticker and return the ticker and a boolean indicating if the ticker was processed successfully"""
    try:
        data = get_stock_data(ticker)
        data.to_parquet(f"data/{ticker}.parquet")
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
        results = list(tqdm(pool.imap_unordered(process_ticker_stock_data, tickers), total=len(tickers)))

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


def get_market_cap(ticker) -> float:
    """Get the market cap for a given ticker"""
    data = web.get_quote_yahoo(ticker)
    market_cap = data['marketCap'][0]

    # Converting market cap to float
    market_cap = float(market_cap)

    return market_cap


def process_ticker_market_cap(ticker) -> tuple:
    """Process a ticker and return the ticker and its market cap"""

    retries = 3
    delay = 1
    for retry in range(retries):
        try:
            market_cap = get_market_cap(ticker)
            return ticker, market_cap
        except Exception:
            if retry < retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return ticker, None


def get_and_save_sp500_market_caps() -> None:
    """Get the market caps for S&P 500 companies and save as a CSV file"""
    tickers = get_sp500_tickers()
    saved_tickers = []
    failed_tickers = []

    # Store the market caps in a dictionary
    market_caps = {}

    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap_unordered(process_ticker_market_cap, tickers), total=len(tickers)))

    for result in results:
        ticker, market_cap = result
        market_caps[ticker] = market_cap

        if market_cap:
            saved_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)

    # Sort the market caps by ticker symbol
    market_caps = dict(sorted(market_caps.items()))

    # Save the market caps to a JSON file
    with open('market_caps.json', 'w') as f:
        json.dump(market_caps, f)
        print("Saved market caps to market_caps.json")


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


def load_daily_change_df() -> pd.DataFrame:
    """Load the daily change DataFrame from the daily_change.parquet file"""
    file_path = "sp500_daily_change.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError("No daily change file found")

    daily_change_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return daily_change_df


def find_first_nonzero_row(df, threshold):
    """
    Find the first row in a dataframe where at least `threshold` percent of the values are non-zero.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        threshold (float): The minimum percentage (0-1) of non-zero values required.

    Returns:
        int: The index of the first row where at least `threshold` percent of the values are non-zero.
             If no such row is found, returns -1.
    """
    # Count the number of non-zero values in each row
    counts = (df != 0).sum(axis=1)

    # Calculate the percentage of non-zero values in each row
    percentages = counts / len(df.columns)

    # Find the index of the first row where the percentage of non-zero values is greater than or equal to the threshold
    filtered_percentages = percentages[percentages >= threshold]
    if len(filtered_percentages) > 0:
        first_row = filtered_percentages.index[0]
        return first_row
    else:
        return -1


if __name__ == "__main__":
    # Get the list of S&P 500 tickers
    print("S&P500 Companies: ", get_sp500_tickers())

    # Download and save the stock data for S&P 500 companies
    get_and_save_sp500_stock_data()
    validate_data()

    # Get the market caps for S&P 500 companies
    get_and_save_sp500_market_caps()

"""
data.py
"""

import pandas as pd


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


if __name__ == "__main__":
    print("S&P500 Companies: ", get_sp500_tickers())

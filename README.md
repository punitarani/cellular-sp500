# S&P 500 Simulator with Cellular Automata

## Usage

### Requirements

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
poetry install
```

### Downloading Latest Data

Update the `start_date` and `end_date` in `data.py` to the desired date range.
The default is 1st Jan 1990 to 31st Dec 2022.
This is over 30 years of data adding up to close to 3 million data points.

Then run the following command:

```bash
python data.py
```

This will perform the following steps:

1. Get's the latest list of stocks in the S&P 500 from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
2. Downloads the historical data for each stock from [Stooq](https://stooq.com/db/h/) using `pandas-datareader`
   - Uses multiprocessing to speed up the download but can take ~2-5 minutes
   - Use any applicable data source for `pandas-datareader` by updating `data = web.DataReader(ticker, 'stooq', start_date, end_date)` in `get_stock_data()`
   - Saves the data to `data/<TICKER>.parquet`. Uses `parquet` format for faster read/write times.
3. Downloads the latest market capitalization data from [Yahoo Finance](https://finance.yahoo.com/) and saves it to `sp500_market_caps.json`

### Generating the grid

Run the `sp500_analysis.ipynb` notebook.

This notebook also explains the background, methodology and implementation of the grid placement algorithm. 

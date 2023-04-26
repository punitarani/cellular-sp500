# S&P 500 Simulator with Cellular Automata

> This won't make you the next [Jim Simons](https://en.wikipedia.org/wiki/Jim_Simons_(mathematician)) but it's a fun way to learn about the stock market and cellular automata.

## What's special?

**Cellular Automata meets Neural Networks**

- Grid placement is calculated analytically by using over 30 years of historical data of the S&P 500
- The grid weights are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) models trained on real historical data

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
The default is _1st Jan 1990_ to _31st Dec 2022._

**4 Million Data Points** over 33 years for 503 stocks in the S&P 500.

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

To find the optimal grid positions for each stock, run the following command:

```bash
python grid.py
```

This saves the grid positions to `sp500_grid.csv`.

### Training the Neural Network

To train the LSTM models for grid weights, run the following command:

```bash
python train.py
```

This will train the LSTM models for each neigboring stock pair and save the weights to `weights/<TICKER>.pth`, the model to `models/<TICKER>.pt` and scalers to `scalers/<TICKER>/.pkl`

> This is a very computationally expensive process. It can take 4-6 hours on an 8 core machine with 16 GB RAM and a CUDA enabled GPU.

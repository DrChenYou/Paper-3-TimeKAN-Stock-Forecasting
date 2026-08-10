# Dataset guide

## Study data

The paper evaluates daily histories for Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), and Apple (AAPL) from 10 January 2020 to 1 July 2025, approximately 1,381 records per asset. It lists the following Investing.com historical-data pages:

- Amazon: https://www.investing.com/equities/amazon-com-inc-historical-data
- NVIDIA: https://www.investing.com/equities/nvidia-corp-historical-data
- Tesla: https://www.investing.com/equities/tesla-motors-historical-data
- Apple: https://www.investing.com/equities/apple-computer-inc-historical-data

Obtain and use data in accordance with the provider's terms. This repository intentionally accepts local CSV exports and does not scrape those pages.

## Required schema

Each CSV needs a date column plus these numeric variables:

| Column | Meaning |
| --- | --- |
| `Open` | Opening price |
| `Close` | Closing price and default target |
| `High` | Daily high |
| `Low` | Daily low |
| `Volume` | Trading volume |
| `Change` | Daily change value or percentage |

Common commas, percent signs, and `K`, `M`, or `B` volume suffixes are parsed. Check the converted values before training, especially if an export uses a different decimal separator.

## Local layout

```text
data/
`-- raw/
    |-- AMZN.csv
    |-- NVDA.csv
    |-- TSLA.csv
    `-- AAPL.csv
```

The `data/` directory is ignored by Git. Preserve the download date, provider, adjustments, timezone, and any manual corrections in a separate experiment record.

## Split and leakage prevention

Rows are sorted chronologically and divided 70%/15%/15%. The feature scaler is fitted on the training rows only. No random shuffling is used to define the partitions. Window construction permits validation and test inputs to use immediately preceding history, but every forecast target is wholly contained in its assigned partition.

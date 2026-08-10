# Model card: TimeKAN

## Model summary

TimeKAN is a multivariate long-horizon time-series model. Cascaded adaptive moving-average filters create four detail components and one final trend. A shared M-KAN block combines a third-order Chebyshev KAN transform with a depthwise temporal convolution. Cross-frequency attention and learned band weights produce a 20-step forecast.

## Intended use

- Academic research on multiscale financial time series.
- Controlled comparisons with long-horizon forecasting baselines.
- Ablation studies of decomposition, KAN transforms, and frequency fusion.

## Out-of-scope use

The model is not a personalized investment recommendation, brokerage system, or guarantee of returns. Do not use a forecast as the sole basis for a financial decision.

## Limitations

- Daily price histories are non-stationary and may change after corporate actions or vendor corrections.
- Results depend on the exact date range, feature definitions, preprocessing, random seed, and forecast origin.
- Prediction error metrics do not capture liquidity, fees, slippage, drawdowns, or execution risk.
- The internal width is not reported in the paper; this repository declares its chosen width in configuration.
- Performance on four large US equities does not establish generalization to other assets or market regimes.

## Evaluation

Keep chronological partitions fixed. Report RMSE, MAE, MAPE, and R-squared in the original target scale. When reporting any return simulation, specify the signal rule, initial capital, position limits, transaction costs, slippage, and whether dividends and splits are adjusted.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit

# Sample data for revenue over time
# data = {'Date': pd.date_range(start='2022-01-01', periods=6, freq='M'),
#         'Revenue': [
#             # 100, 120, 130, 110, 150, 140, 160, 180, 200, 220, 240, 230,
#             #         250, 270, 290, 300, 310, 330, 340, 360, 380, 400, 420, 410,
#                     # 430, 450, 470, 490, 500, 520, 
#                     540, 550, 560, 580, 600, 620]}

# df = pd.DataFrame(data)
# df.set_index('Date', inplace=True)

# Number of folds for cross-validation
# n_splits = 5
# tscv = TimeSeriesSplit(n_splits=n_splits)

# Initialize the Exponential Smoothing model
# data = pd.read_excel('Revenuetest.xlsx')

# data = pd.read_excel('Revenuetest.xlsx')
# data = pd.read_excel('decreaseRevenue.xlsx')
# data = pd.read_excel('increaseRevenue.xlsx')
data = pd.read_excel('SeasonalRevenuetry.xlsx')
data.set_index('Month', inplace=True)


model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=3)

# Perform k-fold cross-validation
# for train_index, test_index in tscv.split(df):
#     train, test = df.iloc[train_index], df.iloc[test_index]
fit_model = model.fit()

# Forecast future values
# forecast_period = len(test)

forecast_period = 12

forecast = fit_model.forecast(steps=forecast_period)
print(forecast)
    # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(train.index, train['Revenue'], label='Train')
    # plt.plot(test.index, test['Revenue'], label='Test', linestyle='--')
    # plt.plot(test.index, forecast, label='Forecast', color='red')
    # plt.title('Revenue Forecast using Exponential Smoothing (Cross-Validation)')
    # plt.xlabel('Date')
    # plt.ylabel('Revenue')
    # plt.legend()
    # plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('EV_sales.csv')

data = df.loc[0:39, ['Date', 'Number_EV']]

data['ds'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Rename columns to 'ds' and 'y' for Prophet
data.rename(columns={'Number_EV': 'y'}, inplace=True)
df= data[['ds', 'y']]  
df.set_index('ds', inplace=True)

print(df)
# Number of folds for cross-validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Initialize the Exponential Smoothing model
model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=6) ### there is a difference when Seasoanl period is increase or decrease e.g seasonal_periods=6

# Perform k-fold cross-validation
for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]
    fit_model = model.fit()

    # Forecast future values
    # forecast_period = len(test)
    forecast_period = 20
    forecast = fit_model.forecast(steps=forecast_period)
    print("Forecast EV_sales for next 20 periods:")
    print(forecast)
    # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(train.index, train['y'], label='Train')
    # plt.plot(test.index, test['y'], label='Test', linestyle='--')
    # ############### to see graph of forecast by usisng Exponantial Smoothing test and forecast values must be equal
    # plt.plot(test.index, forecast, label='Forecast', color='red')      
    # plt.title('EV_sales Forecast using Exponential Smoothing (Cross-Validation)')
    # plt.xlabel('Date')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_excel('SeasonalRevenuetry.xlsx')

df.set_index('Month', inplace=True)

# Number of folds for cross-validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Initialize the Exponential Smoothing model
model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=3) ### there is a difference when Seasoanl period is increase or decrease e.g seasonal_periods=6

# Perform k-fold cross-validation
for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]
    fit_model = model.fit()

    # Forecast future values
    forecast_period = len(test)
    # forecast_period = 12
    forecast = fit_model.forecast(steps=forecast_period)
    print(forecast)
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Revenue'], label='Train')
    plt.plot(test.index, test['Revenue'], label='Test', linestyle='--')
    ############### to see graph of forecast by usisng Exponantial Smoothing test and forecast values must be equal
    plt.plot(test.index, forecast, label='Forecast', color='red')      
    plt.title('Revenue Forecast using Exponential Smoothing (Cross-Validation)')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.show()



########################################################################################################################
########################################################################################################################
########                                                                                                        ########
########     Version 2   API Expose                                                                             ########
########                                                                                                        ########
########################################################################################################################
########################################################################################################################

# from flask import Flask, request, jsonify
# import pandas as pd
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.model_selection import TimeSeriesSplit

# app = Flask(__name__)

# def perform_exponential_smoothing(data, seasonal_periods):
#     # Number of folds for cross-validation
#     n_splits = 5
#     tscv = TimeSeriesSplit(n_splits=n_splits)

#     # Initialize the Exponential Smoothing model
#     model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)

#     # Perform k-fold cross-validation
#     for train_index, test_index in tscv.split(data):
#         train, test = data.iloc[train_index], data.iloc[test_index]
#         fit_model = model.fit()

#         # Forecast future values
#         forecast_period = len(test)
#         forecast = fit_model.forecast(steps=forecast_period)
#         print(forecast)

#     return forecast

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json['data']  # Assuming 'data' is the key for the input data in JSON
#         seasonal_periods = request.json['seasonal_periods']  # Assuming 'seasonal_periods' is the key for the seasonal periods in JSON

#         # Assuming 'Month' is a column in your input data
#         df = pd.DataFrame(data)
#         df.set_index('Month', inplace=True)

#         forecast_result = perform_exponential_smoothing(df, seasonal_periods)

#         return jsonify({'forecast': forecast_result.tolist()})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

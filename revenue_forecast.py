import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima


# Load your data into a pandas DataFrame
data = pd.read_excel('Revenuetest.xlsx')

# Replace 'Date' and 'Revenue' with your actual column names
# For example, if your time series column is 'Time' and revenue column is 'Sales', use df['Time'] and df['Sales']
time_series_column = data['Month']
revenue_column = data['Revenue']

# Use auto_arima to find the optimal parameters
arima_model = auto_arima(revenue_column, seasonal=True, suppress_warnings=True)
# Display the optimal parameters
print(" The given Arima model:", arima_model)
# print("Optimal ARIMA Parameters (p, d, q):", arima_model.order)

# Fit ARIMA model
model = ARIMA(data['Revenue'], order=arima_model.order)
result = model.fit()
# Make predictions
forecast_steps = 20 # Adjust based on your forecast horizon
forecast = result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
# Print or save the forecasted values
print(forecast_values)

plt.figure(figsize=(24, 16))
plt.plot(time_series_column, revenue_column, label='Revenue')
plt.title('Historical Revenue Time Series')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()
plt.show()


########################################################################################################################
########################################################################################################################
########                                                                                                        ########
########     Version 2                                                                                          ########
########                                                                                                        ########
########################################################################################################################
########################################################################################################################

# # app.py

# from flask import Flask, request, jsonify
# import pandas as pd
# from pmdarima import auto_arima

# app = Flask(__name__)


# # Route to handle the ARIMA forecast
# @app.route('/forecast', methods=['POST'])
# def forecast():
#     # Get JSON data from the request
#     request_data = request.get_json()

#     # Extract values from the request
#     input_data = request_data.get('data', [])
#     forecast_steps = request_data.get('forecast_steps')

#     # Merge new data with the existing dataset
#     new_data = pd.DataFrame(input_data)
#     new_data['Date'] = pd.to_datetime(new_data['Date'])
#     # combined_data = pd.concat([df, new_data], ignore_index=True)

#     # Perform ARIMA forecasting
#     arima_model = auto_arima(new_data['Revenue'], seasonal=True, suppress_warnings=True)
#     forecast_values = arima_model.predict(n_periods=forecast_steps)

#     # Prepare response
#     response = {
#         'original_data': new_data.to_dict(orient='records'),
#         'forecast': forecast_values.tolist()
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)

####################################################################################################################
####################################################################################################################
########################                                                                    ########################
########################              Exponential Smoooting                                 ########################
########################                                                                    ########################
####################################################################################################################
####################################################################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.model_selection import TimeSeriesSplit

# # Sample data for revenue over time
# data = {'Date': pd.date_range(start='2020-01-01', periods=36, freq='M'),
#         'Revenue': [100, 120, 130, 110, 150, 140, 160, 180, 200, 220, 240, 230,
#                     250, 270, 290, 300, 310, 330, 340, 360, 380, 400, 420, 410,
#                     430, 450, 470, 490, 500, 520, 540, 550, 560, 580, 600, 620]}

# df = pd.DataFrame(data)
# df.set_index('Date', inplace=True)

# # Number of folds for cross-validation
# n_splits = 5
# tscv = TimeSeriesSplit(n_splits=n_splits)

# # Initialize the Exponential Smoothing model
# model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12)

# # Perform k-fold cross-validation
# for train_index, test_index in tscv.split(df):
#     train, test = df.iloc[train_index], df.iloc[test_index]
#     fit_model = model.fit()

#     # Forecast future values
#     forecast_period = len(test)
#     forecast = fit_model.forecast(steps=forecast_period)
#     print(forecast)
#     # Plot the results
#     plt.figure(figsize=(12, 6))
#     plt.plot(train.index, train['Revenue'], label='Train')
#     plt.plot(test.index, test['Revenue'], label='Test', linestyle='--')
#     plt.plot(test.index, forecast, label='Forecast', color='red')
#     plt.title('Revenue Forecast using Exponential Smoothing (Cross-Validation)')
#     plt.xlabel('Date')
#     plt.ylabel('Revenue')
#     plt.legend()
#     plt.show()



###########################   FaceBook Prophet
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# # Load your data into a pandas DataFrame
# data = pd.read_excel('Revenuetest.xlsx')

# # Rename columns to 'ds' and 'y' for Prophet
# data.rename(columns={'Month': 'ds', 'Revenue': 'y'}, inplace=True)

# # Fit Prophet model with PyStan backend
# prophet_model = Prophet()
# prophet_model.fit(data, algorithm='LBFGS')

# # Create a DataFrame with future dates for forecasting
# future = prophet_model.make_future_dataframe(periods=20, freq='M')  # Adjust based on your forecast horizon

# # Generate forecast
# forecast = prophet_model.predict(future)

# # Print or plot the forecasted values
# forecast_values = forecast[['ds', 'yhat']].tail(20)  # Adjust based on your forecast horizon
# print(forecast_values)

# # Plot the forecast
# prophet_model.plot(forecast)
# plt.title('Prophet Forecast')
# plt.xlabel('Month')
# plt.ylabel('Revenue')
# plt.show()


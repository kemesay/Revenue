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

# plt.figure(figsize=(16, 16))
# plt.plot(time_series_column, revenue_column, label='Revenue')
# plt.title('Historical Revenue Time Series')
# plt.xlabel('Month')
# plt.ylabel('Revenue')
# plt.legend()
# plt.show()


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


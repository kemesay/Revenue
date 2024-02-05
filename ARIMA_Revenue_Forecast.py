import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima


# data = pd.read_excel('Revenuetest.xlsx')
# data = pd.read_excel('decreaseRevenue.xlsx')
# data = pd.read_excel('increaseRevenue.xlsx')
data = pd.read_excel('SeasonalRevenuetry.xlsx')
time_series_column = data['Month']
revenue_column = data['Revenue']

# The use  of auto_arima to find the optimal parameters
arima_model = auto_arima(revenue_column, seasonal=True, suppress_warnings=True)
# print("Optimal ARIMA Parameters (p, d, q):", arima_model.order)
print(" The given Arima model:", arima_model)

# Fit ARIMA model
model = ARIMA(data['Revenue'], order=arima_model.order)
result = model.fit()
# Make predictions
forecast_steps = 20 
forecast = result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
print(forecast_values)

plt.figure(figsize=(20, 10))
plt.plot(time_series_column, revenue_column, label='Revenue')
plt.title('Historical Revenue Time Series')
plt.xlabel('Month')
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
# from pmdarima import auto_arima

# app = Flask(__name__)

# @app.route('/forecast', methods=['POST'])
# def forecast():
#     request_data = request.get_json()

#     input_data = request_data.get('data', [])
#     forecast_steps = request_data.get('forecast_steps')

#     new_data = pd.DataFrame(input_data)
#     new_data['Date'] = pd.to_datetime(new_data['Date'])

#     arima_model = auto_arima(new_data['Revenue'], seasonal=True, suppress_warnings=True)
#     forecast_values = arima_model.predict(n_periods=forecast_steps)

#     response = {
#         'original_data': new_data.to_dict(orient='records'),
#         'forecast': forecast_values.tolist()
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
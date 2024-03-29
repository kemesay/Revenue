##########################   FaceBook Prophet
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# # data = pd.read_excel('Revenuetest.xlsx')
# # data = pd.read_excel('decreaseRevenue.xlsx')
# # data = pd.read_excel('increaseRevenue.xlsx')
# data = pd.read_excel('SeasonalRevenuetry.xlsx')



# data.rename(columns={'Month': 'ds', 'Revenue': 'y'}, inplace=True)

# prophet_model = Prophet()
# prophet_model.fit(data)

# future = prophet_model.make_future_dataframe(periods=20, freq='M')  

# # Generate forecast
# forecast = prophet_model.predict(future)

# # Print or plot the forecasted values
# forecast_values = forecast[['ds', 'yhat']].tail(20) 
# print(forecast_values)

# # Plot the forecast
# prophet_model.plot(forecast)
# plt.title('Prophet Forecast')
# plt.xlabel('Month')
# plt.ylabel('Revenue')
# plt.show()


########################################################################################################################
########################################################################################################################
########                                                                                                        ########
########     Version 2    API Expose                                                                            ########
########                                                                                                        ########
########################################################################################################################
########################################################################################################################
from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        data = pd.DataFrame(request_data['data'])

        # Sort the DataFrame by the 'Date' column
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date')

        periods = request_data['forecast_steps']
        data.rename(columns={'Date': 'ds', 'Revenue': 'y'}, inplace=True)
        # prophet_model = Prophet(
        #     yearly_seasonality=True,
        #     weekly_seasonality=True,
        #     seasonality_mode='multiplicative',
        #     changepoint_prior_scale=0.05
        # )
        prophet_model = Prophet()
        prophet_model.fit(data)

        future = prophet_model.make_future_dataframe(periods=periods, freq='M')

        forecast = prophet_model.predict(future)

        forecast_values = forecast[['ds', 'yhat']].tail(periods)
        result = {'forecast_values': forecast_values.to_dict(orient='records')}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == "__main__":
    # Use Gunicorn to run the app
    import os
    host = '10.1.130.15'
    port = 9011
    os.system(f'gunicorn -w 4 -b {host}:{port} PROPHET_Revenue_Forecast:app')

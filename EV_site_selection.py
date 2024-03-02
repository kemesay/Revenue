import pandas as pd
import numpy as np
data = pd.read_excel('Data_AA_Points_Charging_st.xlsx')

data = data[[ 'type', 'name', 'latitude', 'longitude', 'altitude(ft)', 'utm_easting', 'utm_northing', 'malls_suppermarkets', 'hotels_restaurants' , 'hourly_counted', 'traffi_volume']]
# data1.rename(columns={'traffi_volume()': 'traffi_volume'}, inplace=True)
data['infra_score'] = data['malls_suppermarkets'] + data['hotels_restaurants']
# Step 2: Rename 'type' to 'Id', 'hourly_counted' to 'demographic_score', and 'traffi_volume' to 'traffic_volume'
data.rename(columns={'type': 'Id', 'hourly_counted': 'demographic_score', 'traffi_volume': 'traffic_volume'}, inplace=True)
data.drop(['hotels_restaurants', 'malls_suppermarkets'], axis=1, inplace=True)
data.to_csv('finalData.csv', index=False)

# data = data[data1]

print(data)
import joblib
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
import sklearn.compose._column_transformer
#loading the model

#converting month to numbers
month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,"July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

def get_features_from_db(state1):
    conn = sqlite3.connect(r'C:\Users\muska\OneDrive\Desktop\minor\dengue_pred.db')  # use full path if needed
    cursor = conn.cursor()
    query = """
        SELECT
        WaterLogging_substantial_pc,
        hygiene_total,
        hygiene_ratio,
        season_Post_Monsoon,
        season_Summer,
        season_Winter
        FROM dengue_data4
        WHERE state = ?;
    """
    df = pd.read_sql_query(query, conn, params=(state1,))
    conn.close()
    return df

#getting coordinates
def get_coordinates(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': place_name + ", India",
        'format': 'json'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0'  # Nominatim requires a user-agent
    }  
    time.sleep(1)
    response = requests.get(url, params=params,headers=headers)
    try:
        data = response.json()
    except ValueError:
        print("❌ Not a valid JSON response!")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text[:200])  # Print first 200 chars
        return None
    if data:
        lat = data[0]['lat']
        lon = data[0]['lon']
        return float(lat), float(lon)
    else:
        return None


#getting weekly weather data
def get_weatherData(s,date):
    coords = get_coordinates(s)
    if coords is None:
        print("⚠️ Coordinates not found for:", s)
        return pd.DataFrame()  # or raise custom exception
    lat, lon = coords
    start_date=date
    end_date=start_date + timedelta(days=6)
    end_date_str=end_date.strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date_str}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
        f"&timezone=Asia%2FKolkata"
    )
    response = requests.get(url)
    data = response.json()
    if "daily" not in data:
        raise Exception("Failed to fetch weather data.")
    df = pd.DataFrame(data["daily"])
    df['date'] = pd.to_datetime(df['time'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year

    df['temperature'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
    df.drop(['temperature_2m_max', 'temperature_2m_min'], axis=1, inplace=True)
    df['rainfall_lag1'] = df['precipitation_sum'].shift(1)
    df['rainfall_lag1'].fillna(df['precipitation_sum'], inplace=True)
    weekly_data = df.groupby(['year', 'week']).agg({
        'temperature': 'mean',
        'precipitation_sum': 'sum',
        'relative_humidity_2m_mean': 'mean',
        'rainfall_lag1':"mean"
    }).reset_index()
    weekly_data['rainfall_x_humidity']=weekly_data['precipitation_sum']*weekly_data['relative_humidity_2m_mean']
    weekly_data.rename(columns={'precipitation_sum':'rainfall','relative_humidity_2m_mean':'humidity'},inplace=True)
    final_df= weekly_data.drop(columns=['week','year'])
    return final_df
   

#creating function to retrieve the values and give the predicted answer
def predict_value(state,date):
    date_obj = date
    year = date_obj.year
    month = date_obj.month
    week_num = date_obj.isocalendar().week

    
    print(f"Received input: {state}, {month}, {year},{date_obj}")  # for debugging
    input_df = get_features_from_db(state)

    if(date>datetime.today().date()):  #agar date se bdi hai to future weather data chahiye
        date1=date-timedelta(days=365) #year-1
        date2=date1-timedelta(days=365) #year-2
        date3=date2-timedelta(days=365) #year-3
        d1=get_weatherData(state,date1)
        d2=get_weatherData(state,date2)
        d3=get_weatherData(state,date3)
        other_inputs=pd.concat([d1,d2,d3],axis=0,ignore_index=True) # we get 3 rows of data of 3 years

        other_inputs = other_inputs.mean(axis=0).to_frame().T # getting mean row wise
    else:
        # calling get_weatherDATA
        other_inputs=get_weatherData(state,date)
    other_inputs['week']=week_num
    other_inputs['year']=year
    other_inputs['state']=state
    input_df = input_df.reset_index(drop=True)
    other_inputs = other_inputs.reset_index(drop=True)
    final_input_df= pd.concat([input_df,other_inputs], axis=1)
    #merging the dataframes 
    if final_input_df.empty:
        print("❌ No data found for this date/state/week")
        return "⚠️ No data available for the selected input"
    else:
        class _RemainderColsList:
            def __init__(self, cols):
                self.cols = cols
        final_input_df.rename(columns={"season_Post_Monsoon":'season_Post-Monsoon','rainfall':'rainfall','humidity':'humidity'},inplace=True)
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
        pipeline = joblib.load(r"C:\Users\muska\OneDrive\Desktop\minor\Streamlit_Multipage_App\dengue_model_pipeline.pkl")
               
        y_pred = pipeline.predict(final_input_df)
        ans=int(y_pred[0])
        if ans==1:
            return 'low'
        elif ans==0:
            return 'high'
        else:
            return 'medium'



from flask import Flask
import itertools
import sys
import logging

import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

app = Flask(__name__)


@app.route('/')

def predict(starting_time = None):

    init_data = pd.read_csv('data/device_activations.csv')
    device_activations = pd.read_csv('data/device_activations.csv')
    init_data.time = pd.to_datetime(init_data.time)
    starting_time = min(init_data.time).date()

    earliest_date = starting_time if starting_time is not None else min(init_data.time).date()

    init_data['date'] = init_data['time'].dt.date
    init_data['hour_of_day'] = init_data['time'].dt.hour
    init_data['hour_of_week'] = init_data['time'].dt.dayofweek * 24 + init_data['hour_of_day']
    init_data['dummy']= (init_data['date']-earliest_date)
    init_data['duration_timedelta'] = init_data['dummy'].apply( lambda x:x.days )
    init_data['hour_slot'] =  init_data['duration_timedelta'] * 24 + init_data['hour_of_day']

    all_devices = list(init_data.device.unique())
    n_hours = ((max(init_data.time) - min(init_data.time)).days + 1) * 24
    hour_slot = list(range(n_hours))
    earliest_hour = min(init_data.hour_slot)
    hour_slot = [x+earliest_hour for x in hour_slot]

    blank_df = pd.DataFrame(list(product(all_devices, hour_slot)), columns=['device', 'hour_slot'])
    blank_df['hour_of_day'] = blank_df.hour_slot % 24
    blank_df['hour_of_week'] = blank_df.hour_slot % (7 * 24)
    blank_df['day_num'] = np.floor(blank_df.hour_slot / 24).astype(int)
    blank_df['week_num'] = np.floor(blank_df.hour_slot / (24 * 7)).astype(int)
    blank_df['day_of_week'] = np.floor(blank_df.day_num % 7).astype(int)

    grouped_df_1 = init_data.groupby(['device', 'hour_slot']).agg({'time':'min'}).reset_index()
    grouped_df_1.columns = ['device','hour_slot','earliest_activation']

    grouped_df_2 = init_data.groupby(['device', 'hour_slot']).agg({'time':'max'}).reset_index()
    grouped_df_2.columns = ['device','hour_slot','latest_activation']

    grouped_df_3 = init_data.groupby(['device', 'hour_slot']).agg({'device_activated':'sum'}).reset_index()
    grouped_df_3.columns = ['device','hour_slot','total_activations']

    train_data = pd.merge(blank_df, grouped_df_1,how='left').merge(grouped_df_2,  how='left').merge(grouped_df_3, how='left')
    train_data['is_active'] = (train_data.total_activations.isna() == False).astype(int)
    train_data['type'] = 'train'

    device_activations = pd.read_csv('./data/device_activations.csv')
    devices = list(device_activations['device'].unique())
    pred_time_df = pd.read_csv('data/input.txt', sep='\t')
    next_24_hours = pd.date_range(pred_time_df['time_for_prediction'][0], periods=24, freq='h').ceil('h')
    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, devices))
    df = pd.DataFrame(xproduct, columns=['time', 'device'])
    df['device_activated'] = -1
    df.columns = ['time', 'device', 'device_activated']
    df.time = pd.to_datetime(df.time)

    #Extract the required variables from the date
    earliest_date = starting_time if starting_time is not None else min(df.time).date()

    df['date'] = df['time'].dt.date
    df['hour_of_day'] = df['time'].dt.hour
    df['hour_of_week'] = df['time'].dt.dayofweek * 24 + df[
        'hour_of_day']
    df['dummy']= (df['date']-earliest_date)
    df['duration_timedelta'] = df['dummy'].apply( lambda x:x.days )
    df['hour_slot'] =  df['duration_timedelta'] * 24 + df['hour_of_day']
    # Create a blank dataset for every hour and device
    all_devices = list(df.device.unique())
    n_hours = ((max(df.time) - min(df.time)).days + 1) * 24
    hour_slot = list(range(n_hours))
    earliest_hour = min(df.hour_slot)
    hour_slot = [x+earliest_hour for x in hour_slot]

    blank_df_t = pd.DataFrame(list(product(all_devices, hour_slot)), columns=['device', 'hour_slot'])
    blank_df_t['hour_of_day'] = blank_df_t.hour_slot % 24
    blank_df_t['hour_of_week'] = blank_df_t.hour_slot % (7 * 24)
    blank_df_t['day_num'] = np.floor(blank_df_t.hour_slot / 24).astype(int)
    blank_df_t['week_num'] = np.floor(blank_df_t.hour_slot / (24 * 7)).astype(int)
    blank_df_t['day_of_week'] = np.floor(blank_df_t.day_num % 7).astype(int)

    grouped_df_1_t = df.groupby(['device', 'hour_slot']).agg({'time':'min'}).reset_index()
    grouped_df_1_t.columns = ['device','hour_slot','earliest_activation']

    grouped_df_2_t = df.groupby(['device', 'hour_slot']).agg({'time':'max'}).reset_index()
    grouped_df_2_t.columns = ['device','hour_slot','latest_activation']

    grouped_df_3_t = df.groupby(['device', 'hour_slot']).agg({'device_activated':'sum'}).reset_index()
    grouped_df_3_t.columns = ['device','hour_slot','total_activations']

    test_data = pd.merge(blank_df_t, grouped_df_1_t,how='left').merge(grouped_df_2_t,  how='left').merge(grouped_df_3_t, how='left')
    test_data['is_active'] = (test_data.total_activations.isna() == False).astype(int)
    test_data['type'] = 'predict'
    all_data = pd.concat([train_data, test_data], axis=0)
    
    # Select only the required columns
    active_last_week_df = all_data[['device', 'hour_slot', 'hour_of_week', 'week_num']]
    active_last_week_df.loc[:,'lag_week_num'] = active_last_week_df['week_num'] - 1

    active_last_week_df = pd.merge(
        active_last_week_df,
        all_data[['device', 'hour_of_week', 'week_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_week', 'lag_week_num'],
        right_on=['device', 'hour_of_week', 'week_num']
    )
    # Clean up
    active_last_week_df = active_last_week_df[['device', 'hour_slot', 'is_active']]
    active_last_week_df.columns = ['device', 'hour_slot', 'is_active_last_week']
    active_last_week_df = active_last_week_df.fillna(0)
    active_yesterday_df = all_data[['device', 'hour_slot', 'hour_of_day', 'day_num']]

    active_yesterday_df['lag_day_num'] = active_yesterday_df['day_num']-1

    # Join with last weeks data
    active_yesterday_df = pd.merge(
        active_yesterday_df,
        all_data[['device', 'hour_of_day', 'day_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_day', 'lag_day_num'],
        right_on=['device', 'hour_of_day', 'day_num'])

    # Clean up
    active_yesterday_df = active_yesterday_df[['device', 'hour_slot', 'is_active']]
    active_yesterday_df.columns = ['device', 'hour_slot', 'is_active_yesterday']
    active_yesterday_df = active_yesterday_df.fillna(0)

    all_data.sort_values(by='hour_slot', inplace=True, ascending=True)
    hour_of_day_activation_rate_df = all_data[['device', 'hour_slot', 'hour_of_day', 'day_num', 'is_active']]

    def mean_pre_now(x):
        return np.mean(x[:-1])


    hour_of_day_activation_rate_df_1 = hour_of_day_activation_rate_df.groupby(['device', 'hour_of_day']).agg({'is_active':mean_pre_now}).reset_index(drop=False)
    hour_of_day_activation_rate_df_2 = hour_of_day_activation_rate_df.groupby(['device', 'hour_of_day']).agg({'hour_slot':'max'}).reset_index(drop=False)

    hour_of_day_activation_rate_df = pd.merge(hour_of_day_activation_rate_df_1, hour_of_day_activation_rate_df_2,how='left')
    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df[['device', 'hour_slot', 'is_active']]
    hour_of_day_activation_rate_df.columns = ['device', 'hour_slot', 'daily_activation_rate']
    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df.fillna(0)
   
    all_data.sort_values(by='hour_slot', inplace=True, ascending=True)

    # Select only the required columns
    weekly_activation_rate_df = all_data[['device', 'hour_slot', 'hour_of_week', 'week_num', 'is_active']]
    # Group by device, hour_of_day
    # weekly_activation_rate_df = weekly_activation_rate_df.groupby(['device', 'day_num']).agg({'is_active':np.mean}).reset_index(drop=False)
    weekly_activation_rate_df_1 = weekly_activation_rate_df.groupby(['device', 'hour_of_week']).agg({'is_active':mean_pre_now}).reset_index(drop=False)
    weekly_activation_rate_df_2 = weekly_activation_rate_df.groupby(['device', 'hour_of_week']).agg({'hour_slot':'max'}).reset_index(drop=False)

    weekly_activation_rate_df = pd.merge(weekly_activation_rate_df_1, weekly_activation_rate_df_2,how='left')
    weekly_activation_rate_df = weekly_activation_rate_df[['device', 'hour_slot', 'is_active']]
    weekly_activation_rate_df.columns = ['device', 'hour_slot', 'weekly_activation_rate']
    weekly_activation_rate_df = weekly_activation_rate_df.fillna(0)

    # Get average daily activations for each device
    weekly_device_activation_rate_df = all_data.groupby(['device', 'day_num']).agg({'is_active':'mean'}).reset_index(drop=False)
    weekly_device_activation_rate_df.columns = ['device', 'day_num', 'activation_rate']

    weekly_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_device_activation_rate_df['weeks_activation_rate'] = weekly_device_activation_rate_df \
        .groupby('device')['activation_rate'] \
        .rolling(7).mean() \
        .reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_device_activation_rate_df['lead_day_num'] = weekly_device_activation_rate_df['day_num'] + 1

    weekly_device_activation_rate_df = pd.merge(
        all_data[['device', 'hour_slot', 'day_num']],
        weekly_device_activation_rate_df,
        how='left',
        left_on=['device', 'day_num'],
        right_on=['device', 'lead_day_num'])

    keep_cols = ['device','hour_slot','activation_rate','weeks_activation_rate']
    weekly_device_activation_rate_df = weekly_device_activation_rate_df[keep_cols]
    weekly_device_activation_rate_df.columns = ['device','hour_slot','yesterdays_device_activation_rate',
                                                'last_weeks_device_activation_rate']

    weekly_all_device_activation_rate_df = all_data.groupby('day_num').agg({'is_active':'mean'}).reset_index(drop=False)
    weekly_all_device_activation_rate_df.columns = ['day_num', 'activation_rate']

    weekly_all_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_all_device_activation_rate_df['weeks_activation_rate'] = weekly_all_device_activation_rate_df[
        'activation_rate'] \
        .rolling(7).mean() \
        .reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_all_device_activation_rate_df['lead_day_num'] = weekly_all_device_activation_rate_df['day_num'] + 1

    weekly_all_device_activation_rate_df = pd.merge(
        all_data[['device', 'hour_slot', 'day_num']],
        weekly_all_device_activation_rate_df,
        how='left',
        left_on='day_num',
        right_on='lead_day_num')

    keep_cols = ['device','hour_slot','activation_rate','weeks_activation_rate']
    weekly_all_device_activation_rate_df = weekly_all_device_activation_rate_df[keep_cols]
    weekly_all_device_activation_rate_df.columns = ['device','hour_slot','yesterdays_all_device_activation_rate',
        'last_weeks_all_device_activation_rate']

    modeling_df = pd.merge(all_data,active_last_week_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])
    
    modeling_df = pd.merge(modeling_df,active_yesterday_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])
    modeling_df = pd.merge(modeling_df,hour_of_day_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])

    modeling_df = pd.merge(modeling_df,weekly_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])

    modeling_df = pd.merge(modeling_df,weekly_device_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])

    modeling_df = pd.merge(modeling_df,weekly_all_device_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot'])
    modeling_df = modeling_df.fillna(0).drop_duplicates()

    predict_df = modeling_df[modeling_df['type']== 'predict']
    predict_df = predict_df[[
                    'device',
                    'hour_of_day',
                    'hour_of_week',
                    'day_of_week',
                    'is_active_yesterday',
                    'is_active_last_week',
                    'daily_activation_rate',
                    'weekly_activation_rate'
                ]]
    data = pd.get_dummies(predict_df, prefix_sep = '-')
    loaded_model = pickle.load(open('model_pickle.pk1' , 'rb'))
    predicted = loaded_model.predict(data)
    predicted = pd.Series(predicted, index=data.index)

    results_data = pd.DataFrame(predicted).join(
        modeling_df[['device', 'hour_slot']],
        how = 'left',)
    starting_time_dt = dt.datetime.strptime(starting_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    #Clean up the final dataframe and write to csv at the specified location
    def convert_hour_slot_to_ts(hour_slot):
        return dt.datetime.strftime( starting_time_dt + dt.timedelta(hours = hour_slot), '%Y-%m-%d %H:%M:%S')

    results_data['hour_slot'] = results_data['hour_slot'].apply(convert_hour_slot_to_ts)

    results_data.columns = ['activation_predicted','device','time']
    results_data = results_data.sort_values(by = ['device', 'time'])
    json_data = results_data.to_json(orient='records')
    return json_data

            

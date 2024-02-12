import itertools
import sys
import logging

import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import sklearn
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
# import prophet
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

logging.basicConfig(level=logging.INFO)

def activation_data_process(device_activations, starting_time = None):
    '''
   This step involves pre-processing of the training data to get some basic time variables. 
   The way we need hourly predictions starting the next hour prompted me to get the hour slot for each hour in the training data
   and fill in the gaps with blank hour slots 
    '''

    logging.info('Transforming the activation data')

    device_activations.time = pd.to_datetime(device_activations.time)

    earliest_date = starting_time if starting_time is not None else min(device_activations.time).date()

    device_activations['date'] = device_activations['time'].dt.date
    device_activations['hour_of_day'] = device_activations['time'].dt.hour
    device_activations['hour_of_week'] = device_activations['time'].dt.dayofweek * 24 + device_activations[
        'hour_of_day']
    device_activations['dummy']= (device_activations['date']-earliest_date)
    device_activations['duration_timedelta'] = device_activations['dummy'].apply( lambda x:x.days )
    device_activations['hour_slot'] =  device_activations['duration_timedelta'] * 24 + device_activations[
    'hour_of_day']
    
    # Create a blank dataset for every hour and device
    all_devices = list(device_activations.device.unique())
    n_hours = ((max(device_activations.time) - min(device_activations.time)).days + 1) * 24
    hour_slot = list(range(n_hours))
    earliest_hour = min(device_activations.hour_slot)
    hour_slot = [x+earliest_hour for x in hour_slot]

    blank_df = pd.DataFrame(list(product(all_devices, hour_slot)), columns=['device', 'hour_slot'])
    blank_df['hour_of_day'] = blank_df.hour_slot % 24
    blank_df['hour_of_week'] = blank_df.hour_slot % (7 * 24)
    blank_df['day_num'] = np.floor(blank_df.hour_slot / 24).astype(int)
    blank_df['week_num'] = np.floor(blank_df.hour_slot / (24 * 7)).astype(int)
    blank_df['day_of_week'] = np.floor(blank_df.day_num % 7).astype(int)
    # Cleanup the activation data in order to join with the blank data

    grouped_df_1 = device_activations.groupby(['device', 'hour_slot']).agg({'time':'min'}).reset_index()
    grouped_df_1.columns = ['device','hour_slot','earliest_activation']

    grouped_df_2 = device_activations.groupby(['device', 'hour_slot']).agg({'time':'max'}).reset_index()
    grouped_df_2.columns = ['device','hour_slot','latest_activation']

    grouped_df_3 = device_activations.groupby(['device', 'hour_slot']).agg({'device_activated':'sum'}).reset_index()
    grouped_df_3.columns = ['device','hour_slot','total_activations']

    # Join with the activation data and cleanup
    df = pd.merge(blank_df, grouped_df_1,how='left').merge(grouped_df_2,  how='left').merge(grouped_df_3, how='left')
    df['is_active'] = (df.total_activations.isna() == False).astype(int)
    return df
'''The above returns the transformed training dataset according to hour slots'''

def activation_data_dummy(devices, start_time):
    '''
    Creating dummy activation data for the hourslots that we need to predict. This involves creating the next 24 hour slots for 
    each device. After this step the returned df is again processed like the training dataset to get the same set of inputs
    for prediction
    '''
    logging.info('Creating fake activation data for the prediction data')
    next_24_hours = pd.date_range(start_time, periods=24, freq='h').ceil('h')
    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, devices))
    df = pd.DataFrame(xproduct, columns=['time', 'device'])
    df['device_activated'] = -1
    df.columns = ['time', 'device', 'device_activated']
    return df

def extract_modeling_data(df):
    '''This step involves creating more features ( lag features and different activation rates ) from the 
    transformed training data set.'''

    logging.info('Extracting the modeing data from the transformed activations')

    #checking activation status of the device as of -1 week (previous week)
    active_last_week_df = df[['device', 'hour_slot', 'hour_of_week', 'week_num']]
    active_last_week_df.loc[:,'lag_week_num'] = active_last_week_df['week_num'] - 1

    # Join with last weeks data
    active_last_week_df = pd.merge(
        active_last_week_df,
        df[['device', 'hour_of_week', 'week_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_week', 'lag_week_num'],
        right_on=['device', 'hour_of_week', 'week_num']
    )

    # Clean up
    active_last_week_df = active_last_week_df[['device', 'hour_slot', 'is_active']]
    active_last_week_df.columns = ['device', 'hour_slot', 'is_active_last_week']
    active_last_week_df = active_last_week_df.fillna(0)


    #Checking activation status of the device as of -1 day ( yesterday)
    # Select only the required columns
    active_yesterday_df = df[['device', 'hour_slot', 'hour_of_day', 'day_num']]
    active_yesterday_df['lag_day_num'] = active_yesterday_df['day_num']-1

    # Join with last weeks data
    active_yesterday_df = pd.merge(
        active_yesterday_df,
        df[['device', 'hour_of_day', 'day_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_day', 'lag_day_num'],
        right_on=['device', 'hour_of_day', 'day_num']
    )

    # Clean up
    active_yesterday_df = active_yesterday_df[['device', 'hour_slot', 'is_active']]
    active_yesterday_df.columns = ['device', 'hour_slot', 'is_active_yesterday']
    active_yesterday_df = active_yesterday_df.fillna(0)


    '''The next steps involve calculation of different activation rates for all devices/ stand alone devices 
    for different timelines of day,previous day, week & previous week'''

    df.sort_values(by='hour_slot', inplace=True, ascending=True)
    hour_of_day_activation_rate_df = df[['device', 'hour_slot', 'hour_of_day', 'day_num', 'is_active']]

    # Custom fucntion to calculate mean to avoid current value being used in the calculation
    def mean_pre_now(x):
        return np.mean(x[:-1])

    # Group by device, hour_of_day
    hour_of_day_activation_rate_df_1 = hour_of_day_activation_rate_df.groupby(['device', 'hour_of_day']).agg({'is_active':mean_pre_now}).reset_index(drop=False)
    hour_of_day_activation_rate_df_2 = hour_of_day_activation_rate_df.groupby(['device', 'hour_of_day']).agg({'hour_slot':'max'}).reset_index(drop=False)

    hour_of_day_activation_rate_df = pd.merge(hour_of_day_activation_rate_df_1, hour_of_day_activation_rate_df_2,how='left')
    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df[['device', 'hour_slot', 'is_active']]
    hour_of_day_activation_rate_df.columns = ['device', 'hour_slot', 'daily_activation_rate']
    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df.fillna(0)
   
    #average activation rate for this device at this time of the week
    df.sort_values(by='hour_slot', inplace=True, ascending=True)
    weekly_activation_rate_df = df[['device', 'hour_slot', 'hour_of_week', 'week_num', 'is_active']]

    # Group by device, hour_of_day
    weekly_activation_rate_df_1 = weekly_activation_rate_df.groupby(['device', 'hour_of_week']).agg({'is_active':mean_pre_now}).reset_index(drop=False)
    weekly_activation_rate_df_2 = weekly_activation_rate_df.groupby(['device', 'hour_of_week']).agg({'hour_slot':'max'}).reset_index(drop=False)

    weekly_activation_rate_df = pd.merge(weekly_activation_rate_df_1, weekly_activation_rate_df_2,how='left')
    weekly_activation_rate_df = weekly_activation_rate_df[['device', 'hour_slot', 'is_active']]
    weekly_activation_rate_df.columns = ['device', 'hour_slot', 'weekly_activation_rate']
    weekly_activation_rate_df = weekly_activation_rate_df.fillna(0)

    #average activation rate for this device for the last week

    # Get average daily activations for each device
    weekly_device_activation_rate_df = df.groupby(['device', 'day_num']).agg({'is_active':'mean'}).reset_index(drop=False)
    weekly_device_activation_rate_df.columns = ['device', 'day_num', 'activation_rate']

    weekly_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_device_activation_rate_df['weeks_activation_rate'] = weekly_device_activation_rate_df \
        .groupby('device')['activation_rate'] \
        .rolling(7).mean() \
        .reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_device_activation_rate_df['lead_day_num'] = weekly_device_activation_rate_df['day_num'] + 1

    weekly_device_activation_rate_df = pd.merge(
        df[['device', 'hour_slot', 'day_num']],
        weekly_device_activation_rate_df,
        how='left',
        left_on=['device', 'day_num'],
        right_on=['device', 'lead_day_num']
    )

    # Clean up
    keep_cols = ['device','hour_slot','activation_rate','weeks_activation_rate']
    weekly_device_activation_rate_df = weekly_device_activation_rate_df[keep_cols]
    weekly_device_activation_rate_df.columns = ['device','hour_slot','yesterdays_device_activation_rate','last_weeks_device_activation_rate'
    ]


    #What's the average activation rate for all rooms for the last week

    # Get average daily activations for each device
    weekly_all_device_activation_rate_df = df.groupby('day_num').agg({'is_active':'mean'}).reset_index(drop=False)
    weekly_all_device_activation_rate_df.columns = ['day_num', 'activation_rate']

    weekly_all_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_all_device_activation_rate_df['weeks_activation_rate'] = weekly_all_device_activation_rate_df[
        'activation_rate'].rolling(7).mean().reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_all_device_activation_rate_df['lead_day_num'] = weekly_all_device_activation_rate_df['day_num'] + 1

    weekly_all_device_activation_rate_df = pd.merge(
        df[['device', 'hour_slot', 'day_num']],
        weekly_all_device_activation_rate_df,
        how='left',left_on='day_num',right_on='lead_day_num')

    # Clean up
    keep_cols = ['device','hour_slot','activation_rate','weeks_activation_rate']
    weekly_all_device_activation_rate_df = weekly_all_device_activation_rate_df[keep_cols]
    weekly_all_device_activation_rate_df.columns = ['device','hour_slot','yesterdays_all_device_activation_rate',
                                                    'last_weeks_all_device_activation_rate']



    #Join all of the dataframes into one single modeling dataset
    # Add in whether the device was active this time last week
    modeling_df = pd.merge(df,active_last_week_df,how='left',left_on=['device', 'hour_slot'],right_on=['device', 'hour_slot'])
    # Add in whether the device was active this time yesterday
    modeling_df = pd.merge(modeling_df,active_yesterday_df,how='left',left_on=['device', 'hour_slot'],
                           right_on=['device', 'hour_slot'])

    # Add in average previous activation rate for this time of day
    modeling_df = pd.merge(modeling_df,hour_of_day_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],right_on=['device', 'hour_slot'])

    # Add in average previous activation rate for this time of day & day of week
    modeling_df = pd.merge(modeling_df,weekly_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],right_on=['device', 'hour_slot'])

    # Add in average previous activation rate for this device for yesterday & last week
    modeling_df = pd.merge(modeling_df,weekly_device_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],right_on=['device', 'hour_slot'])

    # Add in average previous activation rate for all device for yesterday & last week
    modeling_df = pd.merge(modeling_df,weekly_all_device_activation_rate_df,how='left',
        left_on=['device', 'hour_slot'],right_on=['device', 'hour_slot'])
    return modeling_df.fillna(0).drop_duplicates()

class Model():
    def __init__(self, data_df):
        
        self.data_df = data_df

    def fit_prophet(self, df):
        """
        :param df: Dataframe (train + test data)
        :return: predictions as defined in the output schema
        """

        def train_fitted_prophet(df):


            # train
            ts_train = (df.query("type=='train'")
                        .rename(columns={'time_slot': 'ds', 'is_active': 'y'})
                        .sort_values('ds')
                        )
            # test
            ts_test = (df.query("type=='predict'")
                       .rename(columns={'time_slot': 'ds', 'is_active': 'y'})
                       .sort_values('ds')
                       )


            m = Prophet(growth='flat', 
                        weekly_seasonality=True,
                        daily_seasonality=True,

                       )
            m.add_regressor('hour_of_day')
            m.add_regressor('hour_of_week')
            m.add_regressor('day_num')
            m.add_regressor('week_num')
            m.add_regressor('is_active_last_week')
            m.add_regressor('is_active_yesterday')
            m.add_regressor('daily_activation_rate')
            m.add_regressor('weekly_activation_rate')
            m.fit(ts_train)


            # at this step we predict the future and we get plenty of additional columns be cautious
            ts_hat = m.predict(ts_test)[["ds", "yhat"]]
            print(ts_hat)

            # ts_train_pred = m.predict(ts_train)[["ds", "yhat"]]
            # print(ts_train_pred)

            # fpr, tpr, thresholds = roc_curve(ts_train['y'], predictions['activation_predicted'])
            # roc_auc = auc(fpr, tpr)
            # print('AUC: {}'.format(roc_auc))
            return ts_hat
            # return ts_train_pred

        return train_fitted_prophet(df)
    
    def train_predict(self):
        p = list()
        for device in list(self.data_df.device.unique()):
            predictions = self.fit_prophet(self.data_df.query(f"device=='{device}'"))
            predictions['yhat'] = predictions['yhat'].clip(lower=0)
            predictions['device'] = device
            predictions.columns = ['time', 'activation_predicted' , 'device']
            p.append(predictions)

        p = pd.concat(p, ignore_index=True)
        
        return p

        
if __name__ == '__main__':
    #Extract the arguments
    pred_time, in_file, out_file = sys.argv[1:]
    device_activations = pd.read_csv(in_file)
    device_activations.time = pd.to_datetime(device_activations.time)

    logging.info('activation data. Shape: {}'.format(device_activations.shape))

    #Variables that are used in the above functions
    starting_time = min(device_activations.time).date()
    devices = list(device_activations['device'].unique())

    train_data = activation_data_process(device_activations)
    predict_data = activation_data_process(activation_data_dummy(devices,pred_time),starting_time=starting_time)
    train_data['type'] = 'train'
    predict_data['type'] = 'predict'
    pred_data_start_hour = min(predict_data['hour_slot'])

    logging.info('Created the training & prediction datasets')
    all_data = train_data._append(predict_data)

    #Create the modeling dataset
    modeling_data = extract_modeling_data(all_data)
    starting_time_dt = dt.datetime.strptime(starting_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    #Clean up the final dataframe and write to csv at the specified location
    def convert_hour_slot_to_ts(hour_slot):
        return dt.datetime.strftime( starting_time_dt + dt.timedelta(hours = hour_slot), '%Y-%m-%d %H:%M:%S')

    modeling_data['time_slot'] = modeling_data['hour_slot'].apply(convert_hour_slot_to_ts)
    model = Model(data_df=modeling_data)
    predictions = model.train_predict()
    predictions['activation_predicted1'] = (predictions['activation_predicted'] >= .4).astype(int)

    # ts_test = (modeling_data.query("type=='predict'")
    #                     .rename(columns={'time_slot': 'ds', 'is_active': 'y'})
    #                     .sort_values('ds')
    #                     )
    # fpr, tpr, thresholds = roc_curve(ts_test['y'], predictions['activation_predicted'])
    # roc_auc = auc(fpr, tpr)
    # print('AUC: {}'.format(roc_auc))
    predictions = predictions.drop('activation_predicted', axis=1) 
    predictions.sort_values(by = ['device', 'time'])
    predictions.to_csv(out_file, index=False)
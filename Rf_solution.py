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

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix


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

def get_config(pred_data_start_hour):
    ''' setting up the config for the model params to train the model by using the input timestamp as the cut off
    for train and test datasets.
    '''
    logging.info('Getting the config')

    config = {
        'data': {
            'splits': {
                'train': {'test_perc': 20,'start_hour': 168,'devices': 'all','type': 'train'},
                'predict': {
                    'test_perc': 0.0,
                    'start_hour': pred_data_start_hour,
                    'devices': 'all',
                    'type': 'predict'
                }
            },
            'xy': {
                'y_col': 'is_active',
                'X_cols': [
                    'device',
                    'hour_of_day',
                    'hour_of_week',
                    'day_of_week',
                    'is_active_yesterday',
                    'is_active_last_week',
                    'daily_activation_rate',
                    'weekly_activation_rate'
                ]
            }
        },
        'model': {
            'data_split': 'train',
            'model_class': RandomForestClassifier,
            'paramaters': {
                'n_estimators': 512,
                'max_depth': 8,
                'min_samples_split': 2,
                'max_features': 4
            }
        }
    }

    return config

def build_model(data, config):
    '''
    Training the model.
    '''
    logging.info('Building the model')

    model = Model(data, config)
    return model

class Data():
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        logging.info('Created the data')

class Model():
    '''
    central class that will train the model
    '''

    def __init__(self, data_df, config):

        logging.info('Initializing the model')

        self.base_data = data_df
        self.config = config

        self._generate_data_splits(config['data'])

        logging.info('Training the model')
        self._build_model(config['model'])

    def _generate_data_splits(self, config):

        splits_config = config['splits']
        splits = {}
        for name, params in splits_config.items():
            data = self.base_data

            start_hour = params['start_hour'] if 'start_hour' in params else 0
            end_hour = params['end_hour'] if 'end_hour' in params else np.inf

            rows = (data['hour_slot'] >= start_hour) & (data['hour_slot'] <= end_hour)
            data = data[rows]

            if params['devices'] != 'all':
                rows = data['device'] in params['devices']
                data = data[rows]

            if 'type' in params:
                rows = data['type'] == params['type']
                data = data[rows]

            xy_config = config['xy']
            y = data[xy_config['y_col']]
            X = data[xy_config['X_cols']]

            test_size = params['test_perc']
            
            if test_size != 0:

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            else:
                X_train = X
                y_train = y
                X_test = X
                y_test = y

            data = Data(X_train, X_test, y_train, y_test)
            splits[name] = data
            self.data_splits = splits

    def _build_model(self, model_config):

        model = model_config['model_class']
        model = model(**model_config['paramaters'])

        data = self.data_splits[model_config['data_split']]

        X_train = data.X_train
        X_test = data.X_test

        X_train = pd.get_dummies(X_train, prefix_sep ='-')
        X_test = pd.get_dummies(X_test, prefix_sep ='-')

        y_train = data.y_train
        y_test = data.y_test

        na_train_rows = X_train.isnull().any(axis=1)
        na_test_rows = X_test.isnull().any(axis=1)

        X_train = X_train[na_train_rows == False]
        y_train = y_train[na_train_rows == False]
        X_test = X_test[na_test_rows == False]
        y_test = y_test[na_test_rows == False]

        data.X_test = X_test
        data.X_train = X_train
        data.y_train = y_train
        data.y_test = y_test

        model.fit(X_train, y_train)
        self.model = model


    


    # def plot_roc_curve(self, data_split='train', use_test=False):

    #     data = self.data_splits[data_split]
    #     X, y = (data.X_test, data.y_test) if use_test else (data.X_train, data.y_train)
    #     X = pd.get_dummies(X, prefix_sep ='-')
    #     predicted = self.model.predict_proba(X)[:, 1]
    #     fpr, tpr, thresholds = roc_curve(y, predicted)
    #     roc_auc = auc(fpr, tpr)
    #     print('AUC: {}'.format(roc_auc))

    #     # print(X[X['device-device_4'] ==1])
    #     # print(y.shape)
    #     predicted = self.model.predict(X)
    #     predicted = pd.Series(predicted, index=X.index)
    #     accuracy = accuracy_score(y, predicted, normalize=True)
    #     print('accuracy: {}'.format(accuracy))

    def plot_roc_curve(self, data_split='train', use_test=False):
        data = self.data_splits[data_split]
        X, y = (data.X_test, data.y_test) if use_test else (data.X_train, data.y_train)

        # Assuming 'devices' is a column in your data
        # X['device'] = device

        X = pd.get_dummies(X, prefix_sep='-')
        predicted_proba = self.model.predict_proba(X)[:, 1]

        # Get the device-specific columns
        device_columns = [col for col in X.columns if col.startswith('device-')]

        # Initialize dictionaries to store results for each device
        device_roc_auc = {}
        device_accuracy = {}
        device_confusion_matrix = {}

        for device_col in device_columns:
            # Extract the device number from the column name
            device_number = device_col.split('_')[-1]

            # Filter data for the current device
            device_indices = X[device_col] == 1
            device_X = X[device_indices]
            device_y = y[device_indices]

            # Calculate ROC curve and AUC for the current device
            fpr, tpr, thresholds = roc_curve(device_y, predicted_proba[device_indices])
            roc_auc = auc(fpr, tpr)
            device_roc_auc[device_number] = roc_auc

            # Calculate accuracy for the current device
            device_predicted = self.model.predict(device_X)
            device_accuracy[device_number] = accuracy_score(device_y, device_predicted, normalize=True)

            cm = confusion_matrix(device_y, device_predicted)
            device_confusion_matrix[device_number] = cm

        # Print or use the device-wise metrics as needed
        for device, roc_auc in device_roc_auc.items():
            print('Device {}: AUC: {}'.format(device, roc_auc))
        
        for device, accuracy in device_accuracy.items():
            print('Device {}: Accuracy: {}'.format(device, accuracy))

        for device, cm in device_confusion_matrix.items():
            print('Device {}: Confusion Matrix:\n{}'.format(device, cm))

        for device, cm in device_confusion_matrix.items():
            total_samples = cm.sum()
            cm_percent = (cm / total_samples) * 100  # Convert to percentage
            print('Device {}: Confusion Matrix (%):\n{}'.format(device, cm_percent))

    def predict(self, data_split = 'predict', use_test=False, predict_proba = False):

        data = self.data_splits[data_split]
        X, y = (data.X_test, data.y_test) if use_test else (data.X_train, data.y_train)
        X = pd.get_dummies(X, prefix_sep = '-')

        predicted = self.model.predict_proba(X)[:, 1] if predict_proba else self.model.predict(X)
        predicted = pd.Series(predicted, index=X.index)

        return predicted
        

if __name__ == '__main__':
    #Extract the arguments
    pred_time, in_file, out_file = sys.argv[1:]

    #Load the activation data and format the time correctly
    device_activations = pd.read_csv(in_file)
    device_activations.time = pd.to_datetime(device_activations.time)

    logging.info('activation data. Shape: {}'.format(device_activations.shape))

    #Variables that are used in the above functions
    starting_time = min(device_activations.time).date()
    devices = list(device_activations['device'].unique())

    #Create the training & prediction data
    train_data = activation_data_process(device_activations)
    predict_data = activation_data_process(activation_data_dummy(devices, pred_time),starting_time=starting_time)
    train_data['type'] = 'train'
    predict_data['type'] = 'predict'
    pred_data_start_hour = min(predict_data['hour_slot'])

    logging.info('training & prediction datasets created')

    #Append the datasets together
    all_data = train_data._append(predict_data)
    #Create the modeling dataset
    modeling_data = extract_modeling_data(all_data)

    logging.info('Model dataset created ')

    config = get_config(pred_data_start_hour)
    model = build_model(modeling_data, config)

    logging.info('Model training done')

    #Predict for the 'predict' data
    preds = model.predict()
    model.plot_roc_curve()

    logging.info('Predicted for the desired day')

    #Create the final dataframe
    results_data = pd.DataFrame(preds).join(
        modeling_data[['device', 'hour_slot']],
        how = 'left',)

    starting_time_dt = dt.datetime.strptime(starting_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    #Clean up the final dataframe and write to csv at the specified location
    def convert_hour_slot_to_ts(hour_slot):
        return dt.datetime.strftime( starting_time_dt + dt.timedelta(hours = hour_slot), '%Y-%m-%d %H:%M:%S')

    results_data['hour_slot'] = results_data['hour_slot'].apply(convert_hour_slot_to_ts)
    results_data.columns = ['activation_predicted','device','time']

    results_data.sort_values(by = ['device', 'time'])
    results_data.to_csv(out_file, index=False)
    logging.info('Predictions saved to: {}'.format(out_file))
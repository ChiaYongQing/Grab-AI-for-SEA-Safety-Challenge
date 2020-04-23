## Grab AI for SEA - Safety Challenge

### Import Libraries
```
import numpy as np
import pandas as pd
import os
import glob
#os.chdir("./data/features/")
```
### Load data 
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

features = pd.concat([pd.read_csv(f) for f in all_filenames])
features = features.sort_values(['bookingID', 'second'])
features.shape

### Feature Engineering
```
# Calculating change in acceleration using sliding window technique (size = 5s)
xAcc_threshold = 2.873
yAcc_threshold = 2.7223
zAcc_threshold = 3.4262

features['acc_x_win5_max'] = features.groupby('bookingID').acceleration_x.rolling(5).max().values
features['acc_y_win5_max'] = features.groupby('bookingID').acceleration_y.rolling(5).max().values
features['acc_z_win5_max'] = features.groupby('bookingID').acceleration_z.rolling(5).max().values

features['acc_x_win5_min'] = features.groupby('bookingID').acceleration_x.rolling(5).min().values
features['acc_y_win5_min'] = features.groupby('bookingID').acceleration_y.rolling(5).min().values
features['acc_z_win5_min'] = features.groupby('bookingID').acceleration_z.rolling(5).min().values

features['acc_x_change'] = features['acc_x_win5_max'] - features['acc_x_win5_min']
features['acc_y_change'] = features['acc_y_win5_max'] - features['acc_y_win5_min']
features['acc_z_change'] = features['acc_z_win5_max'] - features['acc_z_win5_min']

features['acc_x_change_>threshold'] = np.where(features['acc_x_change'] > xAcc_threshold, 1, 0)
features['acc_y_change_>threshold'] = np.where(features['acc_y_change'] > yAcc_threshold, 1, 0)
features['acc_z_change_>threshold'] = np.where(features['acc_z_change'] > zAcc_threshold, 1, 0)
```
```
print(features['acc_x_change'].quantile(0.9))
print(features['acc_y_change'].quantile(0.9))
print(features['acc_z_change'].quantile(0.9))
```
```
ride_features = pd.DataFrame()
ride_features['bookingID'] = features.bookingID.unique()
ride_features.index = ride_features['bookingID']

ride_features['xAcc_>threshold'] =  features.groupby(['bookingID']).sum()['acc_x_change_>threshold']
ride_features['yAcc_>threshold'] =  features.groupby(['bookingID']).sum()['acc_y_change_>threshold']
ride_features['zAcc_>threshold'] =  features.groupby(['bookingID']).sum()['acc_z_change_>threshold']
```
```
# Normal values of gyrometer should be close to 0 at all times - we needed to determine a threshold by counting outlying driving behaviours based on telematics readings. For example, if the 25th and 75th percentile of training data gyroscope readings are -0.03 rad/s and 0.03 rad/s respectively, we will count the number of times a driver exceed these limit (lower than -0.03 rad/s or higher than 0.03 rad/s) throughout a trip. Exceeding the norms might suggest that the driver is behaving dangerously, such as rapid turns and fast lane-changes.

xGyr_threshold = 0.3818
yGyr_threshold = 0.3948
zGyr_threshold = 0.2425

features['gyr_x_win5_max'] = features.groupby('bookingID').gyro_x.rolling(5).max().values
features['gyr_y_win5_max'] = features.groupby('bookingID').gyro_y.rolling(5).max().values
features['gyr_z_win5_max'] = features.groupby('bookingID').gyro_z.rolling(5).max().values

features['gyr_x_win5_min'] = features.groupby('bookingID').gyro_x.rolling(5).min().values
features['gyr_y_win5_min'] = features.groupby('bookingID').gyro_y.rolling(5).min().values
features['gyr_z_win5_min'] = features.groupby('bookingID').gyro_z.rolling(5).min().values

features['gyr_x_change'] = features['gyr_x_win5_max'] - features['gyr_x_win5_min']
features['gyr_y_change'] = features['gyr_y_win5_max'] - features['gyr_y_win5_min']
features['gyr_z_change'] = features['gyr_z_win5_max'] - features['gyr_z_win5_min']

features['gyr_x_change_>threshold'] = np.where(features['gyr_x_change'] > xGyr_threshold, 1, 0)
features['gyr_y_change_>threshold'] = np.where(features['gyr_y_change'] > yGyr_threshold, 1, 0)
features['gyr_z_change_>threshold'] = np.where(features['gyr_z_change'] > zGyr_threshold, 1, 0)
```
```
print(features['gyr_x_change'].quantile(0.9))
print(features['gyr_y_change'].quantile(0.9))
print(features['gyr_z_change'].quantile(0.9))
```
ride_features['xGyr_>threshold'] =  features.groupby(['bookingID']).sum()['gyr_x_change_>threshold']
ride_features['yGyr_>threshold'] =  features.groupby(['bookingID']).sum()['gyr_y_change_>threshold']
ride_features['zGyr_>threshold'] =  features.groupby(['bookingID']).sum()['gyr_z_change_>threshold']
```
```
# We only dealt with speed metrics after calculating gyro & acc metrics, as gps loss only affects speed
gps_loss = features['Speed'] < 0
features[gps_loss].shape
```
```
gps_active = features['Speed'] >= 0
features = features[gps_active]
features.shape
```
### Defining function for feature engineering
```
def getTimeGap(data):
    time_difference = [None]
    time_difference[0] = data['second'].iloc[0]
    for i in range(1, len(data)):
        time_difference.append(data['second'].iloc[i] - data['second'].iloc[i-1])
        if time_difference[i] < 0:
            time_difference[i] = data['second'].iloc[i] - 0
    return(time_difference)
def getSpeedChange(data):
    difference = [None]
    difference[0] = data['Speed'].iloc[0]
    
    for i in range(1, len(data)):
        difference.append(data['Speed'].iloc[i] - data['Speed'].iloc[i-1])
        if data['time_difference'].iloc[i] <= 0:              
            difference[i] = data['Speed'].iloc[i] - 0
    return(difference)    
def getAcceleration(data):
    acceleration = [None]
    acceleration[0] = data['speed_change'].iloc[0]
    
    for i in range(1, len(data)):
        if data['time_difference'].iloc[i] != 0:
            acceleration.append(data['speed_change'].iloc[i]/data['time_difference'].iloc[i])
        else:
            acceleration.append(data['speed_change'].iloc[i])
    return(acceleration)
features['time_difference'] = getTimeGap(features)
features['speed_change'] = getSpeedChange(features)
features['acceleration'] = getAcceleration(features)
```
### Additional features
```
features['Resultant Acceleration'] = np.sqrt ((features['acceleration_x'])**2 + (features['acceleration_y'])**2 + (features['acceleration_z'])**2)
features['Resultant Gyro'] = np.sqrt ((features['gyro_x'])**2 + (features['gyro_y'])**2 + (features['gyro_z'])**2)
features['Resultant Acc & Gyro'] = features['Resultant Acceleration'] * features['Resultant Gyro']

def getAccelerationChange(data):
    acceleration_difference = [None]
    acceleration_difference[0] = data['Resultant Acceleration'].iloc[0]
    
    for i in range(1, len(data)):
        acceleration_difference.append(data['Resultant Acceleration'].iloc[i] - data['Resultant Acceleration'].iloc[i-1])
        if data['time_difference'].iloc[i] <= 0:              
            acceleration_difference[i] = data['Resultant Acceleration'].iloc[i] - 0
    return(acceleration_difference) 

def getBearingChange(data):
    bearing_difference = [None]
    bearing_difference[0] = data['Bearing'].iloc[0]
    
    for i in range(1, len(data)):
        bearing_difference.append(abs(data['Bearing'].iloc[i] - data['Bearing'].iloc[i-1]))
        if data['time_difference'].iloc[i] <= 0:              
            bearing_difference[i] = abs(data['Bearing'].iloc[i] - 0)
    return(bearing_difference)

def getBearingRate(data):
    bearingrate = [None]
    bearingrate[0] = data['bearing_difference'].iloc[0]
    
    for i in range(1, len(data)):
        if data['time_difference'].iloc[i] != 0:
            bearingrate.append(data['bearing_difference'].iloc[i]/data['time_difference'].iloc[i])
        else:
            bearingrate.append(data['bearing_difference'].iloc[i])
    return(bearingrate)

features['acceleration_difference'] = getAccelerationChange(features)
features['bearing_difference'] = getBearingChange(features)
features['bearingrate'] = getBearingRate(features)

features['xy_Acc']=np.sqrt((features['acceleration_x']**2)*(features['acceleration_y']**2))
features['xz_Acc']=np.sqrt((features['acceleration_x']**2)*(features['acceleration_z']**2))
features['yz_Acc']=np.sqrt((features['acceleration_y']**2)*(features['acceleration_z']**2))
```
### Removing entries with big time gap and impossible speed
```
# Removing datapoint with big time gap
big_timeDiff = features['time_difference'] > 30
features[big_timeDiff].shape

ok_timeDiff = features['time_difference'] < 30

features = features[ok_timeDiff]
features.shape

# remove entries with impossible speed i.e. 72m/s = 260km/h 
ok_speed = features['Speed'] < 72

features = features[ok_speed]
features.shape
```

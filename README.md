# Grab AI for SEA - Safety Challenge
## Data Preprocessing
### Import Libraries
```
import numpy as np
import pandas as pd
import os
import glob
#os.chdir("./data/features/")
```
### Load data 
```
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

features = pd.concat([pd.read_csv(f) for f in all_filenames])
features = features.sort_values(['bookingID', 'second'])
features.shape
```
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
### Data Cleaning - removing entries with big time gap and impossible speed
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
### More Features Engineering
```
# max speed
ride_features['maxSpeed'] = features.groupby(['bookingID']).max()['Speed']

# max speed in km/h (for visualisation purposes)
ride_features['maxSpeed_kmh'] = ride_features['maxSpeed'] * 3.6

# avg speed (excludes data points where vehicle is stationary)
moving = features['Speed'] > 0
ride_features['avgSpeed'] = features[moving].groupby(['bookingID']).mean()['Speed']

# duration in minutes
temp = features.groupby(['bookingID']).max()/60
ride_features['duration_minutes'] = temp['second']

# max acceleration
ride_features['maxAcceleration'] = features.groupby(['bookingID']).max()['acceleration']

# max deceleration
ride_features['maxDeceleration'] = -features.groupby(['bookingID']).min()['acceleration']
```
```
ride_features['maxResAcc'] = features.groupby(['bookingID']).max()['Resultant Acceleration']
ride_features['maxResGyr'] = features.groupby(['bookingID']).max()['Resultant Gyro']
ride_features['maxResAccGyr'] = features.groupby(['bookingID']).max()['Resultant Acc & Gyro']
ride_features['maxAccDiff'] = features.groupby(['bookingID']).max()['acceleration_difference']
ride_features['maxBearDiff'] = features.groupby(['bookingID']).max()['bearing_difference']
ride_features['maxBearRate'] = features.groupby(['bookingID']).max()['bearingrate']
ride_features['maxAccXY'] = features.groupby(['bookingID']).max()['xy_Acc']
ride_features['maxAccYZ'] = features.groupby(['bookingID']).max()['yz_Acc']
ride_features['maxAccXZ'] = features.groupby(['bookingID']).max()['xz_Acc']

ride_features['minResAcc'] = features.groupby(['bookingID']).min()['Resultant Acceleration']
ride_features['minResGyr'] = features.groupby(['bookingID']).min()['Resultant Gyro']
ride_features['minResAccGyr'] = features.groupby(['bookingID']).min()['Resultant Acc & Gyro']
ride_features['minAccDiff'] = features.groupby(['bookingID']).min()['acceleration_difference']
ride_features['minBearDiff'] = features.groupby(['bookingID']).min()['bearing_difference']
ride_features['minBearRate'] = features.groupby(['bookingID']).min()['bearingrate']
ride_features['minAccXY'] = features.groupby(['bookingID']).min()['xy_Acc']
ride_features['minAccYZ'] = features.groupby(['bookingID']).min()['yz_Acc']
ride_features['minAccXZ'] = features.groupby(['bookingID']).min()['xz_Acc']

ride_features['meanResAcc'] = features.groupby(['bookingID']).mean()['Resultant Acceleration']
ride_features['meanResGyr'] = features.groupby(['bookingID']).mean()['Resultant Gyro']
ride_features['meanResAccGyr'] = features.groupby(['bookingID']).mean()['Resultant Acc & Gyro']
ride_features['meanAccDiff'] = features.groupby(['bookingID']).mean()['acceleration_difference']
ride_features['meanBearDiff'] = features.groupby(['bookingID']).mean()['bearing_difference']
ride_features['meanBearRate'] = features.groupby(['bookingID']).mean()['bearingrate']
ride_features['meanAccXY'] = features.groupby(['bookingID']).mean()['xy_Acc']
ride_features['meanAccYZ'] = features.groupby(['bookingID']).mean()['yz_Acc']
ride_features['meanAccXZ'] = features.groupby(['bookingID']).mean()['xz_Acc']
```
### Load target variables 
```
os.chdir('..')
target = pd.read_csv('./labels/labels.csv')

# remove labels that appear more than once
filter2 = target.groupby('bookingID').filter(lambda x: len(x) > 1).bookingID.unique()
target = target[~target.bookingID.isin(filter2)]

# remove rides that do not have labels
filter3 = target.bookingID.unique()
X = ride_features[ride_features.bookingID.isin(filter3)]

# remove labels that do not correspond to any rides
filter4 = ride_features.bookingID.unique()
y = target[target.bookingID.isin(filter4)]
y = y.sort_values('bookingID')

# Now we have our X and Y ready for modelling
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
```
## Modelling
```
import numpy as np
import pandas as pd

X = pd.read_csv('./X.csv')
y = pd.read_csv('./y.csv')

# drop bookingID before train test split
X_train = X_train.drop(['bookingID'], axis = 1)
y_train = y_train.drop(['bookingID'], axis = 1)
X_test = X_test.drop(['bookingID'], axis = 1)
y_test = y_test.drop(['bookingID'], axis = 1)

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
```
### Modelling - Logistic Regression
```
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection 
from sklearn import ensemble, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
%matplotlib inline

# normalising everything
selected_cols = X_train.columns
min_max_scaler = preprocessing.MinMaxScaler()
X2_train = min_max_scaler.fit_transform(X_train)
X2_test = min_max_scaler.transform(X_test)

R_square = []
Alpha = list(np.linspace(0.001, 0.00001, 100, endpoint = True))

# using lasso for feature selection
for alpha in Alpha:
    lasso = linear_model.Lasso(alpha = alpha)
    lasso.fit(X2_train, y_train)
    R_square.append(lasso.score(X2_test, y_test))
    print('alpha:', alpha)
    print('R2 score:', lasso.score(X2_test, y_test))
    for i in range(len(selected_cols)):
        print(selected_cols[i], '\t', lasso.coef_[i])
```
```
# plot the Alpha vs. R-square curve for the model
plt.plot(Alpha, R_square, marker='.')
plt.plot([0.0001, 0.0001], [0.12, 0.136], linestyle='--')
plt.xlabel('alpha')
plt.ylabel('r_square')
plt.title('R-square vs Alpha')
# show the plot
plt.show()
```
```
selected variables from lasso @ alpha ~ 0.0001:
zAcc_>threshold 	 0.25427148495348156
xGyr_>threshold 	 -0.025717105765715214
yGyr_>threshold 	 0.17117881207732893
zGyr_>threshold 	 0.06445846413871234
maxSpeed 	 0.12506978166491503
avgSpeed 	 -0.4190286375598843
duration_minutes 	 2.4479167566976274
maxAcceleration 	 0.030834267778525416
maxDeceleration 	 0.0508859573422804
maxResGyr 	 0.0038434765356064695
maxAccDiff 	 0.009104738186662813
maxBearDiff 	 0.23459882257366277
maxBearRate 	 -0.2272838654013198
minResAcc 	 -0.06944215223932351
minAccDiff 	 -0.37813079336265937
minAccXY 	 -0.02882200667602575
minAccXZ 	 -0.10299503264922015
meanResAcc 	 -0.6201788810446625
meanAccDiff 	 0.7332140654089745
meanBearDiff 	 0.8622486587114753
meanBearRate 	 -0.6403330207910954
meanAccXZ 	 0.4946368371833259
```
```
selected_cols = ['zAcc_>threshold',
                 'xGyr_>threshold',
                 'yGyr_>threshold',
                 'zGyr_>threshold',
                 'maxSpeed',
                 'avgSpeed',
                 'duration_minutes',
                 'maxAcceleration',
                 'maxDeceleration',
                 'maxResGyr',
                 'maxAccDiff',
                 'maxBearDiff',
                 'maxBearRate',
                 'minResAcc',
                 'minAccDiff',
                 'minAccXY',
                 'minAccXZ',
                 'meanResAcc',
                 'meanAccDiff',
                 'meanBearDiff',
                 'meanBearRate',
                 'meanAccXZ'
                ]
```
```
X3 = X[selected_cols]
X3_train, X3_test = train_test_split(X3, test_size=0.2, random_state=2019)
X3_train = min_max_scaler.fit_transform(X3_train)
X3_test = min_max_scaler.transform(X3_test)
```
```


log = linear_model.LogisticRegression()

#LogisticRegression
log.fit(X3_train, y_train)
prb = log.predict_proba(X3_test)
danger = prb[:, 1:2]
pred = (log.predict_proba(X3_test)[:,1] >= 0.35).astype(bool)
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, danger)

auc_log = roc_auc_score(y_test, danger)
acc_log = accuracy_score(y_test, pred)
precision_log = precision_score(y_test, pred)
recall_log = recall_score(y_test, pred)
f2_log = metrics.fbeta_score(y_test, pred, beta = 2)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, pred), 
           accuracy_score(y_test, pred), 
           precision_score(y_test, pred),
           recall_score(y_test, pred),
           metrics.fbeta_score(y_test, pred, beta = 2))
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_log, tpr_log, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(roc_auc_score(y_test, danger))
print(auc_report)

# Accuracy: 0.7592082184916061
# Precision: 0.5
# Recall: 0.3850156087408949 
# F2: 0.40357766143106455
```
### Modelling - Neural Network
```
# Model architecture: 1 hidden layer, 10 neurons, batch size: 40, epochs: 50, kernel_initializer: normal

import os
import glob
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.constraints import maxnorm

from tensorflow import keras
%matplotlib inline
```
```
def create_model():

    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal', bias_initializer= 
      "zeros", input_dim=39))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal', bias_initializer= 
      "zeros"))
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    return classifier

seed = 2019
numpy.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X2_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```
```
model = Sequential()
#First Hidden Layer
model.add(Dense(30, activation='relu', kernel_initializer='normal', bias_initializer= 
  "zeros", input_dim=39))
model.add(Dropout(0.1))
#Output Layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal', bias_initializer= 
  "zeros"))
#Compiling the neural network
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

model.summary()
```
```
# Fitting the data to the training dataset
model.fit(X3_train,y_train,validation_data=(X3_test, y_test), batch_size=40, epochs=50)

eval_model=model.evaluate(X3_train, y_train)
eval_model

y_pred=model.predict(X3_test)
```
```
y_pred_threshold =(y_pred>0.35)
fpr_nn, tpr_nn, threshold_nn = roc_curve(y_test, y_pred)

auc_nn = roc_auc_score(y_test, y_pred)
acc_nn = accuracy_score(y_test, y_pred_threshold)
precision_nn = precision_score(y_test, y_pred_threshold)
recall_nn = recall_score(y_test, y_pred_threshold)
f2_nn = metrics.fbeta_score(y_test, y_pred_threshold, beta = 2)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, y_pred_threshold), 
           accuracy_score(y_test, y_pred_threshold), 
           precision_score(y_test, y_pred_threshold),
           recall_score(y_test, y_pred_threshold),
           metrics.fbeta_score(y_test, y_pred_threshold, beta = 2))
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_nn, tpr_nn, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(auc_nn)
print(auc_report)

# Accuracy: 0.7722375344525182
# Precision: 0.5410094637223974
#Recall: 0.35691987513007284 
# F2: 0.38298347476552025
```
```
# plot accuracy

from matplotlib import pyplot
%matplotlib inline

#eval result
history = model.fit(X3_train,y_train,validation_data=(X3_test, y_test), batch_size=40, epochs=50, verbose=0)

_, acc = model.evaluate(X3_train, y_train, verbose=0)
scores, histories = list(), list()
scores.append(acc)
histories.append(history)

axes = plt.gca()
axes.set_ylim([0.35,0.6])

# plot loss
pyplot.title('Cross Entropy Loss')
pyplot.plot(np.linspace(1,len(histories[0].history['loss']),
                        num=len(histories[0].history['loss'])),
            np.array(histories[0].history['loss']),
            color='blue', label='train')
pyplot.plot(np.linspace(1,len(histories[0].history['val_loss']),
                        num=len(histories[0].history['val_loss'])),
            np.array(histories[0].history['val_loss']),
            color='orange', label='test')
pyplot.xlabel("Epoch")
pyplot.ylabel("Loss")
pyplot.legend(loc='upper right')
pyplot.show()

axes = plt.gca()
axes.set_ylim([0.76,0.82])

pyplot.title('Classification Accuracy')
pyplot.plot(np.linspace(1,len(histories[0].history['acc']),
                        num=len(histories[0].history['acc'])),
            np.array(histories[0].history['acc']),
            color='blue', label='train')
pyplot.plot(np.linspace(1,len(histories[0].history['val_acc']),
                        num=len(histories[0].history['val_acc'])),
            np.array(histories[0].history['val_acc']),
            color='orange', label='test')
pyplot.xlabel("Epoch")
pyplot.ylabel("Accuracy")
pyplot.legend(loc='upper left')

pyplot.show()
```
## Modelling - Ensemble Learning
```
# Base decision tree
kf  = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)

dt1_parameters = [{'criterion': ['entropy', 'gini'],
                 'max_depth': list(np.linspace(3, 20, 18, endpoint = True)),
                 'min_samples_split': list(np.linspace(0.01, 0.2, 20))}]  

dt1 = model_selection.GridSearchCV(DecisionTreeClassifier(), dt1_parameters, scoring = 'roc_auc', cv = kf, n_jobs=-1)
dt1.fit(X_train, y_train.values.ravel())

print('best score:', dt1.best_score_)
print('best parameters: ', dt1.best_params_)

# best score: 0.695824851038972
# best parameters:  {'criterion': 'entropy', 'max_depth': 12.0, 'min_samples_split': 0.06999999999999999}
```
```
pred = (dt1.best_estimator_.predict_proba(X_test)[:,1] >= 0.35).astype(bool)
prb = dt1.predict_proba(X_test)
danger = prb[:, 1:2]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, danger)
precision, recall, thresholds = precision_recall_curve(y_test, danger)

auc_dt = roc_auc_score(y_test, danger)
acc_dt = accuracy_score(y_test, pred)
precision_dt = precision_score(y_test, pred)
recall_dt = recall_score(y_test, pred)
f2_dt = metrics.fbeta_score(y_test, pred, beta = 2)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, pred), 
           accuracy_score(y_test, pred), 
           precision_score(y_test, pred),
           recall_score(y_test, pred),
           metrics.fbeta_score(y_test, pred, beta = 2))
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_dt, tpr_dt, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(roc_auc_score(y_test, danger))
print(auc_report)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Curve')
plt.show()

# Accuracy: 0.7629666750187922
# Precision: 0.5115562403697997
# Recall: 0.34547346514047866 
# F2: 0.3694636100600935
```
```
# Plot Learning Curve
title = "Learning Curves"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=2019)

plot_learning_curve(dt1.best_estimator_, title = title, X = X_train, y = y_train, 
                    train_sizes = (50, 100, 150, 200, 300, 500, 1000, 2000, 3000), 
                    cv = cv, n_jobs = -1)
```
```
# Plot Decision Tree
from sklearn.tree import export_graphviz
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

graph = Source(tree.export_graphviz(dt1.best_estimator_,
                                    out_file=None,
                                    feature_names=X_train.columns,
                                    class_names=['no', 'yes'], 
                                    filled = True))

display(SVG(graph.pipe(format='svg')))
```
```
# Write to PDF
from io import StringIO
import pydot

dot_data = StringIO() 
tree.export_graphviz(dt1.best_estimator_, out_file=dot_data,
                     feature_names=X_train.columns,
                     class_names=['no', 'yes'],
                     filled = True)

graph2 = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph2[0].write_pdf("dt_tuned.pdf")
```
## Modelling - Random Forest
```
kf  = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)

rf1_parameters = [{'n_estimators': [500],
                 'criterion': ['entropy'],
                 'max_depth': [11, 12, 13, 14],
                 'bootstrap': [True],
                 'min_samples_split': [0.001, 0.002, 0.003, 0.004, 0.005],
                 'random_state': [2019]}]  

rforest1 = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), rf1_parameters, scoring = 'roc_auc', cv = kf, n_jobs=-1)
rforest1.fit(X_train, y_train.values.ravel())

print('best score:', rforest1.best_score_)
print('best parameters: ', rforest1.best_params_)
# best score: 0.7244339769673417
# best parameters:  {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 14, 'min_samples_split': 0.002, 'n_estimators': 500, 'random_state': 2019}
```
```
pred = (rforest1.best_estimator_.predict_proba(X_test)[:,1] >= 0.35).astype(bool)
prb = rforest1.best_estimator_.predict_proba(X_test)
danger = prb[:, 1:2]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, danger)

auc_rf = roc_auc_score(y_test, danger)
acc_rf = accuracy_score(y_test, pred)
precision_rf = precision_score(y_test, pred)
recall_rf = recall_score(y_test, pred)
f2_rf = metrics.fbeta_score(y_test, pred, beta = 2)

precision, recall, thresholds = precision_recall_curve(y_test, danger)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, pred), 
           accuracy_score(y_test, pred), 
           precision_score(y_test, pred),
           recall_score(y_test, pred),
           metrics.fbeta_score(y_test, pred, beta = 2))
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_rf, tpr_rf, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(roc_auc_score(y_test, danger))
print(auc_report)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Curve')
plt.show()

# Accuracy: 0.7599599097970433
# Precision: 0.5021707670043415
# Recall: 0.3610822060353798 
# F2: 0.38257993384785
```
```
# Ranking feature importance
varimp = list(zip(X_test.columns, rforest1.best_estimator_.feature_importances_))

def sortSecond(val): 
    return val[1] 

varimp.sort(key=sortSecond, reverse=True)

print(varimp[0:5])
```
## Modelling - XGBoost
```
import xgboost as xgb

kf  = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)

xgb1_parameters = [{'n_estimators': [500],
                    'max_depth': [7, 8, 9, 10, 11],
                    'learning_rate' : [0.001, 0.002, 0.003, 0.004, 0.005],
                    'random_state': [2019]}]

xgb1 = model_selection.GridSearchCV(xgb.XGBClassifier(), xgb1_parameters, n_jobs = -1, scoring = 'roc_auc', cv = kf)
xgb1.fit(X_train, y_train.values.ravel())

print('best score:', xgb1.best_score_)
print('best parameters: ', xgb1.best_params_)

# best score: 0.7261337081199871
# best parameters:  {'learning_rate': 0.005, 'max_depth': 7, 'n_estimators': 500, 'random_state': 2019}
```
```
pred = (xgb1.best_estimator_.predict_proba(X_test)[:,1] >= 0.35).astype(bool)
prb = xgb1.best_estimator_.predict_proba(X_test)
danger = prb[:, 1:2]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, danger)
precision, recall, threshold = precision_recall_curve(y_test, danger)

auc_xgb = roc_auc_score(y_test, danger)
acc_xgb = accuracy_score(y_test, pred)
precision_xgb = precision_score(y_test, pred)
recall_xgb = recall_score(y_test, pred)
f2_xgb = metrics.fbeta_score(y_test, pred, beta = 2)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, pred), 
           accuracy_score(y_test, pred), 
           precision_score(y_test, pred),
           recall_score(y_test, pred),
           metrics.fbeta_score(y_test, pred, beta = 2))
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_xgb, tpr_xgb, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(roc_auc_score(y_test, danger))
print(auc_report)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Curve')
plt.show()

# Accuracy: 0.7559508895013781
# Precision: 0.4912043301759134
# Recall: 0.3777315296566077 
# F2: 0.3960288020946979
```
```
# Rank feature importance
varimp = list(zip(X_test.columns, xgb1.best_estimator_.feature_importances_))

def sortSecond(val): 
    return val[1] 

varimp.sort(key=sortSecond, reverse=True)

print(varimp[0:5])

# [('duration_minutes', 0.08535824), ('avgSpeed', 0.0688276), ('maxSpeed', 0.062423415), ('meanBearDiff', 0.047557395), ('maxDeceleration', 0.040922016)]
```
## Modelling - Adaboost
```
kf  = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)

ab1_parameters = [{'n_estimators': [500],
                    'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5],
                    'random_state': [2019]}]

ab1 = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), ab1_parameters, n_jobs = -1, scoring = 'roc_auc', cv = kf)
ab1.fit(X_train, y_train.values.ravel())

# GridSearchCV(cv=StratifiedKFold(n_splits=10, random_state=2019, shuffle=True),
             error_score='raise-deprecating',
             estimator=AdaBoostClassifier(algorithm='SAMME.R',
                                          base_estimator=None,
                                          learning_rate=1.0, n_estimators=50,
                                          random_state=None),
             iid='warn', n_jobs=-1,
             param_grid=[{'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                          'n_estimators': [500], 'random_state': [2019]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='roc_auc', verbose=0)
             
print('best score:', ab1.best_score_)
print('best parameters: ', ab1.best_params_)

# best score: 0.727789066910167
# best parameters:  {'learning_rate': 0.2, 'n_estimators': 500, 'random_state': 2019}
```
```
pred = (ab1.best_estimator_.predict_proba(X_test)[:,1] >= 0.5).astype(bool)
prb = ab1.best_estimator_.predict_proba(X_test)
danger = prb[:, 1:2]
fpr_ab, tpr_ab, thresholds_ab = roc_curve(y_test, danger)
precision, recall, threshold = precision_recall_curve(y_test, danger)

auc_ab = roc_auc_score(y_test, danger)
acc_ab = accuracy_score(y_test, pred)
precision_ab = precision_score(y_test, pred)
recall_ab = recall_score(y_test, pred)
f2_ab = metrics.fbeta_score(y_test, pred, beta = 2)

report = """
The evaluation report is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {} 
F2: {}
""".format(confusion_matrix(y_test, pred), 
           accuracy_score(y_test, pred), 
           precision_score(y_test, pred),
           recall_score(y_test, pred),
           metrics.fbeta_score(y_test, pred, beta = 2)
          )
print(report)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_ab, tpr_ab, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(('Line of No Discrimination', 'ROC'))
plt.show()

auc_report = """
AUC: {}
""".format(roc_auc_score(y_test, danger))
print(auc_report)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Curve')
plt.show()

# Accuracy: 0.7902781257830118
# Precision: 0.7
# Recall: 0.22580645161290322 
# F2: 0.2611940298507463
```
```
#Rank feature importance
varimp = list(zip(X_test.columns, ab1.best_estimator_.feature_importances_))

def sortSecond(val): 
    return val[1] 

varimp.sort(key=sortSecond, reverse=True)

print(varimp[0:5])

# [('duration_minutes', 0.156), ('meanAccDiff', 0.106), ('avgSpeed', 0.082), ('maxDeceleration', 0.064), ('maxSpeed_kmh', 0.048)]
```
## Models Evaluation
```
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_log, tpr_log, marker='.')
plt.plot(fpr_nn, tpr_nn, marker='.')
plt.plot(fpr_dt, tpr_dt, marker='.')
plt.plot(fpr_rf, tpr_rf, marker='.')
plt.plot(fpr_xgb, tpr_xgb, marker='.')
plt.plot(fpr_ab, tpr_ab, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves of All Classifiers')
plt.legend(('Line of No Discrimination','Logistic Regression','Neural Network', 'Decision Tree', 'Random Forest', 'XGBoost', 'AdaBoost'))
plt.show()

report = """
The AUC for each classifier is:
Logistic Regression: {}
Neural Network: {}
Decision Tree: {} 
Random Forest: {}
XGBoost: {}
AdaBoost: {} 

The Accuracy for each classifier is:
Logistic Regression: {}
Neural Network: {}
Decision Tree: {} 
Random Forest: {}
XGBoost: {}
AdaBoost: {} 

The Precision for each classifier is:
Logistic Regression: {}
Neural Network: {}
Decision Tree: {} 
Random Forest: {}
XGBoost: {}
AdaBoost: {} 

The Recall for each classifier is:
Logistic Regression: {}
Neural Network: {}
Decision Tree: {} 
Random Forest: {}
XGBoost: {}
AdaBoost: {} 

The F2 for each classifier is:
Logistic Regression: {}
Neural Network: {}
Decision Tree: {} 
Random Forest: {}
XGBoost: {}
AdaBoost: {} 
""".format(auc_log, auc_nn, auc_dt, auc_rf, auc_xgb, auc_ab,
           acc_log, acc_nn, acc_dt, acc_rf, acc_xgb, acc_ab,
           precision_log, precision_nn, precision_dt, precision_rf, precision_xgb, precision_ab, 
           recall_log, recall_nn, recall_dt, recall_rf, recall_xgb, recall_ab, 
           f2_log, f2_nn, f2_dt, f2_rf, f2_xgb, f2_ab)
print(report)
```
The AUC for each classifier is:
Logistic Regression: 0.715092570651446
Neural Network: 0.7176850983745617
Decision Tree: 0.6998784269686075 
Random Forest: 0.7207295755590127
XGBoost: 0.7210795273075694
AdaBoost: 0.7243135759985988 

The Accuracy for each classifier is:
Logistic Regression: 0.7592082184916061
Neural Network: 0.7722375344525182
Decision Tree: 0.7629666750187922 
Random Forest: 0.7599599097970433
XGBoost: 0.7559508895013781
AdaBoost: 0.7902781257830118 

The Precision for each classifier is:
Logistic Regression: 0.5
Neural Network: 0.5410094637223974
Decision Tree: 0.5115562403697997 
Random Forest: 0.5021707670043415
XGBoost: 0.4912043301759134
AdaBoost: 0.7 

The Recall for each classifier is:
Logistic Regression: 0.3850156087408949
Neural Network: 0.35691987513007284
Decision Tree: 0.34547346514047866 
Random Forest: 0.3610822060353798
XGBoost: 0.3777315296566077
AdaBoost: 0.22580645161290322 

The F2 for each classifier is:
Logistic Regression: 0.40357766143106455
Neural Network: 0.38298347476552025
Decision Tree: 0.3694636100600935 
Random Forest: 0.38257993384785
XGBoost: 0.3960288020946979
AdaBoost: 0.2611940298507463 

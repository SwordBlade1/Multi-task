#!/usr/local/lib64/python3
# coding=utf-8

from xgboost import XGBRegressor as XGBR
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

#------------------------------------2 atomic numbers + 20 SNAP features---------------------------------#

dataset = pd.read_csv('/home/Multi-taskML/Merged2ID20SnapVolPeDF.csv', header = None)

cut_train = np.r_[     0:205828, 228698:423494, 445138:649170, 
                  671841:868112, 889920:1085347, 1107062:1311452, 
                  1334162:1534834, 1557131:1814981, 1843631:2107115
                  ]
cut_test = np.r_[        205828: 228698,423494: 445138,649170:671841, 
                         868112: 889920,1085347: 1107062,1311452:1334162, 
                          1534834: 1557131,1814981: 1843631,2107115:2136392
                 ]
selected_features = np.r_[0 : 22]

X_train = dataset.iloc[cut_train, selected_features].values
y_train = dataset.iloc[cut_train, 25].values

X_test = dataset.iloc[cut_test, selected_features].values
y_test = dataset.iloc[cut_test, 25].values


# Because of the particularity of dataset, GridSearch/CV is not applicable
R2_train_dict = dict()
R2_test_dict = dict()
for N_estimators in np.arange(600, 1100, 100):
    for Max_depth in np.arange(7, 17, 2):
        for Learning_rate in np.arange(0.04, 0.12001, 0.002):
            for Subsample in np.arange(0.55, 0.9001, 0.05):
                for Max_delta_step in np.arange(0, 0.4001, 0.05):
          
                    reg= XGBR(n_estimators = N_estimators,
                            learning_rate = Learning_rate,
                            objective ='reg:squarederror',          
                            booster ='gbtree',                      
                            reg_lambda = 0.5,                       
                            reg_alpha = 1e-5,                       
                            subsample = Subsample,
                            max_delta_step = Max_delta_step,
                            max_depth = Max_depth,
                            colsample_bytree = 0.8,                
                            colsample_bylevel = 1,                           
                            min_child_weight = 0.4,                  
                            n_jobs = -1,
                        )
                    
                    reg.fit(X_train, y_train)

                    R2_train = reg.score(X_train, y_train)
                    R2_test = reg.score(X_test, y_test)
                    R2_train_dict.update({f'N={N_estimators}_Dep={Max_depth}_L={Learning_rate}_S={Subsample}_Step={Max_delta_step}': R2_train})
                    R2_test_dict.update({f'N={N_estimators}_Dep={Max_depth}_L={Learning_rate}_S={Subsample}_Step={Max_delta_step}': R2_test})
                    
                    # Save Model
                    pickle.dump(reg, open('XGBoost_reg.dat', 'wb'))


sorted_train_items = sorted(R2_train_dict.items(), key=lambda item: item[1], reverse=True)
sorted_test_items = sorted(R2_test_dict.items(), key=lambda item: item[1], reverse=True)

sorted_R2_train_dict = dict(sorted_train_items) 
sorted_R2_test_dict = dict(sorted_test_items) 

with open('sorted_R2_train_dict','a+') as f:
    for key, value in sorted_R2_train_dict.items():
        f.write(f'{key}:{value}\n')
with open('sorted_R2_test_dict','a+') as f:
    for key, value in sorted_R2_test_dict.items():
        f.write(f'{key}:{value}\n')

# Feature Importance
importance = reg.get_booster().get_score(importance_type='gain')

for feature, score in importance.items():
    print(f'{feature}: {score}')



#------------------------------------2 atomic numbers + 3 PI features---------------------------------#

dataset = pd.read_csv('/home/zma/Multi-taskML/Merged2ID20SnapVolPeDF.csv', header = None)

cut_train = np.r_[     0:205828, 228698:423494, 445138:649170, 
                  671841:868112, 889920:1085347, 1107062:1311452, 
                  1334162:1534834, 1557131:1814981, 1843631:2107115
                  ]
cut_test = np.r_[        205828: 228698,423494: 445138,649170:671841, 
                         868112: 889920,1085347: 1107062,1311452:1334162, 
                          1534834: 1557131,1814981: 1843631,2107115:2136392
                 ]
selected_features = np.r_[0, 1, 22, 23, 24]

X_train = dataset.iloc[cut_train, selected_features].values
y_train = dataset.iloc[cut_train, 25].values

X_test = dataset.iloc[cut_test, selected_features].values
y_test = dataset.iloc[cut_test, 25].values


# Because of the particularity of dataset, GridSearch/CV is not applicable
R2_train_dict = dict()
R2_test_dict = dict()
for N_estimators in np.arange(600, 900, 100):
    for Max_depth in np.arange(7, 17, 2):
        for Learning_rate in np.arange(0.04, 0.12001, 0.002):
            for Subsample in np.arange(0.55, 0.9001, 0.05):
                for Max_delta_step in np.arange(0, 0.4001, 0.05):
          
                    reg= XGBR(n_estimators = N_estimators,
                            learning_rate = Learning_rate,
                            objective ='reg:squarederror',          
                            booster ='gbtree',                      
                            reg_lambda = 0.5,                       
                            reg_alpha = 1e-5,                       
                            subsample = Subsample,
                            max_delta_step = Max_delta_step,
                            max_depth = Max_depth,
                            colsample_bytree = 0.8,                
                            colsample_bylevel = 1,                           
                            min_child_weight = 0.4,                  
                            n_jobs = -1,
                        )
                    
                    reg.fit(X_train, y_train)

                    R2_train = reg.score(X_train, y_train)
                    R2_test = reg.score(X_test, y_test)
                    R2_train_dict.update({f'N={N_estimators}_Dep={Max_depth}_L={Learning_rate}_S={Subsample}_Step={Max_delta_step}': R2_train})
                    R2_test_dict.update({f'N={N_estimators}_Dep={Max_depth}_L={Learning_rate}_S={Subsample}_Step={Max_delta_step}': R2_test})

                    # Save Model, Call for Figure 5,6
                    pickle.dump(reg, open('XGBoost_reg.dat', 'wb'))

sorted_train_items = sorted(R2_train_dict.items(), key=lambda item: item[1], reverse=True)
sorted_test_items = sorted(R2_test_dict.items(), key=lambda item: item[1], reverse=True)

sorted_R2_train_dict = dict(sorted_train_items) 
sorted_R2_test_dict = dict(sorted_test_items) 

with open('sorted_R2_train_dict','a+') as f:
    for key, value in sorted_R2_train_dict.items():
        f.write(f'{key}:{value}\n')
with open('sorted_R2_test_dict','a+') as f:
    for key, value in sorted_R2_test_dict.items():
        f.write(f'{key}:{value}\n')

# Feature Importance
importance = reg.get_booster().get_score(importance_type='gain')

for feature, score in importance.items():
    print(f'{feature}: {score}')
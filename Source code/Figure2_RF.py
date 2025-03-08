#!/usr/local/lib64/python3
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn import metrics

#------------------------------------20 SNAP features---------------------------------#

dataset = pd.read_csv('/home/Multi-taskML/Merged2ID20SnapVolPeDF.csv', header = None)

cut_train = np.r_[     0:205828, 228698:423494, 445138:649170, 
                  671841:868112, 889920:1085347, 1107062:1311452, 
                  1334162:1534834, 1557131:1814981, 1843631:2107115
                  ]
cut_test = np.r_[        205828: 228698,423494: 445138,649170:671841, 
                         868112: 889920,1085347: 1107062,1311452:1334162, 
                          1534834: 1557131,1814981: 1843631,2107115:2136392
                 ]
selected_features = np.r_[2:22]

X_train = dataset.iloc[cut_train, selected_features].values
y_train = dataset.iloc[cut_train, 25].values

X_test = dataset.iloc[cut_test, selected_features].values
y_test = dataset.iloc[cut_test, 25].values


# Because of the particularity of dataset, GridSearch/CV is not applicable
R2_train_dict = dict()
R2_test_dict = dict()
for N_estimators in range(500,900,100):
	for Max_depth in range(5,15,2):
          
		reg= RF(n_estimators = N_estimators,
				max_depth = Max_depth,
				n_jobs = -1,
				)

		reg.fit(X_train, y_train)

		R2_train = reg.score(X_train, y_train)
		R2_test = reg.score(X_test, y_test)
		R2_train_dict.update({f'N={N_estimators}_Dep={Max_depth}': R2_train})
		R2_test_dict.update({f'N={N_estimators}_Dep={Max_depth}': R2_test})



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



#------------------------------------3 PI features---------------------------------#

dataset = pd.read_csv('/home/zma/Multi-taskML/Merged2ID20SnapVolPeDF.csv', header = None)

cut_train = np.r_[     0:205828, 228698:423494, 445138:649170, 
                  671841:868112, 889920:1085347, 1107062:1311452, 
                  1334162:1534834, 1557131:1814981, 1843631:2107115
                  ]
cut_test = np.r_[        205828: 228698,423494: 445138,649170:671841, 
                         868112: 889920,1085347: 1107062,1311452:1334162, 
                          1534834: 1557131,1814981: 1843631,2107115:2136392
                 ]
selected_features = np.r_[22, 23, 24]

X_train = dataset.iloc[cut_train, selected_features].values
y_train = dataset.iloc[cut_train, 25].values

X_test = dataset.iloc[cut_test, selected_features].values
y_test = dataset.iloc[cut_test, 25].values


# Because of the particularity of dataset, GridSearch/CV is not applicable
R2_train_dict = dict()
R2_test_dict = dict()
for N_estimators in range(500,900,100):
	for Max_depth in range(5,15,2):
          
		reg= RF(n_estimators = N_estimators,
				max_depth = Max_depth,
				n_jobs = -1,
				)

		reg.fit(X_train, y_train)

		R2_train = reg.score(X_train, y_train)
		R2_test = reg.score(X_test, y_test)
		R2_train_dict.update({f'N={N_estimators}_Dep={Max_depth}': R2_train})
		R2_test_dict.update({f'N={N_estimators}_Dep={Max_depth}': R2_test})

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

from xgboost import XGBRegressor as XGBR
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

# Import model
loaded_model = pickle.load(open('/home/Multi-taskML/PredictiveResults/XGBoost/XGBoost_reg.dat', 'rb'))
print('Loaded model from: XGBoost_reg.dat')
reg = loaded_model

tar_dataset = pd.read_csv('${Dataset}/${AlloyDataset}')

length = len(tar_dataset)
cut_test  = np.r_[0.9 * length : length]
selected_features = np.r_[1, 2, 23, 24, 25]

X_test = tar_dataset.iloc[ cut_test , selected_features].values
y_test = tar_dataset.iloc[ cut_test , 26].values

R2_test = reg.score(X_test, y_test)
y_test_pred = reg.predict(X_test)
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

# save dataset
submission = {"Calculated": y_test, "Predicted": y_test_pred}
submission = pd.DataFrame(submission)
submission.to_csv("${MLDir}/TrueAndPredictionResult.csv", float_format='%.6f', index=False)

with open("R2AndRMSE", 'a+') as f:
    f.write("R2AndRMSE\n")				
    f.write(f"R2_test= {R2_test}\n")
    f.write(f"RMSE_test= {RMSE_test}\n")
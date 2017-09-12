import os
import h5py
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import log_loss

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import pandas as pd
# ========================================================
# XGBoost algprithm
# ========================================================
from nn_methods import run_model, get_index, galnn


f = h5py.File("SFData.hdf5", "r")
Xtrain_d = f["X_train"][:]
ytrain = f["y_train"][:]
Xtest_d = f["X_test"][:]
dtest = f["X_test_ID"][:]
le_cat_class = f["le_cat_classes"][:]


# write results for submission
def write_result(y_t, le_cat_classes, dtest_d, filename):
    result = pd.DataFrame(y_t, index=dtest_d)
    result.index.name = 'Id'
    result.columns = le_cat_classes
    result.to_csv(filename + '.csv', float_format='%.5f')

# Final model for XgBoost, train on all data
param = {}
param['booster'] = 'gbtree'
param['objective'] = 'multi:softprob'
param['num_class'] = 39
param['eval_metric'] = 'logloss'
# param['scale_pos_weight'] = 1.0
param['bst:eta'] = 1
param['bst:max_depth'] = 6
# param['bst:colsample_bytree'] = 0.4
# param['gamma'] = 0.5
# param['min_child_weight'] = 5.
param['max_delta_step'] = 1
# param['silent'] = 1
param['early_stopping_rounds'] = 10
param['nthread'] = 30
num_round = 100
plst = list(param.items())
watchlist = []
    
dtrain_x = xgb.DMatrix(Xtrain_d, label=ytrain)
print "Fitting..."
bst = xgb.train(plst, dtrain_x, num_round, watchlist)
bst.save_model("final_xgboost_30_es.model")
curpred = bst.predict(Xtest_d)
write_result(curpred, 'submission_xgboost_30_es')


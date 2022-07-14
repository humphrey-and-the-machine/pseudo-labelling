import timeit

start = timeit.default_timer()

import time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn
patch_sklearn("knn_regressor")
patch_sklearn('random_forest_regressor')

from sklearn import neighbors, neural_network, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import dask
import dask.dataframe as dd

import warnings
warnings.filterwarnings('ignore')

# load the data
filepath = ''
df = dd.read_parquet(filepath+'COSMOS2015_with_pdz_all.parquet', engine='pyarrow')

#subset the data to drop objects at the faint end some objects without photo-z values
df = df[(df.Y_MAG_APER3 <= 24) &
    (df.Y_MAG_APER3 >0) &
    (df.J_MAG_APER3 <= 24) &
    (df.J_MAG_APER3 >0) &
    (df.H_MAG_APER3 <= 24) &
    (df.H_MAG_APER3 >0) &
    (df.PHOTOZ > 0) & (df.PHOTOZ < 9.8)]


# define training features and target(s)
bands_tr = ['u','B','V','r','ip','zp','Y','J','H','Ks']
cols_tr = []
for b in bands_tr: cols_tr += [x for x in df.columns.values.tolist() if b+'_MAG_APER3' in x]

# make colours
colours_tr = []
for i,b1 in enumerate(cols_tr[:-2]):
    for b2 in cols_tr[i+1:]:
        df[b1+'-'+b2] = df[b1]-df[b2]
        colours_tr += [b1+'-'+b2]

cols_tr += colours_tr

cols_use = ['NUMBER','PHOTOZ','SSFR_BEST','MASS_BEST'] + cols_tr
df = df[cols_use].compute()

# set target
target = ['MASS_BEST']

# split into train set, test set for Pseudo-L, and test set for final evaluation
# pseudo labeling is done predicting on the train set
test_size = 0.2 #holdout set
test_size2 = 0.7 #train and test set for pseudo labeling

n_rs = np.random.choice(range(10000),1, replace = False)
X_train, X_val, y_train, y_val = train_test_split(df[cols_tr], df[target],test_size=test_size,random_state=n_rs[0])
X_train, X_test, y_train, y_test = train_test_split(X_train[cols_tr], y_train[target],test_size=test_size2,random_state=n_rs[0])

print('Training set size:',len(X_train),'galaxies')
print('Pseudo-labeling test set size:',len(X_test),'galaxies')
print('Holdout test set size:',len(X_val),'galaxies')

# apply standard scaler, preserving -99,9 missing value flag (see Humphrey et al. 2021)
def scale_data(data,scaler):
    array_tr_sc = scaler.transform(data[cols_tr])
    df_sc = pd.DataFrame(array_tr_sc, index=data.index, columns=cols_tr)

    for col in cols_tr:
        df_sc[col][df_sc[col].isna() == True] = np.max(df_sc[col])

    return df_sc

X_train[X_train[cols_tr] <=-99] = np.nan
X_test[X_test[cols_tr] <=-99] = np.nan
X_val[X_val[cols_tr] <=-99] = np.nan

scaler = StandardScaler()
scaler.fit(X_train[cols_tr])

X_train = scale_data(X_train,scaler)
X_test = scale_data(X_test,scaler)
X_val = scale_data(X_val,scaler)

#tree_method = 'hist'
tree_method = 'gpu_hist'

n_jobs=-1

models = {
        'catboost_1000_11': CatBoostRegressor(logging_level='Silent',n_estimators=1000,max_depth=11,thread_count=-1,task_type="GPU"),
        'knn_7': neighbors.KNeighborsRegressor(n_neighbors=7,n_jobs=n_jobs),
        'lgbm_2000_12': lgb.LGBMRegressor(n_jobs = n_jobs,n_estimators=2000,max_depth = 12, subsample = 0.8,colsample_bytree = 0.9),
        'xgb_200_9': xgb.XGBRegressor(n_estimators=200, max_depth=9,tree_method=tree_method,
                                nthread=n_jobs,subsample=0.8,colsample_bytree=0.8),
        'rf_500_10': RandomForestRegressor(n_estimators=500,max_depth=10,n_jobs=n_jobs),
}

final_dict = {}

for model_name in list(models.keys()):

    clf = models[model_name]

    clf.fit(X_train,y_train)

    preds = clf.predict(X_test).flatten()
    preds_val = clf.predict(X_val).flatten()

    # n rounds of pseudo-labeling: append test data and test set predictions to the training data
    n_iter = 51

    X_train_ps = X_train.append(X_test)

    preds_list = [preds]
    preds_val_list = [preds_val]

    for n in range(n_iter):

        t0=time.time()

        y_train_ps = y_train.append(pd.DataFrame(preds_list[n],columns=target,index=X_test.index))

        clf.fit(X_train_ps,y_train_ps)

        preds_list.append(clf.predict(X_test).flatten())
        preds_val_list.append(clf.predict(X_val).flatten())

        duration = time.time()-t0
        print(model_name,' it: '+str(n),duration/60.)

    mean_ae_test_list = []
    median_ae_test_list = []
    r2_test_list = []

    mean_ae_val_list = []
    median_ae_val_list = []
    r2_val_list = []

    for i in range(n_iter):

        mean_ae_test_list.append(mean_absolute_error(y_test,preds_list[i]))
        mean_ae_val_list.append(mean_absolute_error(y_val,preds_val_list[i]))

        median_ae_test_list.append(median_absolute_error(y_test,preds_list[i]))
        median_ae_val_list.append(median_absolute_error(y_val,preds_val_list[i]))

        r2_test_list.append(r2_score(y_test,preds_list[i]))
        r2_val_list.append(r2_score(y_val,preds_val_list[i]))


    # store results in a dict
    results_dict = {
                    'target': target[0],
                    'mean_ae_test_list': mean_ae_test_list,
                    'mean_ae_val_list':  mean_ae_val_list,
                    'median_ae_test_list': median_ae_test_list,
                    'median_ae_val_list':  median_ae_val_list,
                    'r2_test_list': r2_test_list,
                    'r2_val_list':  r2_val_list
                   }

    final_dict[model_name] = results_dict


with open('save_data_nit'+str(n_iter)+'_rs'+str(n_rs)+'_'+str(target[0])+'.pkl', 'wb') as f:
    pickle.dump(final_dict, f)


stop = timeit.default_timer()
execution_time = (stop - start)/60

print("Program Executed in "+str(execution_time)) # It returns time in minutes

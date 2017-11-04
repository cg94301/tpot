import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from scipy.stats import randint, uniform

import time, os, fnmatch, shutil
import timeit

import simplerandom.iterators as sri
from datetime import datetime

markets = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
           'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC','F_FV', 'F_GC',
           'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
           'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
           'F_S','F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US','F_W', 'F_XX',
           'F_YM']

markets = ['F_AD']

def automl(market):
    dt = datetime.now()
    #rng = sri.KISS(123958, 34987243, 3495825239, 2398172431)
    rng = sri.KISS(dt.microsecond)

    csv = pd.read_csv('data/' + market + '.csv')
    print csv.shape
    print csv.columns

    # Dropping rows with label 0
    # Doing binary logistic regression here
    # in F_AD there are only 181 0 labels
    # but there could be more.
    csv = csv[(csv.y != 0)]

    y = csv['y']
    dates = csv['date']
    # lookback 255. 254 to 0. Exclude 0.
    X = csv.loc[:,'254':'1'] # Accuracy: 0.5172
    
    
    #seed = 342
    #seed = 3165278097
    seed = next(rng)
    print "seed:",seed
    
    cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=seed)
    

    # time: 6252.53945184 
    tpot_config = { 'xgboost.sklearn.XGBClassifier': {
        #'gamma': [0,0.5,1.0],
        #'subsample': [0.4,0.6,0.8,1.0],
        #'colsample_bylevel': [0.5,0.75,1.0],
        'max_depth': [1,2,3],
        'learning_rate': [1,0.1,0.01],
        'silent': [1.0],
        'nthread': [-1],
        'n_estimators': [50,75,100,125,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]}
    }

    # default: gen=5, pop=20
    pipeline_optimizer = TPOTClassifier(generations=10, population_size=100, cv=cv, n_jobs=-1,
                                        random_state=seed, verbosity=3, periodic_checkpoint_folder='checkpoints',config_dict=tpot_config)
    
    
    
    start_time = timeit.default_timer()
    pipeline_optimizer.fit(X, y)
    elapsed = timeit.default_timer() - start_time
    print "time:", elapsed
    
    # pseudo test
    X_test = csv.loc[:9,'254':'1'] # Accuracy: 0.5172
    print X_test.shape
    
    # Performance on test set. Might (probably will) differ from best pipeline score.
    print(pipeline_optimizer.predict(X_test))
    print(pipeline_optimizer.predict_proba(X_test))
    
    # Write out best pipeline
    t = time.localtime()
    timestamp = time.strftime('%m%d%Y_%H%M%S', t)
    pipeline_optimizer.export('export/tpot_exported_pipeline_' + market + '_' + timestamp + '_' + str(seed) + '.py')
    
    #ei = pipeline_optimizer.evaluated_individuals_
    einame = 'clfs/pipe_' + market + '_' + timestamp + '_' + str(seed) + '.pkl'
    joblib.dump(pipeline_optimizer.fitted_pipeline_, einame)    
    tmp = joblib.load(einame)
    #print "read back from joblib:", tmp
    print(tmp.predict_proba(X_test))


if __name__ == '__main__':

    for market in markets:
        automl(market)

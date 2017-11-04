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

import simplerandom.iterators as sri
from datetime import datetime

dt = datetime.now()

#rng = sri.KISS(123958, 34987243, 3495825239, 2398172431)
rng = sri.KISS(dt.microsecond)

csv = pd.read_csv('data/F_SB.csv')
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

#params_grid = {
#    'max_depth': [1, 2, 3, 4],
#    'n_estimators': [125, 130, 135, 140],
#    #'learning_rate': np.linspace(1e-16, 1, 3)
#    'learning_rate': [1, 0.1, 0.01, 0.001]
#}
#
#params_fixed = {
#    'objective': 'binary:logistic',
#    'silent': 1
#}
#

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=seed)

tpot_config = { 'xgboost.sklearn.XGBClassifier': {
    #'gamma': [0,0.5,1.0],
    #'subsample': [0.4,0.6,0.8,1.0],
    #'colsample_bylevel': [0.5,0.75,1.0],
    'max_depth': [1,2,3],
    'learning_rate': [1.0,0.1],
    'silent': [1.0],
    'nthread': [-1],
    'n_estimators': range(20,100,50)}
}

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=10, n_jobs=-1,
                                    random_state=seed, verbosity=3, periodic_checkpoint_folder='checkpoints',config_dict=tpot_config)


t = time.localtime()
timestamp = time.strftime('%m%d%Y_%H%M%S', t)

pipeline_optimizer.fit(X_train, y_train)

# Performance on test set. Might (probably will) differ from best pipeline score.
print(pipeline_optimizer.score(X_test, y_test))

# Write out best pipeline
pipeline_optimizer.export('tpot_exported_pipeline_' + timestamp + '_' + str(seed) + '.py')

ei = pipeline_optimizer.evaluated_individuals_

einame = 'evaluated_individuals_' + timestamp + '_' + str(seed) + '.pkl'
joblib.dump(ei, einame)    
tmp = joblib.load(einame)
#print "read back from joblib:", tmp

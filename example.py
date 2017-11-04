from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

seed = 342

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=seed)

tpot_config = { 'xgboost.sklearn.XGBClassifier': {
    'max_depth': [2,3,4],
    'learning_rate': [1.0],
    'silent': [1.0],
    'n_estimators': [5,10,15]}
}

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=10,
                                    random_state=seed, verbosity=3, periodic_checkpoint_folder='checkpoints',config_dict=tpot_config)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

ei = pipeline_optimizer.evaluated_individuals_

joblib.dump(ei, 'evaluated_individuals.pkl')    
tmp = joblib.load('evaluated_individuals.pkl')
print "read back from joblib:", tmp

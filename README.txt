
1       0.534077849842  XGBClassifier(CombineDFs(input_matrix, input_matrix), XGBClassifier__learning_rate=0.1, XGBClassifier__max_depth=2, XGBClassifier__n_estimators=70, XGBClassifier__nthread=-1, XGBClassifier__silent=1.0)


Gamma not help:
1       0.517211151496  XGBClassifier(input_matrix, XGBClassifier__gamma=1.0, XGBClassifier__learning_rate=0.01, XGBClassifier__max_depth=2, XGBClassifier__n_estimators=570, XGBClassifier__nthread=-1, XGBClassifier__silent=1.0)


Subsample no help:
1       0.53791944186   XGBClassifier(input_matrix, XGBClassifier__learning_rate=0.01, XGBClassifier__max_depth=1, XGBClassifier__n_estimators=420, XGBClassifier__nthread=-1, XGBClassifier__silent=1.0, XGBClassifier__subsample=1.0)

Colsample by tree:
1       0.52403458896   XGBClassifier(CombineDFs(input_matrix, input_matrix), XGBClassifier__colsample_bytree=0.5, XGBClassifier__learning_rate=0.1, XGBClassifier__max_depth=2, XGBClassifier__n_estimators=20, XGBClassifier__nthread=-1, XGBClassifier__silent=1.0)

Colsample by level:
1       0.527875276612  XGBClassifier(input_matrix, XGBClassifier__colsample_bylevel=1.0, XGBClassifier__learning_rate=1.0, XGBClassifier__max_depth=2, XGBClassifier__n_estimators=470, XGBClassifier__nthread=-1, XGBClassifier__silent=1.0)

Try wide range of standard params, depth, number of trees, learning rate.

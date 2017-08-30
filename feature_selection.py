import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

x_train = pd.read_csv('./input/tf_train.csv')
y_train = x_train['Survived'].copy()
del x_train['Survived']

clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(x_train, y_train)

features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = clf.feature_importances_
print features.sort_values(['importance'], ascending=False)

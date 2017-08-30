import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from submit import submit

import tf_algo as tf

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

combined = train.drop('Survived', axis=1).append(test)

# Sex

combined['Sex'] = combined['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# Name

def Name_Title_Code(x):
    if x == 'Mr.':
        return 1
    if (x == 'Mrs.') or (x == 'Ms.') or (x == 'Lady.') or (x == 'Mlle.') or (x == 'Mme'):
        return 2
    if x == 'Miss.':
        return 3
    if x == 'Rev.':
        return 4
    return 5


combined['Name_Title'] = combined['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
combined['Name_Title'] = combined['Name_Title'].apply(Name_Title_Code)
del combined['Name']

# Age

combined['Age_Null_Flag'] = combined['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
data = combined[:train.__len__()].groupby(['Name_Title', 'Pclass'])['Age']
combined['Age'] = data.transform(lambda x: x.fillna(x.mean()))

# SibSp, Parch

combined['Fam_Size'] = np.where((combined['SibSp'] + combined['Parch']) == 0, 'Singleton',
                                np.where((combined['SibSp'] + combined['Parch']) <= 3, 'Small', 'Big'))
del combined['SibSp']
del combined['Parch']

# Ticket

combined['Ticket_Lett'] = combined['Ticket'].apply(lambda x: str(x)[0])
combined['Ticket_Lett'] = combined['Ticket_Lett'].apply(lambda x: str(x))
combined['Ticket_Lett'] = np.where((combined['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']),
                                   combined['Ticket_Lett'],
                                   np.where((combined['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
combined['Ticket_Len'] = combined['Ticket'].apply(lambda x: len(x))
del combined['Ticket']

# Cabin
# TODO

combined['Cabin_Letter'] = combined['Cabin'].apply(lambda x: str(x)[0])
combined['Cabin_num1'] = combined['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
combined['Cabin_num1'].replace('an', np.NaN, inplace=True)
combined['Cabin_num1'] = combined['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
combined['Cabin_num'] = pd.qcut(combined['Cabin_num1'][:train.__len__()], 3)
combined = pd.concat((combined, pd.get_dummies(combined['Cabin_num'], prefix='Cabin_num')), axis=1)

del combined['Cabin']
del combined['Cabin_num']
del combined['Cabin_num1']

# Embarked

combined['Embarked'].fillna('S', inplace=True)

# Fare
# TODO

combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)

# dummies

columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']
for column in columns:
    combined[column] = combined[column].apply(lambda x: str(x))
    good_cols = [column + '_' + i for i in combined[:train.__len__()][column].unique() if
                 i in combined[train.__len__():][column].unique()]
    combined = pd.concat((combined, pd.get_dummies(combined[column], prefix=column)[good_cols]), axis=1)
    del combined[column]

# feature selection

del combined['PassengerId']
targets = train.Survived
train = combined[:train.__len__()]
test = combined[train.__len__():]

clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
# print features.sort_values(['importance'], ascending=False)

model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
test_new = model.transform(test)

# tf_train = pd.DataFrame(train_new)
# tf_train['Survived'] = targets
# tf_test = pd.DataFrame(test_new)
# tf_train.to_csv('./input/tf_train.csv', index=False)
# tf_test.to_csv('./input/tf_test.csv', index=False)
# tf.do(10)

# forest = RandomForestClassifier(max_features='sqrt')
#
# parameter_grid = {
#     'max_depth': [4, 5, 6, 7, 8],
#     'n_estimators': [200, 210, 240, 250],
#     'criterion': ['gini', 'entropy']
# }
# cross_validation = StratifiedKFold(targets, n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(train_new, targets)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

x_train, x_test, y_train, y_test = train_test_split(train_new, targets, test_size=0.20, random_state=513)
rf = RandomForestClassifier(criterion='gini',
                            n_estimators=700,
                            min_samples_split=16,
                            min_samples_leaf=1,
                            max_features='auto',
                            oob_score=True,
                            random_state=13,
                            n_jobs=-1)

rf.fit(x_train, y_train)
print rf.score(x_train, y_train)
print rf.score(x_test, y_test)
print '-' * 30
print(rf.oob_score_)


# xgb = GradientBoostingClassifier(n_estimators=700,
#                                  min_samples_split=128,
#                                  min_samples_leaf=4,
#                                  max_features='auto',
#                                  random_state=13,
#                                  subsample=0.8)
# xgb.fit(x_train, y_train)
# print xgb.score(x_train, y_train)
# print xgb.score(x_test, y_test)

submit(rf.predict(test_new).astype(int))

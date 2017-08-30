from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


# Random Forests
def RF(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    prediction = random_forest.predict(X_test)
    print random_forest.score(x_test, y_test)
    print cross_val_score(random_forest, x_test, y_test, cv=5).mean()
    return prediction


# Gradient Boosting
def GSTDT(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    nEst = 100
    depth = 3
    learnRate = 0.1
    gradient_boosting = GradientBoostingClassifier(n_estimators=nEst, max_depth=depth, learning_rate=learnRate,
                                                   subsample=0.5)
    gradient_boosting.fit(x_train, y_train)
    prediction = gradient_boosting.predict(X_test)
    print gradient_boosting.score(x_test, y_test)
    print cross_val_score(gradient_boosting, x_test, y_test, cv=5).mean()
    return prediction


# xgboost
def XGB(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    xgb = XGBClassifier(max_depth=5, subsample=0.6)
    xgb.fit(x_train, y_train)
    prediction = xgb.predict(X_test)
    print xgb.score(x_test, y_test)
    print cross_val_score(xgb, x_test, y_test, cv=5).mean()
    return prediction


# SVM
def SVM(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    svm = SVC()
    svm.fit(x_train, y_train)
    prediction = svm.predict(X_test)
    print svm.score(x_test, y_test)
    print cross_val_score(svm, x_test, y_test, cv=5).mean()
    return prediction


# Logistic Regression
def LR(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    p = lr.predict(X_test)
    print 'LR:'
    print lr.score(x_test, y_test)
    print cross_val_score(lr, x_test, y_test, cv=5).mean()
    return p


# Neural Network
def ANN(x_train, y_train, X_test):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
    return None

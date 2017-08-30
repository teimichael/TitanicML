import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.data_utils import load_csv
from tflearn.data_preprocessing import DataPreprocessing


def prehandle():
    titanic_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Embarked
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
    test_df['Embarked'] = test_df['Embarked'].fillna('S')
    embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])
    embark_dummies_test = pd.get_dummies(test_df['Embarked'])
    titanic_df = titanic_df.join(embark_dummies_titanic)
    test_df = test_df.join(embark_dummies_test)
    titanic_df.drop(['Embarked'], axis=1, inplace=True)
    test_df.drop(['Embarked'], axis=1, inplace=True)

    # Fare
    test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

    # Age
    average_age_titanic = titanic_df["Age"].mean()
    std_age_titanic = titanic_df["Age"].std()
    count_nan_age_titanic = titanic_df["Age"].isnull().sum()
    average_age_test = test_df["Age"].mean()
    std_age_test = test_df["Age"].std()
    count_nan_age_test = test_df["Age"].isnull().sum()
    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                               size=count_nan_age_titanic)
    rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test,
                               size=count_nan_age_test)
    titanic_df['Age'][np.isnan(titanic_df["Age"])] = rand_1
    test_df["Age"][np.isnan(test_df["Age"])] = rand_2
    titanic_df['Age'] = titanic_df['Age'].astype(int)
    test_df['Age'] = test_df['Age'].astype(int)

    # Sex
    person_dummies_titanic = pd.get_dummies(titanic_df['Sex'])
    person_dummies_test = pd.get_dummies(test_df['Sex'])
    titanic_df = titanic_df.join(person_dummies_titanic)
    test_df = test_df.join(person_dummies_test)
    titanic_df.drop(['Sex'], axis=1, inplace=True)
    test_df.drop(['Sex'], axis=1, inplace=True)

    # Pclass
    pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
    pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
    pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
    pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
    titanic_df = titanic_df.join(pclass_dummies_titanic)
    test_df = test_df.join(pclass_dummies_test)
    titanic_df.drop(['Pclass'], axis=1, inplace=True)
    test_df.drop(['Pclass'], axis=1, inplace=True)

    # print titanic_df[titanic_df.isnull().values == True]
    # print test_df[test_df.isnull().values == True]
    # print titanic_df.head()
    # print test_df.tail()

    titanic_df.to_csv('./input/tf_train.csv', index=False)
    test_df.to_csv('./input/tf_test.csv', index=False)


def train(data, label):
    data_prep = DataPreprocessing()
    data_prep.add_featurewise_zero_center()
    data_prep.add_featurewise_stdnorm()
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.30, random_state=33)

    # Build neural network
    net = tflearn.input_data(shape=[None, len(data[0])], data_preprocessing=data_prep)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32, activation='relu')
    # net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam')

    # Define model
    model = tflearn.DNN(net)

    # Start training (apply gradient descent algorithm)
    model.fit(x_train, y_train, n_epoch=300, batch_size=3, show_metric=True)
    print model.evaluate(x_test, y_test)
    p = model.predict_label(x_test)
    count = 0
    for i in range(0, len(p)):
        if int(y_test[i][1]) == p[i][0]:
            count += 1
    print float(count) / len(p)
    return model


def do(target_column):
    # prehandle()
    data, label = load_csv('./input/tf_train.csv', target_column=target_column,
                           categorical_labels=True, n_classes=2)
    data = np.array(data, dtype=np.float32)

    test = pd.read_csv('./input/tf_test.csv')
    # test.drop(['PassengerId'], axis=1, inplace=True)
    model = train(data, label)
    p = model.predict_label(np.array(test.values, dtype=np.float32))
    prediction = []
    for i in range(0, len(p)):
        prediction.append(p[i][0])
    return prediction

import pandas as pd

import featurehandler as fh
import algorithm as al
import tf_algo as tf
from submit import submit

# # get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
#
# titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
# test_df = test_df.drop(['Name', 'Ticket'], axis=1)
#
# titanic_df, test_df = fh.handle(titanic_df, test_df)
#
# # define training and testing sets
# x_train = titanic_df.drop("Survived", axis=1)
# y_train = titanic_df["Survived"]
# x_test = test_df.drop("PassengerId", axis=1).copy()

x_train = pd.read_csv('./input/x_train.csv')
y_train = titanic_df['Survived']
x_test = pd.read_csv('./input/x_test.csv')

# p = al.XGB(x_train, y_train, x_test)
p = tf.do(0)

submit(p)

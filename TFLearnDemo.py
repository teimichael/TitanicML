import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
from sklearn.model_selection import train_test_split, cross_val_score

# titanic.download_dataset('./titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv

data, labels = load_csv('./titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
def preprocess(passengers, columns_to_delete):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [passenger.pop(column_to_delete) for passenger in passengers]
    for i in range(len(passengers)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        passengers[i][1] = 1. if passengers[i][1] == 'female' else 0.
    return np.array(passengers, dtype=np.float32)


# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore = [1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, random_state=33)
data = x_train
labels = y_train

# Build neural network
net = tflearn.input_data(shape=[None, 6])
# net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32,activation='relu')
# net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam')

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=300, batch_size=3, show_metric=True)
p = model.predict(x_test)
count = 0
for i in range(0, len(p)):
    # print int(y_test[i][1])
    # print 0 if p[i][1] < 0.5 else 1
    # print '-' * 30
    if int(y_test[i][1]) == (0 if p[i][1] < 0.5 else 1):
        count += 1
print model.evaluate(x_test, y_test)
print float(count) / len(p)

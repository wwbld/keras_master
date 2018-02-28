import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import util

FILE = '../data/sample_1.csv'
SAVE = '../../model/model_1'

training_data, policy_training_target, value_training_target, \
testing_data, policy_testing_target, value_testing_target = util.read_csv(FILE)

model = load_model(SAVE)
for i in range(10):
    predictions = model.predict(np.array([testing_data[i]]))
    print("cycle{0}".format(i))

    print("policy is {0}, value is {1}".format(util.get_policy(predictions)[0],
                                               util.get_value(predictions)[1]))

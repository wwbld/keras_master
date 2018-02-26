import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import util

FILE = '../data/sample.csv'

batch_size = 128
policy_num_classes = 64
# due to the mistake of the original file, it should be 2
value_num_classes = 3
epochs = 200

training_data, policy_training_target, value_training_target, \
testing_data, policy_testing_target, value_testing_target = util.read_csv(FILE)

policy_training_target = keras.utils.to_categorical(policy_training_target, policy_num_classes)
value_training_target = keras.utils.to_categorical(value_training_target, value_num_classes)

policy_testing_target = keras.utils.to_categorical(policy_testing_target, policy_num_classes)
value_testing_target = keras.utils.to_categorical(value_testing_target, value_num_classes)

input = Input(shape=(144,))
reshape = Reshape((12,12,1))(input)

x = Conv2D(32, kernel_size=(3,3))(reshape)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(128, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# policy branch
policy_output = Conv2D(4, kernel_size=(3,3))(x)
policy_output = BatchNormalization()(policy_output)
policy_output = LeakyReLU()(policy_output)

policy_output = Flatten()(policy_output)
policy_output = Dense(128)(policy_output)
policy_output = LeakyReLU()(policy_output)

policy_output = Dense(policy_num_classes)(policy_output)
policy_output = Activation('softmax')(policy_output)

# value branch
value_output = Conv2D(1, kernel_size=(3,3))(x)
value_output = BatchNormalization()(value_output)
value_output = LeakyReLU()(value_output)

value_output = Flatten()(value_output)
value_output = Dense(20)(value_output)
value_output = LeakyReLU()(value_output)

value_output = Dense(value_num_classes)(value_output)
value_output = Activation('softmax')(value_output)

model = Model(inputs=input, outputs=[policy_output, value_output])

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Nadam(),
              metrics=['accuracy'])

history = model.fit(training_data, [policy_training_target, value_training_target],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(testing_data, [policy_testing_target, value_testing_target]))
score = model.evaluate(testing_data, [policy_testing_target, value_testing_target], verbose=0)
print('final result: ', score)


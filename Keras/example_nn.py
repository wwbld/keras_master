import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import backend as K
from keras import optimizers
from residual_block import residual_block
import matplotlib.pyplot as plt
import util

FILE = '../data/sample.csv'
SAVE = '../model/model_1'

batch_size = 256
policy_num_classes = 64
value_num_classes = 2
epochs = 0

training_data, policy_training_target, value_training_target, \
testing_data, policy_testing_target, value_testing_target = util.read_csv(FILE)

policy_training_target = keras.utils.to_categorical(policy_training_target, policy_num_classes)
value_training_target = keras.utils.to_categorical(value_training_target, value_num_classes)

policy_testing_target = keras.utils.to_categorical(policy_testing_target, policy_num_classes)
value_testing_target = keras.utils.to_categorical(value_testing_target, value_num_classes)

input = Input(shape=(144,))
reshape = Reshape((12,12,1))(input)

# conv1_1 --> 12x12x64
x = Conv2D(64, kernel_size=(3,3), padding='same')(reshape)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# conv1_2
x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# pool1 --> 6x6
x = MaxPooling2D()(x)

# conv2_1 --> 6x6x128
x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# conv2_2
x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# pool2 --> 3x3
x = MaxPooling2D()(x)

# conv3_1 --> 3x3x256
x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# conv3_2
x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# conv3_3
x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# policy branch
policy_output = Conv2D(2, kernel_size=(3,3), padding='same')(x)
policy_output = BatchNormalization()(policy_output)
policy_output = Activation('relu')(policy_output)

policy_output = Flatten()(policy_output)
policy_output = Dense(256)(policy_output)
policy_output = BatchNormalization()(policy_output)
policy_output = Activation('relu')(policy_output)

policy_output = Dense(policy_num_classes)(policy_output)
policy_output = BatchNormalization()(policy_output)
policy_output = Activation('softmax')(policy_output)

# value branch
value_output = Conv2D(1, kernel_size=(3,3), padding='same')(x)
value_output = BatchNormalization()(value_output)
value_output = Activation('relu')(value_output)

value_output = Flatten()(value_output)
value_output = Dense(128)(value_output)
value_output = BatchNormalization()(value_output)
value_output = Activation('relu')(value_output)

value_output = Dense(value_num_classes)(value_output)
value_output = BatchNormalization()(value_output)
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

model.save(SAVE)

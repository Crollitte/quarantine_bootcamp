import numpy
import scipy
import datetime
from tensorflow.keras.models import Model
from keras.layers import concatenate as merge_l

from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras import backend as K
import tensorflow as tf

CHECKPOINT_PATH = "training_1/cp.ckpt"
LOGS_DIR = "logs/fit/"
NUM_EPOCHS = 10

model = None

def continue_trainig(train_images, train_labels, test_images, test_labels):
  model = reload_model()
  return train(train_images, train_labels, test_images, test_labels, model)

def train(train_images, train_labels, test_images, test_labels, model=model):
  if model is None: model = get_unet()

  # Create a callback that saves the model's weights
  cp_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

  log_dir = LOGS_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Train the model with the new callback
  model.fit(train_images, 
            train_labels,  
            epochs=NUM_EPOCHS,
            validation_data=(test_images,test_labels),
            callbacks=[cp_callback, tensorboard_callback])  # Pass callbacks to training (checkpoint and tensorboard)

# Evaluate the model
def test(test_images,  test_labels, model=model):
  loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
  print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

def reload_model(model=model):
  model = get_unet()
  # Loads the weights
  model.load_weights(CHECKPOINT_PATH)
  return model

def predict(test_images, model=model):
  predictions = model.predict(test_images)
  return predictions

def print_model_summary(model=model):
  model.summary()

def get_unet():
    print('Building Network')
    conv_params = dict(activation='relu', border_mode='same')
    #merge_params = dict(axis=1)
    inputs = Input((3, 406, 438))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first",padding='same')(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),  data_format="channels_first",padding='same')(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first",padding='same')(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),  data_format="channels_first",padding='same')(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3],axis=3)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2],axis=3)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1],axis=3)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
              #    metrics=['accuracy', jaccard_coef, jaccard_coef_int])
              # pode ser que seja legal usar essas métricas definidas pela competição
    return model

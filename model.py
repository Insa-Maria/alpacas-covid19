
# $ pip install -U git+https://github.com/qubvel/efficientnet

import keras
import numpy as np
import efficientnet.keras as efn
import argparse
from PIL import Image
import streamlit as st
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def stage1_model_factory():
    '''
    basic binary classification model with pretrained efficientnet
    '''

    # transfer learning
    efficientnet = efn.EfficientNetB0(
        input_shape=(img_h, img_w, 3), include_top=False, weights="imagenet",
    )
    inp = keras.layers.Input(shape=(img_h, img_w, 3))
    x = efficientnet(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1000, activation="relu")(x)
    x = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model

def stage2_model_factory():
    '''
    severity classification model with multi-inputs (imgs + symptoms)
    '''

    img_in = keras.layers.Input(shape=(img_h, img_w, 3))
    symptom_in = keras.layers.Input(shape=(7,))

    symptom_net = keras.layers.Dense(5, activation="relu")(symptom_in)
    symptom_net = keras.layers.Dense(5, activation="relu")(symptom_net)

    def conv_bn_relu(filters):
        def ret(x):
            x = keras.layers.Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            return x
        return ret


    # transfer learning
    img_in = keras.layers.Input(shape=(img_h, img_w, 3))

    c = conv_bn_relu(8)(img_in)
    c = conv_bn_relu(8)(c)
    c = keras.layers.MaxPooling2D(strides=(2,2))(c)
    c = conv_bn_relu(16)(c)
    c = conv_bn_relu(16)(c)
    c = keras.layers.MaxPooling2D(strides=(2,2))(c)
    c = conv_bn_relu(32)(c)
    c = conv_bn_relu(32)(c)
    c = keras.layers.MaxPooling2D(strides=(2,2))(c)
    c = conv_bn_relu(64)(c)
    c = conv_bn_relu(64)(c)
    c = keras.layers.GlobalAveragePooling2D()(c)

    x = keras.layers.Concatenate()([c, symptom_net])
    x = keras.layers.Dense(10, activation="relu")(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)

    model2 = keras.models.Model(inputs=[img_in, symptom_in], outputs=x)
    return model2

# use this to get the props to a prediction
def probs_to_onehot(pred):
    max_ix = np.argmax(pred, axis=1)
    one_hot = np.zeros_like(pred)
    one_hot[np.arange(pred.shape[0]), max_ix] = 1
    return one_hot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filepath',action='store',dest='filepath')
    results = parser.parse_args()

    path = '/Users/markwaters/Documents/covid-19/covid-19-pneumonia-mild.JPG'

    # Model params
    img_h = 256
    img_w = 256

    target_size = [256,256,3]    

    # pil resize is w h
    img = Image.open(path)
    img = img.resize((target_size[1], target_size[0]), resample=1)
    img = np.asarray(img)

    if len(img.shape) == 3 and target_size[-1] == 1:
        img = Image.open(path).convert("L")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)
        img = np.expand_dims(img, -1)
    elif len(img.shape) == 2 and target_size[-1] == 3:
        img = Image.open(path).convert("RGB")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)

    image = img / 255.0
    # remove alpha channel
    image = image[:, :, 0:3]
    image = np.expand_dims(image, 0)

    model = stage1_model_factory()
 
    model.load_weights('classifier_Fold0.hd5')

    model2 = stage2_model_factory()
    model2.load_weights('classifier_meta_Fold0.hd5') # switch here

    # image needs to be (batch, h, w, c) (c = 3 channels)

    p = model.predict(image)
    result = probs_to_onehot(p)

    symptom_array = np.array([0,0,0,0,0,0,0])
    symptom_array= np.expand_dims(symptom_array, 0)

    p2 = model2.predict([image, symptom_array])
    print('P: {}'.format(p))
    print(result)
    p2 = round((p2[0][0])*10)
    print('P2: {}'.format(p2))





    




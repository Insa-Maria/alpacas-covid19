import keras
import numpy as np
import efficientnet.keras as efn
import argparse
from PIL import Image
import streamlit as st
import os

# Fix the mac os error
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def probs_to_onehot(pred):
    max_ix = np.argmax(pred, axis=1)
    one_hot = np.zeros_like(pred)
    one_hot[np.arange(pred.shape[0]), max_ix] = 1
    return one_hot

def get_image(img):
    # pil resize is w h
    img = img.resize((target_size[1], target_size[0]), resample=1)
    img = np.asarray(img)

    if len(img.shape) == 3 and target_size[-1] == 1:
        img = Image.open(uploaded_image).convert("L")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)
        img = np.expand_dims(img, -1)
    elif len(img.shape) == 2 and target_size[-1] == 3:
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)

    image = img / 255.0
    # remove alpha channel
    image = image[:, :, 0:3]
    image = np.expand_dims(image, 0)

    return image

if __name__ == '__main__':
    # Model params
    img_h = 256
    img_w = 256

    target_size = [256,256,3]    

    model = stage1_model_factory()
 
    model.load_weights('classifier_Fold0.hd5')

    uploaded_image = st.sidebar.file_uploader("Upload an X-Ray Image", type="jpg")
    if(uploaded_image):
        p_image = Image.open(uploaded_image)
        st.image(p_image, caption='This', width=300)
        image = get_image(p_image)
        #p = model.predict(image)





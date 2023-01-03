
# $ pip install -U git+https://github.com/qubvel/efficientnet

import keras
import numpy as np
import efficientnet.keras as efn
import argparse
from PIL import Image
import streamlit as st
import time

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

"""
This project demonstrates the IEEE & Kaggle dataset and Alpacas-Covid-19 model into an interactive app.
It supports physicians and clinicals on the detection and diagnosis of COVID-19

👈 Please, add the required **_Patient Details_** and select **_Run_** in the sidebar to start.
"""


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

    model = keras.models.Model(inputs=[img_in, symptom_in], outputs=x)
    return model

# use this to get the props to a prediction
def probs_to_onehot(pred):
    max_ix = np.argmax(pred, axis=1)
    one_hot = np.zeros_like(pred)
    one_hot[np.arange(pred.shape[0]), max_ix] = 1
    return one_hot

def resize_image(p_image):
    target_size = [256,256,3]    

    # pil resize is w h
    img = p_image
    img = img.resize((target_size[1], target_size[0]), resample=1)
    img = np.asarray(img)

    if len(img.shape) == 3 and target_size[-1] == 1:
        img = p_image.convert("L")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)
        img = np.expand_dims(img, -1)
    elif len(img.shape) == 2 and target_size[-1] == 3:
        img = p_image.convert("RGB")
        img = img.resize((target_size[1], target_size[0]), resample=1)
        img = np.asarray(img)

    image = img / 255.0
    # remove alpha channel
    image = image[:, :, 0:3]
    image = np.expand_dims(image, 0)

    return image

if __name__ == '__main__':
    # Start the st functions from here
    # Create Titles
    # NHS tag
    image_class = Image.open\
    ("/home/mark/Documents/python_projects/covid-hack/gui_images/logo.png")
    st.image(image_class,  width=50)
    st.title('X-Ray Image Analysis')
    # NHS tag
    image_class = Image.open\
    ("/home/mark/Documents/python_projects/covid-hack/gui_images/negative_tag.png")
    st.image(image_class,  width=408)


    #Side Bar
    st.sidebar.title('Patient Details')
    # Patient ID
    st.sidebar.header("1- Patient ID")
    app_patient = st.sidebar.text_input("Insert Patient's ID")
    model_date = st.sidebar.date_input("Date admission", value=None)
    model_time = st.sidebar.time_input("Time admission", value=None)

    #Image Upload
    st.sidebar.header("**2- Upload X-ray image**")
    uploaded_image = st.sidebar.file_uploader("Upload an X-Ray Image")

    ## Patient Symptoms
    st.sidebar.header("**3- Symptoms**")
    #st.sidebar.text("Check those boxes symptoms")
    fever = st.sidebar.checkbox('Fever')
    fever_number = st.sidebar.number_input('Insert fever:')
    cough = st.sidebar.checkbox('Cough')
    dyspnea = st.sidebar.checkbox('Dyspnea')
    consolidation = st.sidebar.checkbox('Consolidation')
    opacity = st.sidebar.checkbox('Opacity')
    shadows = st.sidebar.checkbox('Shadows')

    # Run button
    test_button = st.sidebar.button('______Run Test______')

    if(uploaded_image):
        p_image = Image.open(uploaded_image)
        st.image(p_image, width=408)

    # Model params
    img_h = 256
    img_w = 256
    model = stage1_model_factory()
 
    model.load_weights('classifier_Fold0.hd5')

    # Model two params
    model2 = stage2_model_factory()
    model2.load_weights('classifier_meta_Fold0.hd5') # switch here

    # image needs to be (batch, h, w, c) (c = 3 channels)
    if(test_button):
        image = resize_image(p_image)
        print(type(image))
        print(image.shape)
        p = model.predict(image)
        result = probs_to_onehot(p)
        print(result)
        # Display the result
        if(result[0][1]):
            # If covid positive
            image_class = Image.open\
            ("/home/mark/Documents/python_projects/covid-hack/gui_images/positive_button.png")
            st.image(image_class, use_column_width=False)

            # Severity rating
            st.header('**Severity Rating**')
            severity_rating = 6 #this is a dummy value update
            model_severity = st.slider("Severity Case", 0, 10, severity_rating)
            symptom_array = [1,0,0,0,0,0,0]
            p2 = model.predict([image, symptom_array])

        elif(result[0][0]):
            # if covid Negative
            image_class = Image.open\
            ("/home/mark/Documents/python_projects/covid-hack/gui_images/negative_button.png")
            st.image(image_class, use_column_width=False)





    




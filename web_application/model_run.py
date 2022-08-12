# IMPORT PACKAGES

from utils import inp, hist, normalise, ridge_segment, ridge_orient, frequest, ridge_freq, ridge_filter, image_enhance, gabor, thin, pca
import numpy as np
import cv2
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image 
import tensorflow as tf
import os as os
from io import BytesIO
import base64


def set_bg_hack(main_bg):
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# set_bg_hack('background.png') # uncomment to change the background of web app

# MAIN FUNCTIONS
def preprocessing(option_prep,img):
    if option_prep == 'Fingerprint Inpainting & Denoising':
        img = inp(img)

    if option_prep == 'Histogram Equalization (CLAHE)':
        img = hist(img)

    elif option_prep == 'Gabor + Binarization':
        img = gabor(img)

    elif option_prep == 'Thinning':
        img = thin(img)

    elif option_prep == 'All of the above':
        img = inp(img)
        img = hist(img)
        img = gabor(img)
        img = thin(img)
        img = pca(img)

    else:
        img = img
    return img

def teachable_machine_classification(img, weights_file):

    model = load_model(weights_file)
    img_trf = gray2rgb(img)
    img = cv2.resize(img_trf, (256,256), interpolation = cv2.INTER_AREA)
    image = np.array(img)

    output = np.array(img_trf)

    image = image.reshape(1,256,256,3)
    data = image / 255.0

    output = output / 255.0

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction),model.predict(data),output # return position of the highest probability

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.convert('RGB').save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# DESIGN 

st.title('Fingerprint Recognition & Classification')

st.write("This is an image classification web application to predict the 5 general Fingerprint Classifications")
st.write("- Arch, Right Loop, Left Loop, Whorl and Tented Arch.")

option_img = st.selectbox('Do you want to classify a batch of images or a single image?', ('Batch of Images', 'Single Image'))
st.write('You have selected:', option_img)

if option_img == 'Single Image':
    file = st.file_uploader("Please upload an image file", type=["png","jpg","jpeg"])
    if file is None:
        st.text("Please upload an image file")
    else:
        option_prep = st.selectbox('Which data preprocessing would you like to do?', ('Fingerprint Inpainting & Denoising','Histogram Equalization (CLAHE)', 'Gabor + Binarization', 'Thinning', 'All of the above', 'None of the above'))
        st.write('You have selected:', option_prep)

        with open(os.path.join(os.getcwd(),file.name),"wb") as f: 
            f.write(file.getbuffer())    
        img = cv2.imread(os.path.join(os.getcwd(),file.name),0)

        img = preprocessing(option_prep,img)

        prediction, probs, data = teachable_machine_classification(img, "vgg16_pca_preprocessed_256_10_16.h5")
        st.image(data, use_column_width=True,clamp=True)

        if prediction == 0:
            st.write("It is an Arch fingerprint!")
        elif prediction == 1:
            st.write("It is a Left Loop fingerprint!")
        elif prediction == 2:
            st.write("It is a Right Loop fingerprint!")
        elif prediction == 3:
            st.write("It is a Tented Arch fingerprint!")
        else:
            st.write("It is a Whorl fingerprint!")
    
        st.text("Probability Table")
        df = pd.DataFrame(probs, columns=('Arch', 'Left Loop', 'Right Loop', 'Tented Arch', 'Whorl'))
        df.index = ['Prob']
        st.table(df)

        ## Original image came from cv2 format, fromarray convert into PIL format
        result = Image.fromarray(img)
        st.markdown(get_image_download_link(result,'processed_'+file.name,'Download processed_'+ file.name), unsafe_allow_html=True)

if option_img == 'Batch of Images':
    file = st.file_uploader("Please upload the image files", type=["png","jpg", "jpeg"], accept_multiple_files=True)
    if file is None:
        st.text("Please upload the image files")
    else:
        option_prep = st.selectbox('Which data preprocessing would you like to do?', ('Fingerprint Inpainting & Denoising', 'Histogram Equalization (CLAHE)', 'Gabor + Binarization', 'Thinning', 'All of the above', 'None of the above'))
        st.write('You have selected:', option_prep)

        name = []
        fclass = []
        prob = []

        for files in file:
            with open(os.path.join(os.getcwd(),files.name),"wb") as f: 
                f.write(files.getbuffer())    
            img = cv2.imread(os.path.join(os.getcwd(),files.name),0)

            img = preprocessing(option_prep,img)

            name.append(files.name)
            
            prediction, probs, data = teachable_machine_classification(img, "vgg16_pca_preprocessed_256_10_16.h5")
            ind = np.argmax(probs)
            prob.append(probs[0][ind])

            if ind == 0:
                fclass.append('A')

            elif ind == 1:
                fclass.append('LL')

            elif ind == 2:
                fclass.append('RL')

            elif ind == 3:
                fclass.append('TA')

            else:
                fclass.append('W')
            
            result = Image.fromarray(img)
            st.markdown(get_image_download_link(result,'processed_'+files.name,'Download processed_'+ files.name), unsafe_allow_html=True)
            

        df = pd.DataFrame({'File': name, 'Class':fclass, 'Prob' : prob})
        st.write(df)
        csv = df.to_csv().encode('utf-8')
        st.download_button('Download Classifications', csv, file_name='Classification.csv')

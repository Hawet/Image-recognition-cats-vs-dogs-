#import os
#os.system(r'cmd /k "streamlit run C:\Users\hawet\Desktop\хацкерство\kaggle_comps\Dogs_cats\GUI.py"')
import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
from skimage import io
import matplotlib
import os
import streamlit as st
st.header('Dogs vs Cats app')
st.title('dogs_and_cats')
uploaded_file = st.file_uploader("Choose an image", ["jpg","jpeg","png"])
st.write('Chose an image, and i will tell you is it a cat or a dog')
if uploaded_file is not None:
    st.image(uploaded_file)
#st.button('Classification')
if st.button('Classification'):
    model = tf.keras.models.load_model(r'C:\Users\hawet\Desktop\хацкерство\kaggle_comps\Dogs_cats\mymodel.h5')
    uploaded_file = PIL.Image.open(uploaded_file)
    uploaded_file = uploaded_file.resize((100,100))
    uploaded_file = uploaded_file.convert('L')
    uploaded_file = np.asarray(uploaded_file)
    uploaded_file=np.array(uploaded_file)
    img = uploaded_file.reshape(1,100,100,1)
    pred =  model.predict(img)
    if pred<0.5:
        st.write('Its a CAT!!!')
    else:
        st.write('Its a DOG')
    st.write('DOG prob = ',float(pred))
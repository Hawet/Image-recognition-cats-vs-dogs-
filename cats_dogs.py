import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
from skimage import io
import matplotlib
import os
import sklearn.model_selection as sk

def load_data(data_directory):
    i = 1
    labels = []
    images = []
    for filename in os.listdir(data_directory):
        if filename.startswith('cat'):
            labels.append(0)
        else:
            labels.append(1)
        image = PIL.Image.open(data_directory + '\\'+ filename)
        image = image.resize((100,100))
        image = image.convert('L')
        image = np.asarray(image)
        images.append(image)
        print(np.shape(images))
    return images , labels


directory = r'C:\Users\hawet\Desktop\хацкерство\kaggle_comps\Dogs_cats\train'
print(len(os.listdir(directory)))
train_images , train_labels = load_data(directory)
train_images=np.array(train_images)
train_labels=np.array(train_labels)
X_train, X_test, y_train, y_test = sk.train_test_split(train_images,train_labels,test_size=0.10, random_state = 42)
X_test = np.expand_dims(X_test, -1)
X_train = np.expand_dims(X_train, -1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# compile model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10 , validation_data=(X_test, y_test))
model.save('mymodel.h5')


#!/usr/bin/env python
# coding: utf-8

# Ce notebook est destiné à la création d'un modèle de réseau de neurones convolutionnel sur le dataset MNIST

# In[30]:


import tensorflow as tf 


# In[31]:


from tensorflow.keras import layers


# In[32]:


from tensorflow.keras.datasets import mnist


# In[33]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[34]:


import matplotlib.pyplot as plt
import numpy as np


# In[36]:


model = tf.keras.Sequential()

model.add(layers.InputLayer(input_shape=(28, 28)))
model.add(layers.Reshape(target_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add((layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (13, 13, 32))))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation = 'relu'))
model.add(layers.Dense(50, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()


# In[37]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[38]:


x_train = x_train / 255
x_test = x_test / 255

history_1=model.fit(x_train, y_train, epochs = 1)


# In[39]:


predictions = model.predict_classes(x_test)


# In[46]:


accuracy = history_1.history['accuracy']
accuracy


# In[42]:


model.evaluate(x_test, y_test)


# In[43]:


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("mnist_model.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

# Save the model to disk
open("mnist_model_quantized.tflite", "wb").write(tflite_model)


# In[ ]:





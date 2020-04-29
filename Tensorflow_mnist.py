#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.keras import layers


# In[4]:


from tensorflow.keras.datasets import mnist


# In[5]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np


# In[91]:


plt.imshow(x_train[0], cmap='gray')
print(x_test.shape[0], x_test.shape[1])


# In[132]:


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


# In[133]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[135]:


x_train = x_train / 255
x_test = x_test / 255

history_1=model.fit(x_train, y_train, epochs = 1)


# In[136]:


predictions = model.predict_classes(x_test)


# In[137]:


x_axis = np.arange(10000)
plt.plot(x_axis, y_test, 'b', label = 'True value')
plt.plot(x_axis, predictions, 'r', label = 'Predicted Value')
c=0
for i in range(0, 10000):
    if predictions[i] == y_test[i]:
        c+=1
print(c)
print(c*100/10000)


# In[75]:


accuracy = history_1.history['accuracy']


# In[138]:


model.evaluate(x_test, y_test)


# In[139]:


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


# 

# In[140]:



# Instantiate an interpreter for each model
mnist_model = tf.lite.Interpreter('mnist_model.tflite')
mnist_model_quantized = tf.lite.Interpreter('mnist_model_quantized.tflite')
# Allocate memory for each model
mnist_model.allocate_tensors()
mnist_model_quantized.allocate_tensors()


#Compute accuracy for converted model
def evaluate_model(interpreter):

    # Create arrays to store the results
    mnist_model_predictions = []

    # Run each model's interpreter for each value and store the results in arrays
    for i in x_test:

        input_details = interpreter.get_input_details()
        i = np.expand_dims(i, axis= 0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], i)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        mnist_model_predictions.append(np.argmax(output_data))

    count = 0
    for j in range(len(mnist_model_predictions)):
        if mnist_model_predictions[j] == y_test[j]:
            count+=1
    count = 100 * count / len(mnist_model_predictions)
    return count
print("Original model accuracy:", model.evaluate(x_test, y_test))
print("Converted model accuracy:", evaluate_model(mnist_model_quantized))
print("Quantized model accuracy:", evaluate_model(mnist_model_quantized))


# In[141]:


import os
basic_model_size = os.path.getsize("mnist_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("mnist_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)


# In[ ]:





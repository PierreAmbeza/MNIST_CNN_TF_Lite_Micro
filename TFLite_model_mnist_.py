#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf


# In[19]:


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[22]:


# Instantiate an interpreter for each model
mnist_model = tf.lite.Interpreter('mnist_model.tflite')
mnist_model_quantized = tf.lite.Interpreter('mnist_model_quantized.tflite')
# Allocate memory for each model
mnist_model.allocate_tensors()
mnist_model_quantized.allocate_tensors()
x_test = x_test / 255

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
print("Original model accuracy:", evaluate_model(mnist_model))
print("Quantized model accuracy:", evaluate_model(mnist_model_quantized))


# In[10]:


import os
basic_model_size = os.path.getsize("mnist_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("mnist_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)


# In[11]:


tf.__version__


# In[12]:


get_ipython().system('xxd -i mnist_model_quantized.tflite > mnist_model_quantized.cc')


# In[13]:


get_ipython().system('xxd -i mnist_model.tflite > mnist_model.cc')


# In[ ]:





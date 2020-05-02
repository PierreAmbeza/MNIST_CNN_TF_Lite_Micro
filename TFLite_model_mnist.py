#!/usr/bin/env python
# coding: utf-8

# # MNIST-CNN-TFLite Micro
# 
# Dans un précédent notebook, nous avons créer un modèle capable de prédire le chiffre écrit dans une image du dataset **MNIST** et converti ce modèle en version **TFLite**. Nous allons donc tester dans ce notebook la précision de ces modèles convertis par rapport au modèle d'origine. 

# In[83]:


import tensorflow as tf


# In[91]:


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[92]:


import numpy as np
x_test = x_test / 255


# Nous allons créer une fonction **evaluate_model** qui permettra de faire les prédictions de nos modèles convertis et de calculer la précision de ces derniers. Enfin, nous comparons cette précision avec celle du modèle d'origine.

# In[101]:


# Instantiate an interpreter for each model
mnist_model = tf.lite.Interpreter('mnist_model-2.tflite')
mnist_model_quantized = tf.lite.Interpreter('mnist_model_quantized-2.tflite')
# Allocate memory for each model
mnist_model.allocate_tensors()
mnist_model_quantized.allocate_tensors()
#Normalize test data


#Compute accuracy for converted models
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
    print(count)
    count = 100 * count / len(mnist_model_predictions)
    return count

#new_model = tf.keras.models.load_model('mnist_model_origin')

#loss, acc = new_model.evaluate(x_test,  y_test)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
#print("TFLite model accuracy:", evaluate_model(mnist_model))
print("TFLite model accuracy: {:5.2f}".format(evaluate_model(mnist_model)))
print("TFLite Quantized model accuracy:", evaluate_model(mnist_model_quantized))


# Nous voyons que la précision des modèles convertis est égale ou presque à celle du modèle d'origine.
# 
# Désormais, nous pouvons regarder la taille de chacun de nos modèles pour se rendre compte du gain en taille obtenu suite à la conversion en **TF Lite**.

# In[94]:


'''import os
#Compare size of converted models with origina model
basic_model_size = os.path.getsize("mnist_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("mnist_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
ratio = basic_model_size / quantized_model_size
print("Ratio between basic_model and quantized model is %d" % ratio)
original_model_size = os.path.getsize("mnist_model_origin/saved_model.pb")+os.path.getsize("mnist_model_origin/variables/variables.data-00000-of-00001")+os.path.getsize("mnist_model_origin/variables/variables.index")
print("Original model is %d bytes" % original_model_size)
ratio = original_model_size / basic_model_size
print("Ratio between basic_model and quantized model is %d" % ratio)'''


# Nous pouvons voir que le modèle d'origine fait 1,6 Mo environ. Le modèle **TF Lite** non quantifié est deux fois plus petit avec 700 Ko environ et le modèle **TF Lite** quantifié est quant à lui 3 fois plus petit que le modèle non quantifié et fait environ 180 Ko.
# 
# Maintenant nous allons convertir les deux modèles **TF Lite** en modèles **TF Lite Micro** à l'aide de la commande **xxd**.

# In[95]:


#!xxd -i mnist_model_quantized.tflite > mnist_model_quantized.cc


# In[96]:


#!xxd -i mnist_model.tflite > mnist_model.cc


# Nous ajoutons également une fonction permettant de créer un fichier **.h** comportant les données de test et que nous pourrons utiliser pour faire les prédictions avec **TF Lite Micro**.

# In[98]:


'''def write_test_data_header(file_name, x, y, n_values):
  """
  Function to write a c header file containing a set of test input and output
  data for this classification model
  :param file_name: name of the header file to create
  :param x: input data numpy array
  :param y: output data numpy array, first dimension much match above
  :return: Nothing
  """

  with open(file_name, "w") as header:
    header.write("// MNIST test data\n\n")

    header.write("int mnistSampleCount = %d;\n\n" % n_values)

    header.write("float mnistInput[%d][784] = {\n" % n_values)
    for i in range(n_values):
      if i != 0:
        header.write(",\n")
      header.write("{ ")
      row = x[i].reshape(1, 784).astype(np.int)
      np.savetxt(header, row, delimiter=', ', newline='', fmt='%d')
      header.write(" }")
    header.write("};\n\n")

    header.write("int mnistOutput[%d] = { " % n_values)
    np.savetxt(header,
               y[0:n_values].reshape(1, n_values),
               delimiter=', ',
               newline='',
               fmt='%d')
    header.write(" };\n")

write_test_data_header("mnist_test_data.h", x_test, y_test, 50)'''


# In[100]:


#model_test_size = os.path.getsize("mnist_test_data.h")
#print("La taille du fichier de test est de %d octets" % model_test_size)


# In[ ]:





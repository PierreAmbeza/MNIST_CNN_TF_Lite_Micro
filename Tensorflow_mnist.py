#!/usr/bin/env python
# coding: utf-8

# # MNIST-CNN-TFLite Micro
# Ce notebook est destiné à la création d'un modèle de réseau de neurones convolutionnel sur le dataset MNIST. Pour cela, nous allons utiliser les libraires de machine learning Tensorflow et Keras.

# In[75]:


import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Nous pouvons charger les données du dataset mnist à l'aide de la fonction load_data, ce qui nous permet d'avoir des données d'entrainement et de test. On affiche ensuite la forme des données que nous passeront en entrée à notre modèle. Cela nous servira ensuite à déterminer l'**input_shape** de notre modèle.

# In[100]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)


# Nous créons désormais notre modèle à partir de la fonction Sequential de Keras. Ce modèle est composé d'une première couche de **Reshape** des données d'entrées, ensuite de couches de **Conv2D** et de **MaxPooling** et enfin, d'une couche pour "applatir" les données en sortie après la seconde couche de **MaxPooling** et de couches **Dense** pour déterminer la catégorie à laquelle appartient l'image d'entrée.

# In[97]:


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


# Nous devons désormais définir comment se fera l'apprentissage, notamment le calcul de la **loss**. Dans notre cas, nous avons plusieurs catégories et nous devons donc utiliser une **categorical_crossentropy**. Nous ajoutons également l'**accuracy** dans les **metrics** car nous voulons savoir la précision de notre modèle.

# In[98]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# La prochaine étape est désormais d'entraîner notre modèle sur les données d'entrainement. Mais avant cela, nous normalisons nos images d'entrainement et de test en divisant chaque valeur de pixel par la valeur maximale soit 255.

# In[101]:


x_train = x_train / 255
x_test = x_test / 255

history_1=model.fit(x_train, y_train)


# Nous voyons donc que nous avons obtenu une précision de notre modèle de 95.5% pour l'entrainement de notre modèle.
# 
# Par la suite, nous allons prédire les valeurs de sorties pour les données de test. Comme nous avons plusieurs classes de chiffres, nous utilisons la méthode **predict_classes**. Nous allons ensuite comparer ces prédictions avec les vraies valeurs et calculer la précision.

# In[90]:


predictions = model.predict_classes(x_test)

accuracy = 0
for i in range(0, predictions.shape[0]):
    if predictions[i] == y_test[i]:
        accuracy += 1
        
accuracy = accuracy / 100
print(accuracy)


# Nous voyons que nous obtenons une précision de 98.32 % sur les données d'entrainement. Nous pouvons vérifier ce calcul à l'aide de la fonction **evaluate**.

# In[102]:


loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy*100)


# Nous obtenons bel et bien la même **accuracy**.
# 
# Nous savons désormais que notre modèle est capable d'identifier quel est le chiffre manuscrit sur les images du dataset **MNIST** avec une grande précision. Nous allons donc pouvoir convertir ce modèle en version **Lite** dans un premier temps puis ensuite en version **Micro**. Nous allons convertir ce modèle tout d'abord sans quantification puis ensuite avec quantification. Le modèle avec quantification sera donc de taille plus petite. En revanche, il semble qu'actuellement **TF Lite Micro** ne supporte pas la quantification. Enfin, nous enregistrons également notre modèle original.

# In[103]:


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("mnist_model_dense.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

# Save the model to disk
open("mnist_model_quantized_dense.tflite", "wb").write(tflite_model)

model.save('mnist_model_dense_origin')


# In[ ]:





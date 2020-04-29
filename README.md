# Mnist_CNN_TFLite_Micro
Projet de Big Data et Technologies Embarquées


Ce projet a pour but d'implémenter un modèle de réseaux de neurones convolutionnel et d'utiliser ce modèle avec la librairie Tensorflow Lite Micro de Google. Pour cela, nous avons utilisé le dataset MNIST, comportant des chiffres manuscrits.

## Création du modèle

La première étape dans ce projet est de créer notre modèle en utilisant la libraire Tensorflow classique. Le modèle est définie dans le fichier **Tensorflow_mnist_.py**.
Notre modèle est représenté de la façon suivante:

![Description du modele](Model.png)
Format: ![Alt Text](url)

Ce fichier permet donc de créer notre modèle, de l'entraîner et enfin de sauvegarder ce modèle. Ici, deux versions sont sauvegardées:

1. **mnist_model.tflite**:ce fichier correspond à une version lite du modèle défini. C'est ce fichier que nous utiliserons par la suite pour TF Lite Micro
1. **model_mnist_quantized.tflite**:ce fichier correspond à une version lite également mais avec une taille encore plus petite que mnist_model.tflite. Il est important d'avoir des modèles de taille la plus petite possible lorsque nous souhaitons les déployer sur des microcontroleurs.

## Test des modèles convertis

Une fois que nous avons converti nos modèles, il faut bien évidemment les tester. Le test s'effectue avec le fichier **TFLite_model_mnist_.py**


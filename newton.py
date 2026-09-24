import tensorflow as tf
from keras import layers, models, utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
 
# Carregar dados MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
# Pré-processamento dos dados
# Redimensionar e normalizar as imagens de treinamento e teste
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
 
# Converter rótulos em codificação one-hot
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)
 
# Construir o modelo CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Camada de entrada explicitando as dimensões
    layers.Conv2D(32, (3, 3), activation='relu'),  # Camada convolucional com 32 filtros de tamanho 3x3
    layers.MaxPooling2D((2, 2)),  # Camada de pooling para redução de dimensionalidade
    layers.Conv2D(64, (3, 3), activation='relu'),  # Segunda camada convolucional com 64 filtros de tamanho 3x3
    layers.MaxPooling2D((2, 2)),  # Outra camada de pooling
    layers.Conv2D(64, (3, 3), activation='relu'),  # Terceira camada convolucional com 64 filtros de tamanho 3x3
    layers.Flatten(),  # Achatamento dos dados para entrada na camada densa
    layers.Dense(64, activation='relu'),  # Camada densa com 64 unidades e função de ativação ReLU
    layers.Dense(10, activation='softmax')  # Camada de saída com 10 unidades para classificação de 10 classes e função de ativação softmax
])
 
# Compilar o modelo
model.compile(optimizer='adam',  # Otimizador Adam
              loss='categorical_crossentropy',  # Função de perda para classificação categórica
              metrics=['accuracy'])  # Métrica de avaliação da acurácia
 
# Treinar o modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
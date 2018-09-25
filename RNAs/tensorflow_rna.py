# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np

# Dataset
dataset_motorista = np.array(np.genfromtxt('input.csv', delimiter=',')) # 1380 x 4
dataset_output = np.array(np.genfromtxt('output.csv', delimiter=','))           # 1380 x 1

test_dataset_motorista = []
test_dataset_output = []

# Criando modelo da rede neural com:
# - 1 camada de entrada com 4 neurônios
# - 2 camada escondida com 7 neurônios
# - 1 camada de saída com 1 neurônio que representa o motorista ou nao
model = keras.Sequential([
    keras.layers.Dense(4, input_shape=(4,), activation=tf.nn.relu),
    keras.layers.Dense(7, activation=tf.nn.relu),
    keras.layers.Dense(7, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

# Compilando a rede neural
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

# Esta é a etapa do treino, é aqui onde passamos o dataset
# para treinar e as labels esperadas
model.fit(dataset_motorista, dataset_output, epochs=100)

# Recuperando a taxa de loss e a precisão do nosso modelo
# através de um outro dataset para teste
test_loss, test_acc, test_binarycrossentropy = model.evaluate(dataset_motorista, dataset_output)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Recuperando as predictions
predictions = model.predict(dataset_motorista)

for i in range(0, len(predictions)):
    print("[",i,"]", predictions[i], " => ", dataset_output[i])
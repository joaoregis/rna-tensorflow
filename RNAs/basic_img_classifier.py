# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import matplotlib.pyplot as plt
import HelperFunctions as hf

fashion_mnist = keras.datasets.fashion_mnist

# Buscando dataset da base de dados do tensorflow
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Classes de roupas/ sapatos que a rede neural deve reconhecer
class_names = ['T-shirt', 'Calça', 'Pulôver', 'Vestido', 'Casaco'
               'Sandália', 'Camisa', 'Sapatilha', 'Bolsa', 'Bota']

# Essa linha transforma todos os pixels das imagens em
# valores entre 0.0 e 1.0 sem perder a característica
train_images = train_images / 255.0
test_images = test_images / 255.0

# Criando modelo da rede neural com:
# - 1 camada de entrada com (28 * 28) neurônios
# - 1 camada escondida com 128 neurônios
# - 1 camada de saída com 10 neurônios que representam
#   a posição no vetor de class names
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compilando a rede neural, ou seja, criando-a em memória
# e definindo as métricas que a biblioteca deve se ater
# no caso é definido "accuracy" ou "precisão".
# Accuracy irá calcular o quanto a prediction está próxima das labels definidas
# Existem dezenas de métricas possíveis, como por exemplo cálculo de falsos-positivos e outros,
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Esta é a etapa do treino, é aqui ohnde passamos o dataset
# para treinar e as labels esperadas
model.fit(train_images, train_labels, epochs = 5)

# Recuperando a taxa de loss e a precisão do nosso modelo
# através de um outro dataset para teste
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Recuperando as predictions para as
# imagens do dataset de teste
predictions = model.predict(test_images)

# Montando gráficos e exibindo resultados
imageIndexToShow = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

hf.plot_image(imageIndexToShow, predictions, test_labels, test_images, class_names)

plt.subplot(1,2,2)

hf.plot_value_array(imageIndexToShow, predictions, test_labels)

plt.show()

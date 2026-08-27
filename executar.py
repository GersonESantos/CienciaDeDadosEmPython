import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Carrega os dados (do 01pY.PY)
print("Baixando/carregando dataset MNIST...")
(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

# 2. Plota os 5 primeiros dígitos (do 02PY.py)
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.tight_layout()
    plt.imshow(x_treino[i].reshape(28, 28), cmap='gray')
    plt.title('Rótulo: {}'.format(y_treino[i]))
    plt.xticks([])
    plt.yticks([])

# 3. Exibe a janela gráfica
plt.show()

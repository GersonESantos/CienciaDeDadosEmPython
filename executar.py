import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("=" * 60)
print("🧠 Pipeline de Pré-processamento MNIST para LeNet-5")
print("=" * 60)

# 1. Carregamento dos dados (data_loader.py)
print("\n[Etapa 1] Carregando dataset MNIST original...")
(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()
print(f"-> Amostras de Treino: {x_treino.shape}, Amostras de Teste: {x_teste.shape}")

# 2. Visualização dos 5 primeiros dígitos (visualization.py)
print("\n[Etapa 2] Exibindo os 5 primeiros dígitos (feche a janela do gráfico para continuar)...")
plt.figure(figsize=(10, 2.5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.tight_layout()
    plt.imshow(x_treino[i].reshape(28, 28), cmap='gray')
    plt.title('Rótulo: {}'.format(y_treino[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 3. Separação dos subconjuntos e expansão de dimensão (data_split.py)
print("\n[Etapa 3] Separando dados em Treino (55k), Validação (5k) e Teste (10k)...")
quantidade_dados_treino = 55000

x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
y_validacao = y_treino[quantidade_dados_treino:]

x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
y_treino = y_treino[:quantidade_dados_treino]

x_teste = x_teste[..., np.newaxis]

print(f"-> Formato com canal: Treino {x_treino.shape}, Validação {x_validacao.shape}, Teste {x_teste.shape}")

# 4. Preenchimento com Zeros (data_padding.py / 28x28 -> 32x32)
print("\n[Etapa 4] Aplicando Zero-Padding para a arquitetura LeNet-5 (32x32x1)...")
x_treino = np.pad(x_treino, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_validacao = np.pad(x_validacao, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_teste = np.pad(x_teste, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("\n--- RESUMO FINAL DOS TENSORES PRONTOS PARA A LENET-5 ---")
print(f"Conjunto de Treinamento: {x_treino.shape}")
print(f"Conjunto de Validação:   {x_validacao.shape}")
print(f"Conjunto de Testes:      {x_teste.shape}")
print("=" * 60)

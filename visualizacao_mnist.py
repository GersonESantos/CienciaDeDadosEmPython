import tensorflow as tf
import matplotlib.pyplot as plt

def carregar_e_visualizar():
    # 1. Carregamento dos dados conforme as diretrizes do repositório
    print("Carregando o dataset MNIST...")
    (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()
    
    print(f"Dados carregados com sucesso!")
    print(f"Conjunto de Treinamento: {x_treino.shape[0]} imagens de tamanho {x_treino.shape[1]}x{x_treino.shape[2]}")
    print(f"Conjunto de Teste: {x_teste.shape[0]} imagens de tamanho {x_teste.shape[1]}x{x_teste.shape[2]}\n")

    # 2. Visualização dos 5 primeiros registros do conjunto de treinamento (Sanity Check)
    print("Gerando a visualização dos 5 primeiros registros...")
    plt.figure(figsize=(10, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.tight_layout()
        # Redimensiona e exibe a imagem em escala de cinza
        plt.imshow(x_treino[i].reshape(28, 28), cmap='gray')
        # Define o título com o rótulo correspondente
        plt.title(f'Rótulo: {y_treino[i]}')
        # Remove as marcações dos eixos para uma visualização mais limpa
        plt.xticks([])
        plt.yticks([])
    
    # Exibe a janela com os gráficos
    plt.show()

if __name__ == "__main__":
    carregar_e_visualizar()

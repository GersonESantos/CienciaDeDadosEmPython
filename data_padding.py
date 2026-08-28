import numpy as np
import tensorflow as tf

def carregar_preparar_e_pad(quantidade_dados_treino=55000):
    """
    Executa o pipeline completo de preparação dos dados para a LeNet-5:
    1. Carregamento do MNIST (60k treino, 10k teste)
    2. Divisão em Treino (55k), Validação (5k) e Teste (10k)
    3. Expansão de dimensão para canal único (28x28x1)
    4. Zero-Padding de 2 pixels nas bordas para adequação ao tamanho da LeNet-5 (32x32x1)
    """
    print("1. Carregando dados MNIST...")
    (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

    print(f"2. Dividindo Treino ({quantidade_dados_treino}) e Validação ({len(x_treino) - quantidade_dados_treino})...")
    x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
    y_validacao = y_treino[quantidade_dados_treino:]

    x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
    y_treino = y_treino[:quantidade_dados_treino]

    x_teste = x_teste[..., np.newaxis]

    print("3. Aplicando Zero-Padding (28x28 -> 32x32) para a arquitetura LeNet-5...")
    # Adiciona 2 pixels de zeros nas bordas: ((batch), (altura), (largura), (canal))
    x_treino = np.pad(x_treino, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validacao = np.pad(x_validacao, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_teste = np.pad(x_teste, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print("\nInformações sobre as mudanças dos dados de entrada:\n")
    print('Conjunto de Treinamento: {}'.format(x_treino.shape))
    print('Conjunto de Validação:   {}'.format(x_validacao.shape))
    print('Conjunto de Testes:      {}'.format(x_teste.shape))

    return (x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste)

if __name__ == "__main__":
    carregar_preparar_e_pad()

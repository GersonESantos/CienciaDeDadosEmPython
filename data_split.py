import numpy as np
import tensorflow as tf

def carregar_e_separar_dados(quantidade_dados_treino=55000):
    """
    Carrega a base MNIST e organiza em 3 subconjuntos (Treino, Validação, Teste)
    com a adição da dimensão de canal (28, 28, 1) para redes neurais convolucionais (CNNs).
    """
    print("1. Carregando dados originais do MNIST...")
    (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

    print(f"2. Separando em Treino ({quantidade_dados_treino}) e Validação ({len(x_treino) - quantidade_dados_treino})...")
    
    # Conjunto de Validação: do índice 55000 até o final (5.000 amostras) + canal (np.newaxis)
    x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
    y_validacao = y_treino[quantidade_dados_treino:]

    # Conjunto de Treinamento: do início até o índice 55000 (55.000 amostras) + canal (np.newaxis)
    x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
    y_treino = y_treino[:quantidade_dados_treino]

    # Conjunto de Testes: 10.000 amostras + canal (np.newaxis)
    x_teste = x_teste[..., np.newaxis]

    # Exibição das informações e formatos
    print("\n--- Resumo dos Conjuntos de Dados ---")
    print('Formato da Imagem: {}'.format(x_treino[0].shape))
    print('Conjunto de Treinamento: {} registros'.format(len(x_treino)))
    print('Conjunto de Validação:   {} registros'.format(len(x_validacao)))
    print('Conjunto de Testes:      {} registros'.format(len(x_teste)))

    return (x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste)

if __name__ == "__main__":
    carregar_e_separar_dados()

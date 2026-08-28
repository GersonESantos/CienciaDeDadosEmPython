import numpy as np
import tensorflow as tf

def carregar_preparar_e_normalizar(quantidade_dados_treino=55000):
    """
    Pipeline completo de pré-processamento para a LeNet-5:
    1. Carregamento do MNIST (60k treino, 10k teste)
    2. Divisão em Treino (55k), Validação (5k) e Teste (10k)
    3. Expansão de dimensão de canal (28x28x1)
    4. Preenchimento Zero-Padding (28x28 -> 32x32)
    5. Normalização de pixels (0 a 255 -> 0.0 a 1.0)
    """
    print("1. Carregando dados MNIST...")
    (x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

    print(f"2. Dividindo Treino ({quantidade_dados_treino}) e Validação ({len(x_treino) - quantidade_dados_treino})...")
    x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
    y_validacao = y_treino[quantidade_dados_treino:]

    x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
    y_treino = y_treino[:quantidade_dados_treino]

    x_teste = x_teste[..., np.newaxis]

    print("3. Aplicando Zero-Padding (28x28 -> 32x32)...")
    x_treino = np.pad(x_treino, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validacao = np.pad(x_validacao, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_teste = np.pad(x_teste, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print("4. Normalizando valores dos pixels de [0, 255] para [0.0, 1.0]...")
    normalizar_dados = lambda t: t / 255.0
    x_treino = normalizar_dados(x_treino)
    x_validacao = normalizar_dados(x_validacao)
    x_teste = normalizar_dados(x_teste)

    print("\n--- RESUMO DOS DADOS NORMALIZADOS ---")
    print(f"Formato Treinamento: {x_treino.shape} (Min: {x_treino.min():.2f}, Max: {x_treino.max():.2f})")
    print(f"Formato Validação:   {x_validacao.shape} (Min: {x_validacao.min():.2f}, Max: {x_validacao.max():.2f})")
    print(f"Formato Testes:      {x_teste.shape} (Min: {x_teste.min():.2f}, Max: {x_teste.max():.2f})")

    return (x_treino, y_treino), (x_validacao, y_validacao), (x_teste, y_teste)

if __name__ == "__main__":
    carregar_preparar_e_normalizar()

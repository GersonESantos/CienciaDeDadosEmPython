import numpy as np

# Preenchimento (padding) dos dados de entrada
# Adiciona 2 pixels de zeros nas bordas superior/inferior e esquerda/direita: 28x28 -> 32x32
x_treino = np.pad(x_treino, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_validacao = np.pad(x_validacao, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_teste = np.pad(x_teste, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print('Informações sobre as mudanças dos dados de entrada: ', end='\n\n')
print('Conjunto de Treinamento: {}'.format(x_treino.shape))
print('Conjunto de Validação:   {}'.format(x_validacao.shape))
print('Conjunto de Testes:      {}'.format(x_teste.shape))

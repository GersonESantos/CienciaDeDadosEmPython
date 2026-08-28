#### Separação dos conjuntos de dados

O próximo passo consiste em organizar os dados em três subconjuntos.

import numpy as np
 
quantidade_dados_treino = 55000
 
x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
y_validacao = y_treino[quantidade_dados_treino:]
x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
y_treino = y_treino[:quantidade_dados_treino]
 
x_teste =  x_teste[..., np.newaxis]
 
print('Formato da Imagem:{}'.format(x_treino[0].shape), end = '\n')
print('Conjunto de Treinamento:{} registros'.format(len(x_treino)))
print('Conjunto de Validação:{}registros'.format(len(x_validacao)))
print('Conjunto de Testes:      {} registros'.format(len(x_teste)))

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
 

 Logo no início do código, importamos o pacote “numpy” que faz a manipulação de vetores. A base MNIST possui um total de 70.000 registros, sendo que 60.000 são separados para treinamento e 10.000 para testes. Aqui, nós separamos os dados de treinamento em duas partes: 55.000 registros para treinamento e 5.000 para validação do modelo.

O resultado da execução do código é:

Formato da Imagem: (28, 28, 1);
Conjunto de Treinamento: 55000 registros;
Conjunto de Validação: 5000 registros;
Conjunto de Testes: 10000 registros.
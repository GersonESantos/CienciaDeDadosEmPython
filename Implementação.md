# Implementação da arquitetura LeNet-5

Agora, vamos estudar o passo a passo de uma implementação da arquitetura LeNet-5, aplicada a reconhecer imagens na base de dados MNIST, que é uma das mais utilizadas para testes de algoritmos de aprendizado de máquina.
Na nossa implementação, optamos por importar as bibliotecas nos blocos de programação em que elas são necessárias, pois fica mais claro entendermos a necessidade de utilizá-las.

O projeto está dividido da seguinte maneira:

#### Carregamento dos dados

A base de dados MNIST já está disponível no **Tensorflow**, inclusive, com a separação dos dados para treinamento e teste. O que precisamos é importar o Tensorflow e, depois, carregar os dados.

O código que implementamos para visualização dos dados é expresso por:

import tensorflow as tf
(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

Perceba que, quando carregamos os dados, temos dois pares ordenados: um de **treinamento** e outro de **testes**. Outro ponto a se observar é que as variáveis **“x_treino”** e **“x_teste”** se referem aos **dados das imagens**, enquanto as variáveis **“y_treino”** e **“y_teste”** se referem aos **rótulos das variáveis**.

#### Visualização dos dados

Agora que já temos nossos dados carregados na memória, podemos visualizá-los para verificar se tudo aconteceu corretamente até determinado momento. Para isso, vamos utilizar a biblioteca **“matplotlib.pyplot”** para imprimir os dados.

O código que implementamos para visualização dos dados é fornecido por:

import matplotlib.pyplot as plt
 
for i in range(5):
  plt.subplot(1, 5, i+1)
  plt.subplot(1, 5, i+1)
  plt.tight_layout()
  plt.imshow(x_treino[i].reshape(28, 28), cmap='gray')
  plt.title('Rótulo:{}'.format(y_treino[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

Logo no início do código, importamos a biblioteca “matplotlib.pyplot”. Em seguida, usamos um laço para imprimir os cinco primeiros registros do conjunto de dados de treinamento.

Veja a seguir a saída do programa, onde são apresentados o rótulo e a imagem.
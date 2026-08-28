Preenchimento (padding) dos dados de entrada
Agora, precisamos fazer um ajuste nos dados de entrada. A camada de entrada da arquitetura LeNet-5 consiste em imagens com dimensões de 32 × 32. Como vimos da execução do bloco anterior, as imagens do MNIST têm dimensões 28 × 28.

Então, para aplicar a LeNet-5, precisamos preencher a entrada com zeros (0) a fim de torná-la 32 × 32. Para isso, vamos usar a função “pad” da biblioteca numpy.

O código para preenchimento dos dados fica assim:


x_treino = np.pad(x_treino, ((0,0),(2,2),(2,2),(0,0)), 'constant')
 x_validacao = np.pad(x_validacao, ((0,0),(2,2),(2,2),(0,0)), 'constant')
 x_teste = np.pad(x_teste, ((0,0),(2,2),(2,2),(0,0)), 'constant')
 
 print('Informações sobre as mudanças dos dados de entrada: ', end='\n\n')
 print('Conjunto de treinamento: {}'.format(x_treino.shape))
 print('Conjunto de Validação: {}'.format(x_validacao.shape))
 print('Conjunto de Testes: {}'.format(x_teste.shape))

 Veja a seguir como é dado o resultado da execução do código.

Informações sobre as mudanças dos dados de entrada:

Conjunto de Treinamento: (55000, 32, 32, 1).
Conjunto de Validação: (5000, 32, 32, 1).
Conjunto de Testes: (10000, 32, 32, 1).
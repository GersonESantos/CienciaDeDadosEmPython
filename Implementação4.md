#### Normalização dos dados

Os dados de entrada do MNIST estão em escalas de cinza, ou seja, vão de 0 a 255. Então vamos transformá-los para dados no intervalo de 0 a 1. Para isso, basta dividirmos os valores das variáveis por 255.

O código para normalização dos dados é:

```python
normalizar_dados = lambda t: t / 255.0
x_treino = normalizar_dados(x_treino)
x_validacao = normalizar_dados(x_validacao)
x_teste = normalizar_dados(x_teste)
```
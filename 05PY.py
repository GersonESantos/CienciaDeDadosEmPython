# Normalização dos dados de entrada (0 a 255 -> 0.0 a 1.0)
normalizar_dados = lambda t: t / 255.0

x_treino = normalizar_dados(x_treino)
x_validacao = normalizar_dados(x_validacao)
x_teste = normalizar_dados(x_teste)

print('Valores mínimo e máximo após normalização:')
print('Mínimo: {:.2f}, Máximo: {:.2f}'.format(x_treino.min(), x_treino.max()))

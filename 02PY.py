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
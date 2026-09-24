import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def plot_confusion_matrix(model, test_images, test_labels):
	"""Gera a matriz de confusão para um modelo já treinado."""
	predictions = model.predict(test_images, verbose=0)
	predicted_labels = np.argmax(predictions, axis=1)
	true_labels = (
		np.argmax(test_labels, axis=1)
		if test_labels.ndim > 1
		else test_labels
	)

	confusion_matrix = tf.math.confusion_matrix(
		labels=true_labels,
		predictions=predicted_labels,
		num_classes=10,
	).numpy()

	plt.figure(figsize=(10, 8))
	sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='viridis')
	plt.xlabel('Classe Predita')
	plt.ylabel('Classe Verdadeira')
	plt.title('Matriz de Confusão')
	plt.tight_layout()
	plt.show()

	return confusion_matrix


if __name__ == '__main__':
	from newton import model, test_images, test_labels

	plot_confusion_matrix(model, test_images, test_labels)

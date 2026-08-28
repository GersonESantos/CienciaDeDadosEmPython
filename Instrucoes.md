# 📋 Guia de Instruções de Execução Passo a Passo

Este documento contém o passo a passo detalhado para configurar o ambiente e executar cada uma das etapas do pipeline de pré-processamento para a rede **LeNet-5** no dataset **MNIST**.

---

## 🛠️ 1. Pré-requisitos e Instalação

Antes de executar os scripts, certifique-se de que as dependências necessárias estão instaladas no seu computador.

Abra o terminal (**PowerShell** ou **Prompt de Comando**) e execute:

```powershell
pip install tensorflow matplotlib numpy
```

> **Verificação:** Para confirmar se tudo foi instalado com sucesso, execute:
> ```powershell
> python -c "import tensorflow, matplotlib, numpy; print('Ambiente configurado com sucesso!')"
> ```

---

## 🚀 2. Opção Rápida: Executar Todo o Pipeline

Se você deseja executar todas as 5 etapas de uma só vez com logs explicativos e visualização gráfica dos dados, execute o script unificado:

```powershell
python executar.py
```

> **Nota:** Durante a execução, uma janela do **Matplotlib** será aberta na tela com os 5 primeiros dígitos. **Feche a janela do gráfico** para que o script continue para as etapas de divisão, padding e normalização.

---

## 🧩 3. Execução Passo a Passo (Módulos Individuais)

Se preferir acompanhar e entender cada etapa separadamente:

### Etapa 1: Carregamento dos Dados
- **Arquivo:** [`data_loader.py`](data_loader.py)
- **O que faz:** Importa o TensorFlow e faz o download automático das matrizes de treino e teste do MNIST.
- **Comando:**
  ```powershell
  python data_loader.py
  ```

---

### Etapa 2: Inspeção Visual dos Dados
- **Arquivo:** [`visualizacao_mnist.py`](visualizacao_mnist.py) ou [`visualization.py`](visualization.py)
- **O que faz:** Renderiza uma grade horizontal com as 5 primeiras imagens do conjunto de treinamento e seus respectivos rótulos (Ground Truth).
- **Comando:**
  ```powershell
  python visualizacao_mnist.py
  ```

---

### Etapa 3: Separação dos Subconjuntos e Expansão de Canal
- **Arquivo:** [`data_split.py`](data_split.py)
- **O que faz:** 
  1. Divide os 60.000 dados originais em: **55.000 para Treinamento** e **5.000 para Validação** (mantendo 10.000 para Teste).
  2. Adiciona a dimensão de canal único com `np.newaxis` (`28x28` $\rightarrow$ `28x28x1`) para compatibilidade com camadas `Conv2D`.
- **Comando:**
  ```powershell
  python data_split.py
  ```

**Saída esperada no terminal:**
```text
Formato da Imagem: (28, 28, 1)
Conjunto de Treinamento: 55000 registros
Conjunto de Validação:   5000 registros
Conjunto de Testes:      10000 registros
```

---

### Etapa 4: Preenchimento de Zeros (Zero-Padding para 32x32)
- **Arquivo:** [`data_padding.py`](data_padding.py)
- **O que faz:** Adiciona 2 pixels de zeros nas 4 bordas de cada imagem ($28 \times 28 \rightarrow 32 \times 32$) para adequação à camada de entrada da arquitetura **LeNet-5**.
- **Comando:**
  ```powershell
  python data_padding.py
  ```

**Saída esperada no terminal:**
```text
Informações sobre as mudanças dos dados de entrada:

Conjunto de Treinamento: (55000, 32, 32, 1)
Conjunto de Validação:   (5000, 32, 32, 1)
Conjunto de Testes:      (10000, 32, 32, 1)
```

---

### Etapa 5: Normalização dos Dados ([0.0, 1.0])
- **Arquivo:** [`data_normalization.py`](data_normalization.py)
- **O que faz:** Divide os valores dos pixels por 255.0 para trazê-los do intervalo $[0, 255]$ para $[0.0, 1.0]$, garantindo estabilidade e rápida convergência do gradiente.
- **Comando:**
  ```powershell
  python data_normalization.py
  ```

**Saída esperada no terminal:**
```text
--- RESUMO DOS DADOS NORMALIZADOS ---
Formato Treinamento: (55000, 32, 32, 1) (Min: 0.00, Max: 1.00)
Formato Validação:   (5000, 32, 32, 1) (Min: 0.00, Max: 1.00)
Formato Testes:      (10000, 32, 32, 1) (Min: 0.00, Max: 1.00)
```

---

## 🌐 4. Executando no Google Colab (Passo a Passo em Células)

Você pode acessar o notebook pronto diretamente no link:
👉 **[Abrir Notebook no Google Colab](https://colab.research.google.com/drive/1qq1aWl2uJs-BvFPtWcjmUAmY2Gxi3-PC#scrollTo=edkI5lnRDLXR)**

Caso queira criar um notebook novo do zero:
1. Acesse [colab.research.google.com](https://colab.research.google.com/) e crie um **Novo Notebook**.
2. Crie e execute as 5 células em sequência (<kbd>Shift</kbd> + <kbd>Enter</kbd>):

### Célula 1 — Carga dos Dados
```python
import tensorflow as tf
(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()
print("Dados carregados com sucesso!")
```

### Célula 2 — Visualização Gráfica
```python
import matplotlib.pyplot as plt

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.tight_layout()
    plt.imshow(x_treino[i].reshape(28, 28), cmap='gray')
    plt.title(f'Rótulo: {y_treino[i]}')
    plt.xticks([])
    plt.yticks([])
plt.show()
```

### Célula 3 — Divisão dos Conjuntos e Canal
```python
import numpy as np

quantidade_dados_treino = 55000

x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
y_validacao = y_treino[quantidade_dados_treino:]

x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
y_treino = y_treino[:quantidade_dados_treino]

x_teste = x_teste[..., np.newaxis]

print("Treino:", x_treino.shape)
print("Validação:", x_validacao.shape)
print("Teste:", x_teste.shape)
```

### Célula 4 — Zero-Padding para 32x32 (LeNet-5)
```python
x_treino = np.pad(x_treino, ((0,0), (2,2), (2,2), (0,0)), 'constant')
x_validacao = np.pad(x_validacao, ((0,0), (2,2), (2,2), (0,0)), 'constant')
x_teste = np.pad(x_teste, ((0,0), (2,2), (2,2), (0,0)), 'constant')

print("Treino:", x_treino.shape)
print("Validação:", x_validacao.shape)
print("Teste:", x_teste.shape)
```

### Célula 5 — Normalização dos Pixels [0, 1]
```python
normalizar_dados = lambda t: t / 255.0

x_treino = normalizar_dados(x_treino)
x_validacao = normalizar_dados(x_validacao)
x_teste = normalizar_dados(x_teste)

print("Status final pronto para a rede LeNet-5:")
print(f"Treino:    {x_treino.shape} -> Valores entre [{x_treino.min():.2f}, {x_treino.max():.2f}]")
print(f"Validação: {x_validacao.shape} -> Valores entre [{x_validacao.min():.2f}, {x_validacao.max():.2f}]")
print(f"Teste:     {x_teste.shape} -> Valores entre [{x_teste.min():.2f}, {x_teste.max():.2f}]")
```

---

## ❓ 5. Resolução de Dúvidas Frequentes

| Problema | Causa | Solução |
| :--- | :--- | :--- |
| `import: The term 'import' is not recognized` | Digitar comando Python diretamente no PowerShell. | Execute scripts com `python nome_do_arquivo.py`. |
| `NameError: name 'x_treino' is not defined` | Executar um script que depende de variáveis de outro arquivo sem tê-las carregado. | Utilize o script completo do módulo (ex: `data_split.py`, `data_padding.py`, `data_normalization.py` ou `executar.py`). |
| O terminal parece "travado" após a Etapa 2 | A janela gráfica do Matplotlib foi aberta e está aguardando interação. | Feche a janela da imagem para o script prosseguir. |

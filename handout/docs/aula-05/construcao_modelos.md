# Construindo um MLP com TensorFlow/Keras

Na aula passada, você implementou partes centrais de uma rede neural manualmente para entender o mecanismo. Isso era necessário.

Agora precisamos sair do modo "entendi a conta" e entrar no modo "consigo construir um modelo sem desperdiçar tempo com infraestrutura".

É exatamente para isso que existe o **TensorFlow/Keras**.

---

## O que é TensorFlow/Keras?

O **TensorFlow** é uma biblioteca de computação numérica voltada para aprendizado de máquina. Ela lida com o trabalho pesado: operações vetorizadas, cálculo automático de gradientes, execução em GPU e treinamento de modelos.

O **Keras** é a API de alto nível usada para definir modelos de forma mais legível. Em vez de implementar manualmente forward propagation, backpropagation e atualização de pesos, você descreve a arquitetura e deixa o framework cuidar do resto.

!!! note "Não confunda as camadas da abstração"
    - **TensorFlow**: faz a computação
    - **Keras**: organiza a construção do modelo

Em outras palavras: o Keras é a interface que você usa; o TensorFlow é o motor por baixo.

---

!!! NOTE "Vídeos"
    Antes de começar, assista os seguintes vídeos do curso de Machine Learning do Andrew Ng.

    - [Construindo Modelos com Tensorflow/Keras](https://www.youtube.com/watch?v=2XX4GUOGGKs&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=2) (Vídeos 7 a 9)
    - [Parâmetros da Rede](https://www.youtube.com/watch?v=zJp2z4n4L5w&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=12) (Vídeos 17 a 20)
    - [Problemas Multiclasse](https://www.youtube.com/watch?v=bxqUGKgUztM&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=17) (Vídeos 22 a 25)
    - [Avaliando um modelo](https://www.youtube.com/watch?v=56BloCH2JAU&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=26) (Vídeos 33 e 34)

## O Modelo Completo

Ao invés de construir passo a passo, vamos destrinchar um código completo. Abaixo está um exemplo de MLP para classificação multiclasse com imagens `32x32`.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Flatten(input_shape=(32, 32)),
    Dense(1024),
    BatchNormalization(),
    Activation('relu'),

    Dense(512),
    BatchNormalization(),
    Activation('relu'),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),

    Dense(64),
    BatchNormalization(),
    Activation('relu'),

    Dense(10, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(X_train_normalized, y_train, epochs=20)

_, train_accuracy = model.evaluate(X_train_normalized, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)

print(f"Acurácia no conjunto de treino: {train_accuracy * 100:.2f}%")
print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")
```

!!! warning "Atenção ao `input_shape`"
    Se a imagem tiver apenas um canal, o formato pode ser `input_shape=(32, 32)`.
    Se for uma imagem RGB, o formato correto é `input_shape=(32, 32, 3)`.

---

## Definindo o Modelo com `Sequential`

O `Sequential` é a forma mais simples de criar uma rede neural no Keras. Você passa uma lista ordenada de camadas e o modelo entende que a saída de uma vira a entrada da próxima.

```python
model = Sequential([
    Flatten(input_shape=(32, 32)),

    Dense(1024),
    BatchNormalization(),
    Activation('relu'),

    Dense(10, activation='softmax'),
])
```

Essa abordagem funciona bem quando a arquitetura é literalmente sequencial, sem desvios, sem múltiplas entradas e sem múltiplas saídas. Para um MLP clássico, é exatamente o que queremos.

### Tipos de camadas usados aqui

### `Flatten`

```python
Flatten(input_shape=(32, 32))
```

O `Flatten` transforma uma estrutura multidimensional em um vetor 1D. Uma imagem `32x32`, por exemplo, vira um vetor com `1024` valores.

Sem isso, uma camada `Dense` não sabe como receber a imagem diretamente.

### `Dense`

```python
Dense(1024)
```

`Dense` é a camada totalmente conectada.
Cada neurônio recebe todas as ativações da camada anterior.

O principal parâmetro é:

- `units`: número de neurônios da camada

Você também pode passar a ativação direto:

```python
Dense(10, activation='softmax')
```

Ou separar a ativação em outra camada, como no caso abaixo:

```python
Dense(256)
BatchNormalization()
Activation('relu')
```

### `BatchNormalization`

```python
BatchNormalization()
```

Essa camada normaliza as ativações intermediárias do modelo durante o treinamento. Na prática, ela ajuda a estabilizar a distribuição dos valores que passam pela rede, o que normalmente melhora a otimização.

Não é mágica. Ela não conserta um modelo ruim. Mas costuma deixar o treino menos instável.

### `Activation`

```python
Activation('relu')
```

Aqui a ativação foi separada da camada `Dense`.
Isso deixa explícita a ordem:

1. transformação linear
2. normalização
3. não-linearidade

No exemplo, usamos:

- `relu` nas camadas ocultas
- `softmax` na camada de saída

!!! tip "Por que `softmax` no fim?"
    Porque estamos em classificação multiclasse.
    A `softmax` transforma os scores finais em probabilidades que somam 1.

---

## `model.compile`

Definir as camadas não basta. O modelo ainda não sabe **como aprender**.

É o `compile` que configura:

- qual algoritmo de otimização será usado
- qual função de perda será minimizada
- quais métricas serão exibidas

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

### `optimizer`

```python
optimizer=Adam(learning_rate=0.001)
```

O otimizador decide como os pesos serão atualizados a partir do gradiente.

O **Adam** é uma escolha padrão muito forte porque combina:

- adaptação automática da taxa de aprendizado por parâmetro
- convergência geralmente mais rápida do que gradient descent puro

O parâmetro principal aqui é:

- `learning_rate`: tamanho do passo dado a cada atualização

Se esse valor for alto demais, o treino oscila ou diverge.
Se for baixo demais, o modelo aprende em câmera lenta.

### `loss`

```python
loss='sparse_categorical_crossentropy'
```

A `loss` é a função que o modelo tenta minimizar.

Nesse caso, usamos `sparse_categorical_crossentropy`, que é apropriada quando:

- o problema é **multiclasse**
- os rótulos são inteiros, como `0, 1, 2, ..., 9`

!!! danger "Erro comum"
    Não misture formato do target com a loss.

    - `sparse_categorical_crossentropy`: labels inteiros
    - `categorical_crossentropy`: labels em one-hot encoding

Se você errar esse pareamento, o treino até pode rodar, mas o resultado pode ficar errado ou inconsistente.

### `metrics`

```python
metrics=['accuracy']
```

As métricas servem para monitorar desempenho.
Elas não definem o gradiente do treino; quem faz isso é a `loss`.

Aqui usamos `accuracy`, que responde uma pergunta simples:

> de todas as amostras, quantas foram classificadas corretamente?

---

## `model.fit`

Depois de construir e compilar o modelo, vem o treinamento.

```python
model.fit(X_train_normalized, y_train, epochs=20)
```

Esse comando faz o ciclo completo:

1. forward propagation
2. cálculo da loss
3. backpropagation
4. atualização dos pesos

Isso se repete ao longo das épocas.

### Parâmetros mais importantes

```python
history = model.fit(
    X_train_normalized,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)
```

- `x`: features de treino
- `y`: rótulos de treino
- `epochs`: quantas vezes o modelo verá o conjunto de treino inteiro
- `batch_size`: quantas amostras são processadas antes de atualizar os pesos
- `validation_split`: fração do treino separada para validação
- `verbose`: nível de detalhe mostrado no terminal

O retorno do `fit` normalmente é armazenado em `history`.
Isso é útil porque o histórico guarda métricas por época, o que ajuda a detectar overfitting depois.

!!! info "Validação não é perfumaria"
    Se você só olha métrica de treino, você não está avaliando generalização.
    Você está avaliando o quão bem o modelo decorou os dados que já viu.

---

## `model.evaluate`

Treinar não é o mesmo que medir qualidade.

Para avaliar o modelo em um conjunto específico, usamos `evaluate`.

```python
_, train_accuracy = model.evaluate(X_train_normalized, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
```

O método retorna:

1. o valor da loss
2. os valores das métricas configuradas no `compile`

No exemplo:

- a primeira chamada mede desempenho no treino
- a segunda mede desempenho no teste

E aí sim conseguimos comparar:

```python
print(f"Acurácia no conjunto de treino: {train_accuracy * 100:.2f}%")
print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")
```

Se a acurácia de treino estiver muito acima da acurácia de teste, isso é um alerta.
Vamos entrar nisso na próxima seção.

---

## Resumo do Fluxo

Em Keras, a rotina básica de construção de um modelo segue sempre a mesma lógica:

1. definir arquitetura
2. compilar
3. treinar
4. avaliar

Em código:

```python
model = Sequential([...])
model.compile(...)
model.fit(...)
model.evaluate(...)
```

Simples na interface.
Menos simples no que acontece por baixo.
Mas a grande vantagem é essa: você já entendeu a teoria na aula anterior, então agora pode usar o framework sem virar apertador de botão.

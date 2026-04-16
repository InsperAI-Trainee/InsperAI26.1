# Overfitting e Underfitting

Construir um modelo é trivial perto de construir um modelo que generaliza.

Você pode montar uma arquitetura bonita, treinar por horas e terminar com um modelo inútil.
Os dois erros clássicos são:

- **underfitting**: o modelo é fraco demais para aprender o padrão
- **overfitting**: o modelo aprende demais o treino e mal generaliza

---

## Underfitting

O **underfitting** acontece quando o modelo não consegue capturar a estrutura dos dados.

Em linguagem simples:

- ele erra no treino
- ele erra na validação
- ele erra no teste

Ou seja: não é um problema de generalização. É um problema de capacidade ou de treinamento insuficiente.

### Como detectar underfitting

Os sinais mais comuns são:

- loss de treino alta e ainda ruim depois de várias épocas
- acurácia de treino baixa
- acurácia de validação também baixa
- curvas de treino e validação ruins ao mesmo tempo

```python
history = model.fit(
    X_train_normalized,
    y_train,
    epochs=10,
    validation_split=0.2,
    verbose=0,
)

print(history.history['accuracy'][-1])
print(history.history['val_accuracy'][-1])
```

Se ambos os valores continuarem baixos, o modelo provavelmente não está conseguindo aprender o suficiente.

### Como mitigar underfitting

#### 1. Aumentar a complexidade do modelo

Se o modelo é simples demais, ele não vai representar padrões complexos.

```python
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Às vezes isso é pouco.

```python
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512),
    BatchNormalization(),
    Activation('relu'),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),

    Dense(10, activation='softmax'),
])
```

Mais camadas e mais neurônios aumentam a capacidade do modelo.

!!! warning "Sem caricatura"
    Mais complexo não significa automaticamente melhor.
    Significa apenas que o modelo tem mais capacidade.
    Se usar isso sem critério, você troca underfitting por overfitting.

#### 2. Treinar melhor

Às vezes o modelo não é fraco; ele só foi pouco treinado.

Você pode ajustar parâmetros como:

- mais épocas
- learning rate mais adequado
- batch size melhor calibrado

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    X_train_normalized,
    y_train,
    epochs=40,
    batch_size=128,
)
```

#### 3. Obter mais dados

Mais dados ajudam quando o conjunto atual é pequeno ou pouco representativo.

Mas seja honesto: **mais dados não salvam um modelo claramente incapaz**.
Se a arquitetura é simplória demais, você só vai errar em escala maior.

---

## Overfitting

O **overfitting** acontece quando o modelo se ajusta demais ao conjunto de treino.

Ele aprende padrões reais, sim, mas também aprende ruído, coincidências e peculiaridades que não se repetem fora daquele conjunto.

O resultado clássico é este:

- treino muito bom
- validação pior
- teste pior ainda

### Como detectar overfitting

O sinal mais conhecido é a diferença grande entre treino e validação.

```python
_, train_accuracy = model.evaluate(X_train_normalized, y_train, verbose=0)
_, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)

print(f"Treino: {train_accuracy:.4f}")
print(f"Teste: {test_accuracy:.4f}")
```

Se o treino estiver alto demais e o teste cair bastante, desconfie.

Outra forma melhor de enxergar isso é observar o histórico do treinamento:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='treino')
plt.plot(history.history['val_loss'], label='validação')
plt.legend()
plt.show()
```

Se a `loss` de treino continua caindo, mas a `val_loss` começa a subir, o modelo está memorizando em vez de generalizar.

!!! danger "Erro comum"
    Acurácia alta no treino não prova que o modelo é bom.
    Prova apenas que ele ficou bom em acertar aquilo que já viu.

---

## Como mitigar overfitting

Não existe bala de prata.
O que existe é um conjunto de técnicas que reduzem a tendência do modelo a memorizar ruído.

### Regularização L2

A regularização L2 pune pesos muito altos.

Ela adiciona um termo extra à loss, forçando o modelo a evitar soluções exageradamente sensíveis.

```python
from tensorflow.keras import regularizers

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax'),
])
```

Quando usar?

- quando o modelo está flexível demais
- quando os pesos começam a crescer de forma exagerada

### Dropout

O `Dropout` desliga aleatoriamente parte dos neurônios durante o treino.

Isso impede que a rede dependa demais de caminhos específicos.

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(10, activation='softmax'),
])
```

O principal parâmetro é:

- `rate`: fração de neurônios desligados temporariamente

Se exagerar no dropout, o modelo pode perder capacidade demais e voltar para underfitting.

### Normalização dos dados

Antes do treino, os dados precisam estar em uma escala razoável.

Para imagens, isso normalmente significa trazer os pixels para o intervalo `[0, 1]`.

```python
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0
```

Isso não é a mesma coisa que `BatchNormalization`.

### Batch Normalization

`BatchNormalization` atua **dentro** da rede, sobre ativações intermediárias.

```python
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax'),
])
```

Ela costuma:

- estabilizar o treino
- permitir taxas de aprendizado mais razoáveis
- reduzir sensibilidade à inicialização

Mas vale repetir: `BatchNormalization` ajuda bastante, porém não substitui validação séria, regularização e dados decentes.

### Ajustando parâmetros de treino

Muitas vezes o overfitting vem do processo de treino, não só da arquitetura.

Exemplos:

- épocas demais
- learning rate inadequado
- ausência de validação
- continuar treinando depois do ponto ideal

Uma técnica simples e útil é **Early Stopping**:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train_normalized,
    y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
)
```

Aqui, o treino para quando a validação deixa de melhorar por algumas épocas.

Isso é melhor do que insistir em 100 épocas só porque o número parece sério.

### Obtenção de dados

Se o modelo está decorando o treino, uma das melhores saídas é aumentar a diversidade dos dados.

Isso pode acontecer de duas formas:

- coletar mais exemplos reais
- gerar variações plausíveis dos exemplos existentes

No caso de imagens, uma abordagem comum é **data augmentation**:

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])
```

Isso não cria informação do nada, então não trate augmentation como milagre.
Mas, quando bem usado, ajuda bastante a reduzir memorização superficial.

---

## Normalização vs Batch Normalization

Essa confusão aparece o tempo todo, então vale separar de forma explícita:

| Técnica | Onde atua | Exemplo |
|---|---|---|
| **Normalização dos dados** | Antes da rede | `X = X / 255.0` |
| **Batch Normalization** | Dentro da rede | `BatchNormalization()` |

Uma prepara a entrada.
A outra estabiliza ativações internas.
Não são equivalentes.

---

## Resumo Honesto

Se o modelo está ruim, existem duas perguntas que você precisa fazer:

1. ele é fraco demais para aprender?
2. ele está decorando em vez de generalizar?

Se a resposta para a primeira for "sim", você está em **underfitting**.
Se a resposta para a segunda for "sim", você está em **overfitting**.

O ponto não é decorar o nome do problema.
O ponto é aprender a olhar para as métricas e tomar a decisão certa:

- aumentar capacidade quando falta capacidade
- regularizar quando sobra flexibilidade
- coletar dados melhores quando o dataset é ruim

Esse diagnóstico vale muito mais do que ficar trocando hiperparâmetro no escuro.

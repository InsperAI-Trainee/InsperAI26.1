# Melhorando o Modelo

Montar um modelo que roda é o começo, não o fim.

O primeiro MLP que você treina normalmente sofre de um ou mais problemas bem previsíveis:

- aprende devagar demais
- oscila durante o treino
- memoriza o conjunto de treino
- fica sensível demais a escolhas ruins de hiperparâmetros

Nesta página, o foco é pragmático: como sair de um modelo que apenas existe para um modelo que treina melhor e generaliza melhor.

---

## Normalizando os Dados de Treino

Antes de discutir arquitetura ou regularização, comece pelo básico.

Se as entradas chegam em escalas ruins, o treinamento fica mais difícil do que precisava ser. O otimizador precisa lidar com gradientes desequilibrados, algumas features dominam outras e a convergência tende a piorar.

Para imagens, o caso mais comum é simples:

```python
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0
```

Aqui, os pixels deixam de estar no intervalo `[0, 255]` e passam para `[0, 1]`.

Para dados tabulares, a lógica costuma ser outra: centralizar e escalar cada feature.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

!!! danger "Erro conceitual comum"
    O `scaler` deve ser ajustado no treino e reaplicado nos outros conjuntos.
    Se você faz `fit` separadamente em validação ou teste, contaminou a avaliação.

### Por que isso ajuda?

Quando as entradas têm escalas mais consistentes:

- o treino fica numericamente mais estável
- o learning rate fica menos sensível
- a rede aprende mais rápido
- comparar o efeito dos hiperparâmetros deixa de ser loteria

Isso não substitui `BatchNormalization`.
São coisas diferentes:

- **normalização dos dados**: atua antes da rede
- **BatchNormalization**: atua dentro da rede

---

## Ajustando Parâmetros de Treino

Depois que os dados estão em ordem, os dois hiperparâmetros que mais mudam o comportamento do treino são:

- `learning_rate`
- `batch_size`

Se você errar feio nesses dois, o resto da arquitetura quase não importa.

### `learning_rate`

O `learning_rate` define o tamanho do passo dado pelo otimizador a cada atualização.

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

Na prática:

- **alto demais**: o treino oscila, não converge ou até diverge
- **baixo demais**: o modelo melhora em câmera lenta e pode parar longe do melhor ponto

Um ajuste pequeno nesse valor pode mudar bastante o resultado:

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

!!! tip "Regra prática"
    Se a loss parece caótica ou não desce, suspeite do learning rate antes de sair empilhando camada.

### `batch_size`

O `batch_size` define quantas amostras entram em cada atualização de gradiente.

```python
history = model.fit(
    X_train_normalized,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
)
```

Quando o batch_size não é o dataset inteiro, o batch de treinamento é chamado de **mini-batch**. 

!!! Note "Gradient Descent"
    O nome do algoritmo de Gradient Descent usado para treinamento em mini-batches é chamado **Mini-Batch Gradient Descent.** Caso o batch seja o dataset inteiro, o algoritmo vira o **Batch Gradient Descent**.

O efeito mais comum é este:

- **batch pequeno**: treino mais ruidoso, mais atualizações, menos memória, às vezes melhor generalização
- **batch grande**: treino mais estável por passo, menos atualizações por época, mais uso de memória

Nenhum deles é "o certo" por definição.
Depende do problema, da arquitetura e do hardware.

!!! warning "Relação com `BatchNormalization`"
    Se o batch é pequeno demais, `BatchNormalization` tende a piorar porque as estatísticas do lote ficam ruidosas.

### Ordem sensata de ajuste

Se você tentar ajustar tudo ao mesmo tempo, só vai se confundir.

Uma ordem mais limpa é:

1. normalizar os dados
2. escolher uma arquitetura base
3. ajustar `learning_rate`
4. ajustar `batch_size`
5. só depois partir para técnicas adicionais

---

## Batch Normalization
Normalizar o batch significa olhar para as ativações produzidas por uma camada **dentro de um mini-batch específico** e colocá-las em uma escala mais controlada.

Agora sim faz sentido falar de `BatchNormalization`: depois que os dados já entram bem escalados e o treino básico já existe.

```python
Dense(256)
BatchNormalization()
Activation('relu')
```

No arranjo mais comum, a camada entra **depois da transformação linear** e **antes da ativação**.

### Como ela normaliza as ativações?


Imagine que uma camada `Dense(256)` produziu, para um batch de `128` amostras, uma matriz de ativações com formato:

```python
(128, 256)
```

Isso significa:

- `128` linhas: uma para cada amostra do batch
- `256` colunas: uma para cada neurônio da camada

O `BatchNormalization` normaliza **coluna por coluna**. Para cada um dos 256 neurônios, ele olha os 128 valores produzidos naquele batch, calcula a média e a variância desses valores, e então transforma essa coluna para ficar aproximadamente com:

- média próxima de `0`
- variância próxima de `1`

Em outras palavras: ele não normaliza uma única amostra isolada.
Ele usa o comportamento do **lote inteiro** para estabilizar as ativações que serão passadas para a próxima etapa da rede.

Matematicamente, para cada mini-batch, a camada calcula a **média** e a **variância** das ativações daquela dimensão. Depois, centraliza e reescala os valores:

$$
\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Só que ela não para aí. Depois da normalização, a camada ainda aprende dois parâmetros:

$$
y = \gamma \hat{z} + \beta
$$

Onde:

- $\mu_B$: média do mini-batch
- $\sigma_B^2$: variância do mini-batch
- $\epsilon$: termo pequeno para evitar divisão por zero
- $\gamma$: escala aprendida
- $\beta$: deslocamento aprendido

Ou seja: a camada padroniza as ativações do lote e depois aprende qual escala final ainda faz sentido para a rede.

### O que isso melhora?

Em muitos casos, `BatchNormalization` ajuda a:

- estabilizar o treino
- reduzir sensibilidade à inicialização
- permitir learning rates um pouco mais agressivos
- evitar que algumas ativações explodam ou colapsem

Durante o treino, ela usa estatísticas do mini-batch atual.
Durante a inferência, usa médias móveis acumuladas ao longo do treinamento.

### Quando não usar cegamente

`BatchNormalization` não é uma camada universalmente correta.

Casos em que você deve pelo menos questionar o uso:

- **batch muito pequeno**
- **camada de saída**
- **modelo já simples e estável**
- **cenários em que a dependência do batch atrapalha**

Se o ganho real não aparece, remova e compare.
Colocar por reflexo é só ruído arquitetural.

---

## Overfitting

### O que é Overfitting

Depois que o treino ficou estável, aparece outro problema: o modelo pode ficar bom demais no treino e piorar fora dele.

Esse é o cenário clássico de **overfitting**:

- treino muito bom
- validação pior
- teste pior

O que queremos aqui não é um modelo "impressionante" no treino.
É um modelo que generaliza.

!!! note "Intuição rápida"
    A intuição correta não é "o modelo aprendeu demais" no sentido de ter ficado inteligente demais.
    É quase o contrário: ele ficou **específico demais** para o conjunto de treino.

Pense assim: imagine que você quer aprender a reconhecer cachorros.
O objetivo certo é aprender padrões gerais, como formato do focinho, tipo de orelha, textura do pelo e proporções do corpo.

Mas um modelo em overfitting começa a aprender detalhes que não deveriam importar tanto:

- a iluminação específica das fotos de treino
- o fundo mais frequente
- um ruído da câmera
- pequenas coincidências estatísticas daquele conjunto

Ou seja: em vez de aprender **a classe**, ele aprende **o jeito exato como aquela base apareceu**.

Outra forma de enxergar isso é pela **fronteira de decisão**.

Um modelo que generaliza bem encontra uma separação que captura a estrutura principal dos dados.
Um modelo em overfitting força a fronteira para passar o mais perto possível de cada ponto de treino, inclusive dos pontos estranhos, ruidosos ou pouco representativos.

![Underfitting x Overfitting](https://media.licdn.com/dms/image/v2/D4D12AQHcGf9AlSO4lQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1669390461210?e=2147483647&v=beta&t=O17nqR96ymYb8hMFBMczZ6il6_7eVOSabPVc45s5hM0)

O resultado é perverso:

- no treino, ele parece excelente
- fora do treino, ele fica frágil

Porque dados novos nunca vêm exatamente com as mesmas peculiaridades do conjunto de treino.

!!! note "Resumo da intuição"
    Overfitting é quando o modelo confunde **padrão** com **acidente**.
    Ele não aprende só o que se repete de forma útil; aprende também o que era ruído, coincidência ou detalhe irrelevante.

## Evitando Overfitting

Existem várias formas de reduzir essa tendência. As mais úteis aqui são controlar a magnitude dos pesos, reduzir coadaptação excessiva entre neurônios e parar o treino antes que ele comece a memorizar demais.

### Regularização L2

A regularização L2 pune **pesos excessivamente altos**.

A intuição é simples: quando a rede precisa de **pesos muito grandes** para acertar o treino, isso costuma ser sinal de que ela está montando uma regra **sensível demais**, **rígida demais** e **específica demais** para aquele conjunto. Em vez de aprender uma **separação mais estável**, ela começa a depender de **ajustes agressivos** que funcionam muito bem no treino, mas tendem a quebrar quando os dados mudam um pouco.

Na prática, em vez de minimizar apenas a loss original do problema, o modelo passa a minimizar uma versão **regularizada** da função de custo:

$$
\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_i w_i^2
$$

Onde:

- $\mathcal{L}$: loss original
- $\lambda$: força da regularização
- $w_i$: pesos do modelo

Em redes neurais, essa ideia aparece de forma mais completa como a soma dos quadrados dos pesos das camadas:

$$
\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_{l} \sum_{i,j} \left(W^{[l]}_{ij}\right)^2
$$

O efeito prático é simples: pesos muito grandes deixam a loss artificialmente mais cara.
Então o treinamento passa a preferir soluções com pesos menores e menos agressivos.

Isso importa porque pesos grandes demais costumam deixar a rede sensível demais a pequenas variações da entrada.
Quando isso acontece, o modelo encaixa o treino com força excessiva e tende a generalizar pior.

```python
from tensorflow.keras import regularizers

model = Sequential([
    Flatten(input_shape=(32, 32)),
    Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
    ),
    Dense(10, activation='softmax'),
])
```

No Keras, `kernel_regularizer=regularizers.l2(0.001)` aplica essa penalização aos pesos da camada.

!!! note "O que muda no treinamento?"
    A regularização L2 altera o gradiente.
    O modelo continua tentando reduzir o erro de previsão, mas agora também sofre pressão para não deixar os pesos crescerem demais.

Na prática, isso força a rede a preferir soluções menos extremas.
O modelo continua aprendendo, mas com menos liberdade para memorizar ruído do treino.

Quando usar?

- quando o modelo está flexível demais
- quando o treino melhora muito mais do que a validação

### Dropout

O `Dropout` desliga aleatoriamente parte dos neurônios durante o treino.

A intuição é a seguinte: sem `Dropout`, a rede pode criar dependências muito específicas entre neurônios. Alguns neurônios começam a "confiar demais" em outros, formando caminhos internos que funcionam muito bem para o conjunto de treino, mas que podem ser frágeis fora dele.

Com `Dropout`, a cada batch, parte dos neurônios fica temporariamente inativa. Isso força a rede a não depender de um único caminho específico para acertar. Ela precisa distribuir melhor a informação entre vários neurônios e aprender representações mais robustas.

Pense como um grupo de alunos resolvendo exercícios. Se sempre a mesma pessoa responde tudo, o grupo parece bom, mas só porque depende dela. Se essa pessoa some aleatoriamente em alguns exercícios, os outros precisam aprender também. O `Dropout` faz algo parecido com a rede.

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Flatten(input_shape=(32, 32)),
    Dense(512, activation='relu'),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(10, activation='softmax'),
])
```

Isso reduz a dependência da rede em caminhos muito específicos e dificulta memorização superficial.

O principal parâmetro é:

- `rate`: fração de neurônios temporariamente desligados

Se exagerar, o modelo perde capacidade demais e começa a sofrer de underfitting.

### Early Stopping

Outra forma bem prática de reduzir overfitting é simplesmente **parar o treino na hora certa**.

Muitas vezes, depois de certo ponto, o que acontece é isto:

- a loss de treino continua caindo
- a validação para de melhorar
- o modelo começa a memorizar o conjunto de treino

O `EarlyStopping` serve exatamente para isso: monitorar uma métrica de validação e interromper o treinamento quando ela deixa de melhorar.

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
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
)
```

Os parâmetros mais importantes aqui são:

- `monitor`: qual métrica será acompanhada
- `patience`: quantas épocas sem melhora o treino tolera antes de parar
- `restore_best_weights`: faz o modelo voltar para os pesos da melhor época observada

Na prática, isso evita insistir em épocas extras que só serviriam para piorar a generalização.

!!! warning "Não use regularização no escuro"
    Se o modelo já está fraco ou subtreinado, jogar `Dropout`, L2 ou `EarlyStopping` por cima não corrige a causa real.
    Essas técnicas ajudam a conter excesso de flexibilidade ou excesso de treino, não a salvar um modelo mal formulado.

---

## Resumo Prático

Se o modelo base já roda, a sequência mais racional para melhorar costuma ser:

1. normalizar corretamente os dados
2. ajustar `learning_rate`
3. ajustar `batch_size`
4. usar `BatchNormalization` se o treino estiver instável
5. adicionar L2, `Dropout` ou `EarlyStopping` se aparecer overfitting

O erro mais comum aqui é sair mexendo em tudo ao mesmo tempo.
Fazer isso é chute, não tuning!

---

## Modelo aprimorado

Agora juntando as técnicas anteriores, podemos partir do modelo original da página 1 e aplicar melhorias de forma coerente:

- normalização dos dados de entrada
- ajuste explícito de `learning_rate` e `batch_size`
- `BatchNormalization` entre `Dense` e `Activation`
- regularização L2 nas camadas ocultas
- `Dropout` depois das ativações
- `EarlyStopping` monitorando a validação

```python
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam

X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

model = Sequential([
    Flatten(input_shape=(32, 32)),

    Dense(1024, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(256, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    Dense(64, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),

    Dense(10, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    X_train_normalized,
    y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
)

_, train_accuracy = model.evaluate(X_train_normalized, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)

print(f"Acurácia no conjunto de treino: {train_accuracy * 100:.2f}%")
print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")
```

!!! warning "Não copie como receita universal"
    Esse modelo é uma versão coerente do modelo base com as técnicas da aula, não uma arquitetura garantidamente ótima.
    Se o modelo começar a sofrer underfitting, reduza regularização, reduza `Dropout` ou simplifique as intervenções.

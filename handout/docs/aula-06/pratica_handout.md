# Prática

Clone o repositório da prática no link:

- [Link do Classroom](TODO)  <!-- TODO: colar link do Classroom quando disponível -->

A atividade deve ser entregue até dia **TODO/TODO às 23h59**.  <!-- TODO: preencher data de entrega -->

!!! NOTE "Antes de começar"
    Se você chegou aqui sem ter lido as três páginas conceituais anteriores, volte. A prática assume que você já consegue:

    - Explicar por que um MLP é subótimo para imagens
    - Descrever o que um kernel faz e como stride/padding afetam a saída
    - Ler a arquitetura `Conv → ReLU → Pool → Flatten → Dense → Softmax` sem se perder

---

## O problema

Nesta prática, você vai treinar uma CNN para **classificar imagens de mãos em dígitos de 0 a 5** — uma tarefa de classificação multiclasse com 6 classes.

![Exemplos do Signs Dataset](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/FingerSpellingAbc.svg/640px-FingerSpellingAbc.svg.png)

O dataset é o **Signs Dataset**, usado originalmente no curso [*Convolutional Neural Networks*](https://www.coursera.org/learn/convolutional-neural-networks) da **Deep Learning Specialization** (Andrew Ng / DeepLearning.AI / Coursera).

### Características do dataset

| Propriedade | Valor |
|---|---|
| Dimensão das imagens | **64 × 64 × 3** (RGB) |
| Exemplos de treino | **1 080** |
| Exemplos de teste | **120** |
| Classes | **6** (dígitos 0, 1, 2, 3, 4, 5) |

É um dataset pequeno por padrões modernos, o que é ótimo para a prática: o modelo treina rápido, cabe facilmente na memória e dá margem para você experimentar arquiteturas diferentes sem esperar horas.

!!! warning "Atenção ao formato"
    As imagens chegam como tensores NumPy de 0 a 255. Antes de alimentar a CNN, **normalize** dividindo por 255 para que fiquem na faixa $[0, 1]$. Se esquecer desse passo, o treino vai parecer errado — os gradientes ficam enormes logo na primeira camada.

!!! info "Por que multiclasse e não binária?"
    Você já viu classificação binária na aula 2 (sigmoide + `binary_crossentropy`) e classificação multiclasse na aula 4 (softmax + `categorical_crossentropy`). Aqui usamos **multiclasse**: 6 neurônios de saída com `softmax`, rótulos em formato *one-hot* e loss `categorical_crossentropy`.

---

## O Que Você Vai Fazer

1. **Carregar o Signs Dataset** e visualizar algumas imagens de cada uma das 6 classes para ganhar intuição sobre o que a rede vai ver.
2. **Normalizar** as imagens (dividir por 255) e converter os rótulos para **one-hot encoding** com 6 classes.
3. **Construir uma CNN** com a API `Sequential` do Keras, usando `Conv2D`, `MaxPooling2D`, `Flatten` e `Dense`, fechando com uma `Dense(6, activation='softmax')`.
4. **Compilar e treinar** o modelo com otimizador Adam e `categorical_crossentropy`, monitorando loss e accuracy no treino e na validação.
5. **Avaliar** no conjunto de teste e **inspecionar alguns erros** — qual classe foi mais confundida? Isso diz algo sobre como a rede enxerga os sinais?

---

## Dicas finais

- A arquitetura que vimos em [Arquitetura de uma CNN](arquitetura_cnn.md) é um excelente ponto de partida. Não precisa ser criativo: copie aquele modelo e veja até onde ele chega.
- Se a acurácia no treino sobe rápido mas a de validação trava, você está **overfitando**. É esperado num dataset pequeno. Tente diminuir o número de filtros ou adicionar uma camada `Dropout` depois da densa.
- Poucas épocas já devem te dar acurácia alta. CNN + dataset pequeno = treino rápido.
- `model.summary()` antes de treinar para conferir se as formas das camadas batem com o que você esperava.

!!! tip "Quando o modelo começar a funcionar"
    Separe 5 minutos para **olhar as predições erradas**. Quais dígitos o modelo confunde entre si? A confusão é "racional" (ex.: 1 e 2, que têm silhueta parecida) ou aleatória? Esse tipo de análise é o que separa um projeto de ML acadêmico de um que, eventualmente, vai para produção.

---

!!! Author
    - **Vitor Salomão**

<div class="handout-nav" markdown>
[← Anterior: Arquitetura de uma CNN](arquitetura_cnn.md){ .md-button }
[Próxima: Aula 7 — ResNet e Transfer Learning →](../aula-07/index.md){ .md-button .md-button--primary }
</div>

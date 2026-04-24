# Prática

Clone o repositório da prática no link:

- [Link do Classroom](TODO)  <!-- TODO: colar link do Classroom quando disponível -->

A atividade deve ser entregue até dia **TODO/TODO às 23h59**.  <!-- TODO: preencher data de entrega -->

!!! NOTE "Antes de começar"
    Esta prática assume que você já leu as duas páginas conceituais desta aula. Em particular, você precisa estar confortável com:

    - A ideia de **skip connection** e por que arquiteturas modernas (MobileNetV2 incluso) a usam
    - A diferença entre **feature extraction** e **fine-tuning**
    - O papel do `preprocess_input` e por que **não** se divide por 255 quando se usa MobileNetV2

---

## O problema

Nesta prática, você vai treinar um classificador binário para responder uma pergunta simples:

> **Essa imagem contém uma alpaca?**

A resposta sai como uma probabilidade (entre 0 e 1). Acima de 0.5 → "alpaca"; abaixo → "não alpaca". É uma **classificação binária** clássica, mas com uma diferença importante em relação ao que você fez até aqui: você **não vai treinar uma CNN do zero**. Vai usar **MobileNetV2 pré-treinada em ImageNet** como base de features, adicionando apenas um head próprio.

O dataset é o **Alpaca dataset**, do curso [*Convolutional Neural Networks*](https://www.coursera.org/learn/convolutional-neural-networks) da Deep Learning Specialization (Andrew Ng / DeepLearning.AI / Coursera), onde é usado exatamente para demonstrar transfer learning com MobileNetV2.

### Organização do dataset

O dataset chega em **duas pastas**, uma por classe:

```
dataset/
├── alpaca/
│   ├── img_0.jpg
│   ├── img_1.jpg
│   └── ...
└── not_alpaca/
    ├── img_0.jpg
    ├── img_1.jpg
    └── ...
```

Esse formato é exatamente o que a função `tf.keras.utils.image_dataset_from_directory` espera — ela infere as classes pelo nome das subpastas e devolve um `tf.data.Dataset` pronto para treino.

!!! tip "Use `image_dataset_from_directory` de ponta a ponta"
    Essa função já te dá splits de treino e validação automaticamente (via `validation_split` e `subset`), redimensiona as imagens para o tamanho que você pedir (use **(160, 160)** ou **(224, 224)** — tamanhos padrão para MobileNetV2), e batcha os dados. Não reinvente a roda carregando imagem a imagem com PIL.

---

## O Que Você Vai Fazer

1. **Carregar o dataset** com `image_dataset_from_directory()`, dividindo em treino e validação; visualizar algumas amostras de cada classe.
2. **Aplicar `preprocess_input` do MobileNetV2** (escala para [-1, 1]) e, opcionalmente, algumas camadas de **data augmentation** (`RandomFlip`, `RandomRotation`) dentro do modelo para compensar o dataset pequeno.
3. **Construir o modelo**:
    - `MobileNetV2(include_top=False, weights='imagenet')` como base
    - `base_model.trainable = False` (congelar tudo)
    - `GlobalAveragePooling2D` + `Dense(1, activation='sigmoid')` no topo
4. **Compilar** com otimizador Adam, loss `binary_crossentropy`, métrica `accuracy` — e treinar por algumas épocas (**feature extraction** pura).
5. **Avaliar** na validação. Plotar curvas de loss e accuracy para conferir que o modelo está aprendendo sem overfitar demais.
6. *(Extensão opcional)* **Fine-tuning**: destravar as últimas camadas da base, **baixar o learning rate** (para algo como `1e-5`) e treinar por mais algumas épocas.

---

## Dicas finais

- Espere uma **acurácia de validação bem alta já no feature extraction** — o MobileNetV2 já viu uma quantidade enorme de animais em ImageNet; as features dele para alpaca são ótimas desde o primeiro epoch.
- Rode `model.summary()` antes de treinar e confira o número de **trainable params** (deve ser muito pequeno — só o head) vs **non-trainable params** (na casa dos milhões — a base congelada).
- Se você for fazer fine-tuning na parte opcional, **sempre** recompile o modelo depois de mexer em `trainable`. Caso contrário, o Keras usa as configurações antigas e o treino não surte efeito esperado.
- Em máquinas sem GPU (Colab gratuito / notebooks locais), o MobileNetV2 treina confortavelmente. Não precisa escalar para algo mais pesado.

!!! warning "Erros clássicos nesta prática"
    - **Normalizar com `/255`** em vez de `preprocess_input`. A acurácia fica aleatória.
    - **Esquecer `include_top=False`** na hora de carregar o MobileNetV2. Você acaba com o head de 1000 classes do ImageNet em cima, e o modelo inteiro vira inútil.
    - **Passar `training=True` para a base** durante o `fit` quando ela está congelada. Isso atualiza as estatísticas de BN mesmo sem treinar pesos.
    - **Fazer fine-tuning sem abaixar o learning rate**. A base esquece tudo em 1 batch.

---

!!! Author
    - **Vitor Salomão**

<div class="handout-nav" markdown>
[← Anterior: Transfer Learning](transfer_learning.md){ .md-button }
</div>

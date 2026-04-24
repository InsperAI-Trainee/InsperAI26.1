# Aula 7 — ResNet e Transfer Learning

!!! NOTE "Vídeos"
    Antes de começar, assista os vídeos do curso de Deep Learning do Andrew Ng. Esta aula cobre dois temas que são trabalhados em sequência na mesma semana do curso.

    - [Convolutional Neural Networks — Week 2: Deep CNN models](https://www.youtube.com/watch?v=_gRTluL-LL8&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) (Coursera / DeepLearning.AI)

    Leitura complementar (opcional, nível técnico mais alto):

    - He et al., *Deep Residual Learning for Image Recognition* — [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385) *(o paper original da ResNet)*

Na aula anterior você construiu uma CNN do zero e a treinou em um dataset pequeno. Ela funciona — mas tem dois limites que ficam visíveis logo que o problema sobe de escala:

- **Quando você tenta ir mais fundo**, a rede começa a treinar pior, não melhor. É um problema real, e tem nome: *degradation*.
- **Quando você tem poucos dados**, treinar uma arquitetura robusta do zero não é viável. Milhares de parâmetros + centenas de exemplos = overfitting garantido.

Esta aula é sobre as respostas canônicas para esses dois problemas — e sobre o que acontece no meio do caminho entre "ideia" e "código que roda":

1. **ResNet** — a arquitetura que destravou redes realmente profundas, via uma mudança quase simples demais para ser verdade: deixar a rede aprender **o resíduo**.
2. **Arquiteturas de CNNs** — como a ResNet se encaixa na história maior das CNNs, por que existem várias arquiteturas e por que a escolha padrão em projeto aplicado hoje é a **MobileNetV2**.
3. **Transfer Learning** — a prática de pegar um modelo já treinado em um dataset enorme (ImageNet) e reaproveitar aqueles pesos para resolver seu próprio problema com muito menos dados.

No final, você vai juntar tudo: usar o **MobileNetV2** pré-treinado como base para classificar imagens próprias, via transfer learning.

---

## Nesta aula veremos:

**1. Redes profundas e ResNets**

- Por que redes profundas sem atalhos degradam (o *degradation problem*)
- A ideia de **skip connection** e o bloco residual
- Por que o atalho resolve o vanishing gradient e a dificuldade de aprender a identidade

**2. Arquiteturas de CNNs**

- Panorama das CNNs marcantes — LeNet, AlexNet, VGG, Inception, ResNet
- Restrições do mundo real: tamanho, latência, consumo
- Por que a **MobileNetV2** existe, como ela usa a skip connection da ResNet, e por que ela é o padrão para projetos aplicados

**3. Transfer Learning**

- Por que transfer learning funciona — hierarquia de features
- **Feature extraction** vs **fine-tuning** — quando usar cada um
- Pipeline completo em Keras com MobileNetV2
- Preparação para a prática com o **Alpaca dataset**

---

## Estrutura da Aula

Esta aula está dividida em três páginas conceituais + a prática:

1. [**Redes profundas e ResNets**](resnet.md)
2. [**Arquiteturas de CNNs**](arquiteturas_cnn.md)
3. [**Transfer Learning**](transfer_learning.md)
4. [**Prática**](pratica_handout.md)

---

## Referências

Deep Learning Specialization — **Andrew Ng**, DeepLearning.AI

- Course 4 — *Convolutional Neural Networks*, Week 2 (Deep CNN models)

He, K., Zhang, X., Ren, S., Sun, J. (2015). **Deep Residual Learning for Image Recognition**.
*arXiv preprint arXiv:1512.03385.*

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. *arXiv preprint arXiv:1801.04381.*

GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**.
3. ed. Sebastopol: O'Reilly Media, 2022.

- Capítulo 14 — *Deep Computer Vision Using Convolutional Neural Networks*
- Capítulo 15 — *Reusing Pretrained Layers*

---

!!! Author
    - **Vitor Salomão**

<div class="handout-nav" markdown>
[← Aula 6 — Prática](../aula-06/pratica_handout.md){ .md-button }
[Próxima: Redes profundas e ResNets →](resnet.md){ .md-button .md-button--primary }
</div>

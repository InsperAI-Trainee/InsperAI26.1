# Aula 6 — Visão Computacional e CNNs

!!! NOTE "Vídeos"
    Antes de começar, assista os vídeos do curso de Deep Learning do Andrew Ng. São a principal referência conceitual desta aula.

    - [Convolutional Neural Networks — Week 1: Foundations of CNNs](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) (Coursera / DeepLearning.AI)
    - [Convolutional Neural Networks — Week 2: Deep CNN models](https://www.youtube.com/watch?v=_gRTluL-LL8&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) *(opcional — para quem quiser aprofundar em arquiteturas clássicas)*

    Adicionalmente, uma abordagem visual excelente do 3Blue1Brown:

    - [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)

Até aqui, tudo que você modelou foi uma tabela: colunas de *features*, linhas de amostras, previsões saindo do outro lado. Agora entra outra família de dados — **imagens**. E com ela, uma ferramenta nova: as **Redes Neurais Convolucionais (CNNs)**.

O objetivo desta aula é responder, na ordem:

1. Por que um MLP não é a melhor escolha quando a entrada é uma imagem?
2. O que é uma convolução, e por que essa operação resolve exatamente os problemas do MLP?
3. Como empilhar essas operações em uma arquitetura que **realmente** funciona?

A matemática aqui é mínima. O que importa é construir **intuição visual** — o tipo de intuição que te permite olhar para uma arquitetura nova e entender o porquê de cada camada estar ali.

---

## Nesta aula veremos:

- Por que MLPs não são ideais para imagens
- A operação de convolução e seus parâmetros (kernel, stride, padding)
- Camadas de pooling e fully-connected
- Arquitetura típica de uma CNN
- Preparação para a atividade prática com o **Signs Dataset**

---

## Estrutura da Aula

Esta aula está dividida em três páginas conceituais + a prática:

1. [**Por que MLPs falham em imagens**](visao_computacional.md)
2. [**A operação de convolução**](convolucao.md)
3. [**Arquitetura de uma CNN**](arquitetura_cnn.md)
4. [**Prática**](pratica_handout.md)

---

## Referências

Deep Learning Specialization — **Andrew Ng**, DeepLearning.AI

- Course 4 — *Convolutional Neural Networks*, Week 1 (Foundations of CNNs)
- Course 4 — *Convolutional Neural Networks*, Week 2 (Deep CNN models)

GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**.
3. ed. Sebastopol: O'Reilly Media, 2022.

- Capítulo 14 — *Deep Computer Vision Using Convolutional Neural Networks*

3Blue1Brown — *But what is a convolution?*

---

!!! Author
    - **Vitor Salomão**

<div class="handout-nav" markdown>
[← Aula 5 — Melhorando o Modelo](../aula-05/melhorando_modelo.md){ .md-button }
[Próxima: Por que MLPs falham em imagens →](visao_computacional.md){ .md-button .md-button--primary }
</div>

# Aula 5 — Construção de Modelos

Na aula 4, você abriu a caixa-preta das redes neurais e viu o que acontece dentro de um MLP. Ótimo. Mas entender a matemática não significa que faz sentido reimplementar tudo na mão para sempre.

Nesta aula, vamos dar o próximo passo: usar um framework de verdade para **construir, compilar, treinar e avaliar modelos**. Depois, vamos discutir como sair de um modelo que apenas roda para um modelo que **treina melhor, converge melhor e generaliza melhor**.

---

## Nesta aula veremos:

- Como montar um MLP com **TensorFlow/Keras**
- O papel de `Sequential`, camadas, otimizador, loss e métricas
- O que `model.compile`, `model.fit` e `model.evaluate` realmente fazem
- Como normalizar os dados de treino
- Como ajustar `learning_rate` e `batch_size`
- O papel da `BatchNormalization`
- Como reduzir overfitting com L2 e `Dropout`

---

## Estrutura da Aula

Esta aula está dividida em duas partes:

1. [**Construindo um MLP com TensorFlow/Keras**](construcao_modelos.md)
2. [**Melhorando o Modelo**](melhorando_modelo.md)

---

## Referências

Machine Learning Specialization - Andrew Ng, DeepLearning.AI em colaboração com Stanford Online:

Deep Learning Specialization — Andrew Ng, DeepLearning.AI

GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**.
3. ed. Sebastopol: O'Reilly Media, 2022.

- Capítulo 10 — *Introduction to Artificial Neural Networks with Keras*

!!! Author
    - **Thomas Kassabian**

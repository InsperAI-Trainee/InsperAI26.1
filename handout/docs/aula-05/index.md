# Aula 5 — Construção de Modelos

Na aula 4, você abriu a caixa-preta das redes neurais e viu o que acontece dentro de um MLP. Ótimo. Mas entender a matemática não significa que faz sentido reimplementar tudo na mão para sempre.

Nesta aula, vamos dar o próximo passo: usar um framework de verdade para **construir, compilar, treinar e avaliar modelos**. Depois, vamos discutir o problema que separa um modelo que só parece bom de um modelo que realmente generaliza: **underfitting** e **overfitting**.

---

## Nesta aula veremos:

- Como montar um MLP com **TensorFlow/Keras**
- O papel de `Sequential`, camadas, otimizador, loss e métricas
- O que `model.compile`, `model.fit` e `model.evaluate` realmente fazem
- O que é **underfitting**
- O que é **overfitting**
- Como diagnosticar falta de capacidade ou excesso de memorização
- Técnicas para melhorar generalização

!!! info "A ideia central"
    Construir modelo é a parte fácil.
    O difícil é construir um modelo que aprenda o padrão certo, em vez de decorar ruído.

---

## Estrutura da Aula

Esta aula está dividida em duas partes:

1. [**Construindo um MLP com TensorFlow/Keras**](construcao_modelos.md)
2. [**Overfitting e Underfitting**](overfitting_underfitting.md)

---

## Referências

GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**.
3. ed. Sebastopol: O'Reilly Media, 2022.

- Capítulo 10 — *Introduction to Artificial Neural Networks with Keras*
- Capítulo 11 — *Training Deep Neural Networks*

!!! Author
    - **Thomas Kassabian**

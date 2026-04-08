# Aula 2 — Classificação e Regressão Logística

Nesta aula, saímos de problemas de **previsão contínua** e entramos em problemas de **decisão entre classes**.
Em vez de prever um valor como preço ou temperatura, agora o modelo precisa responder perguntas como:

- isso é spam ou não?
- a transação parece fraude ou não?
- o tumor parece benigno ou maligno?

O objetivo da aula não é só treinar um classificador com `scikit-learn`.
É entender:

- o que muda quando o target deixa de ser contínuo
- como a **Regressão Logística** transforma um score linear em probabilidade
- qual é o papel da **função sigmoide**
- como a decisão final depende de threshold
- por que avaliar um classificador é mais sutil do que olhar uma única métrica

---
## Workflow da Aula

A sequência desta aula é:

1. Entender o que muda quando saímos de regressão e entramos em classificação.
2. Ver como a Regressão Logística produz scores, probabilidades e classes.
3. Entender a função sigmoide e a fronteira de decisão.
4. Construir um baseline com `LogisticRegression`.
5. Avaliar esse baseline com matriz de confusão, acurácia, precisão, recall e F1-score.
6. Discutir por que a melhor métrica depende do **custo do erro**.

---
## Material

Esta aula é composta por um handout e uma atividade prática em notebook.

Sequência recomendada:

1. Leia [O Problema de Classificação](classificacao.md).
2. Leia [Regressão Logística e Sigmoide](logistica.md).
3. Leia [Métricas e Trade-offs](metricas.md).
4. Faça a atividade em [Prática](pratica.md).

---
## Dataset da Aula

A prática usa o **Breast Cancer Wisconsin Dataset**, disponível no `scikit-learn`.

Cada linha representa uma amostra com 30 features numéricas extraídas de exames.
O target é binário:

- `0 = malignant`
- `1 = benign`

!!! warning "Não confunda rótulo numérico com classe de interesse"
    O valor `1` não significa automaticamente "caso mais importante".
    Em classificação, você precisa sempre definir explicitamente qual classe está tratando como **positiva** e qual erro custa mais caro.

---
## Referências

- **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (Aurélien Géron) - Capítulos 3 e 4
- Andrew Ng, **Machine Learning Specialization** - classificação logística e métricas de classificação

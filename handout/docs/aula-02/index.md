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
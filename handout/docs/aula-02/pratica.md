# Prática

O notebook `classificadores.ipynb` é a atividade principal da aula 2.
O handout concentra a teoria; o notebook coloca o classificador em execução.

---
## O que você vai fazer

1. Carregar o Breast Cancer Wisconsin Dataset.
2. Fazer uma inspeção curta da base e do target.
3. Separar treino e teste.
4. Treinar um baseline com `LogisticRegression`.
5. Inspecionar previsões, coeficientes e a lógica da decisão.
6. Gerar previsões no conjunto de teste.
7. Calcular e interpretar matriz de confusão, acurácia, precisão, recall e F1-score.

---
## O que interessa de verdade

Não perca tempo transformando esta aula em uma nova EDA longa.
O foco aqui é outro:

- classificação binária
- regressão logística como primeiro classificador
- papel da sigmoide e do threshold
- interpretação das previsões
- leitura correta das métricas
- relação entre métrica e custo do erro

---
## Cuidados

- Não misture treino e teste.
- Não trate acurácia como critério suficiente.
- Não interprete precisão e recall sem declarar a classe positiva.
- Se usar o dataset do `sklearn` como está, lembre que `1 = benign`.

---
## Ponte para a próxima aula

Nesta aula você usa a regressão logística como modelo de classificação e entende sua estrutura.
Na próxima, a pergunta muda:

**o que o modelo está fazendo por dentro quando chamamos `.fit()`?**

É isso que o notebook `lr-bgd.ipynb` vai abrir.

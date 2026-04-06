# Treinando o Modelo

Esta página acompanha as etapas **4 a 8** do notebook:

1. Preparação dos dados
2. Treino do modelo baseline
3. Avaliação do baseline
4. Limpeza dos dados suspeitos
5. Comparação entre modelo bruto e modelo limpo

O handout não traz as saídas prontas.  
Ele serve como roteiro de interpretação para o que você vai executar no notebook.

---

## 4. Preparação dos Dados

No notebook, primeiro treinamos um **baseline com os dados brutos**.

Nesta etapa, você deve:

- separar features e target em `X` e `y`
- fazer `train_test_split` com `test_size=0.2` e `random_state=42`
- confirmar o tamanho dos conjuntos de treino e teste

### O que observar

- O target `MedHouseVal` não entra em `X`.
- O conjunto de teste precisa ficar isolado até a avaliação.
- O split é a referência para medir generalização, não memorização.

!!! warning "Por que não avaliar no treino?"
    Se você medir desempenho apenas nos dados usados no ajuste, não está avaliando generalização.  
    Está medindo o quanto o modelo se adaptou ao próprio treino.

---

## 5. Treino do Modelo Baseline

Agora, no notebook, treine um `LinearRegression()` nos dados brutos.

Use nomes consistentes, para facilitar a comparação posterior:

- `model_bruto`
- `coef_brutos`

### O que observar

- O intercepto é o valor base do modelo.
- Cada coeficiente indica o efeito marginal de uma feature, mantendo as demais fixas.
- O sinal e a magnitude dos coeficientes precisam ser interpretados com cuidado, especialmente quando há multicolinearidade.

### Perguntas para responder no notebook

1. Qual feature tem o maior coeficiente positivo?
2. Algum sinal parece contraintuitivo?
3. O ranking por valor absoluto faz sentido econômico?

!!! note "Ligação com a teoria"
    Se os coeficientes parecerem misteriosos, volte para a página [Regressão Linear](teoria.md), especialmente a parte de notação vetorial e interpretação dos pesos.

---

## 6. Avaliação do Baseline

No notebook, gere previsões para treino e teste e organize as métricas em `metricas_brutas`.

As métricas mínimas desta aula são:

- `R²`
- `RMSE`
- `MAE`

Além disso, monte o gráfico de **valor real vs valor predito**.

### O que observar

- Se o desempenho em treino e teste é parecido ou muito diferente.
- Se o `R²` de teste parece razoável para um baseline simples.
- Se há sinais visuais de limite artificial no target.
- Se a dispersão dos pontos sugere que a relação não é perfeitamente linear.

### Perguntas para responder no notebook

1. O modelo parece generalizar ou memorizar?
2. O gráfico `real vs predito` revela efeito do teto em `MedHouseVal`?
3. O erro do modelo parece pequeno o bastante para dizer que a regressão linear resolveu o problema?

---

## 7. Limpando Dados Suspeitos e Treinando de Novo

Agora o notebook repete o pipeline após uma limpeza simples:

- remover linhas com `MedHouseVal == 5.0`
- remover linhas com `AveOccup > 20`

Use nomes consistentes:

- `df_limpo`
- `X_limpo`, `y_limpo`
- `X_train_limpo`, `X_test_limpo`, `y_train_limpo`, `y_test_limpo`
- `model_limpo`
- `metricas_limpas`

### O que observar

- Quantas linhas foram removidas.
- Se os coeficientes mudaram de forma relevante.
- Se as métricas melhoraram de maneira clara ou apenas marginal.

!!! note "Leitura correta da limpeza"
    Melhorar a métrica é importante, mas não é a única razão para limpar dados.  
    Às vezes a limpeza melhora mais a coerência do problema do que o número final do `R²`.

---

## 8. Comparando Bruto vs Limpo

A última etapa do notebook é comparar os dois modelos lado a lado.

A comparação principal deve incluir:

- `R²`
- `RMSE`
- `MAE`

Se fizer sentido, compare também:

- coeficientes
- resíduos
- gráfico `real vs predito`

### Perguntas para responder no notebook

1. O `R²` de teste melhorou?
2. `RMSE` e `MAE` diminuíram?
3. A limpeza ajudou mais na interpretação do problema ou na performance?
4. O baseline limpo parece um ponto de partida melhor para as próximas aulas?

!!! tip "Fechamento da aula"
    O resultado mais importante não é “qual número ficou maior”.  
    É entender como decisões sobre dados mudam o comportamento do modelo.

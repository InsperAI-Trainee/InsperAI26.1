# Métricas e Trade-offs

Agora que já existe um modelo concreto para classificação, a pergunta deixa de ser "como ele prevê?" e passa a ser:

**como saber se ele está errando do jeito certo ou do jeito perigoso?**

Avaliar um classificador não é só medir quantas previsões ele acertou.
Essa é a forma mais rápida de esconder um modelo ruim.

O ponto central é simples:

- classificadores erram de formas diferentes
- erros diferentes têm custos diferentes
- a métrica certa depende desse custo

---
## Matriz de Confusão

A matriz de confusão organiza as previsões em quatro grupos:

| Real \ Predito | Positivo | Negativo |
|---|---:|---:|
| Positivo | TP | FN |
| Negativo | FP | TN |

Onde:

- `TP` = verdadeiro positivo
- `TN` = verdadeiro negativo
- `FP` = falso positivo
- `FN` = falso negativo

Sem essa matriz, as métricas viram números soltos.
Com ela, você enxerga **o padrão dos erros**.

Ela também ajuda a conectar a teoria da aula com a decisão do modelo:

- a regressão logística produz probabilidade
- o threshold produz a classe
- a matriz mostra a consequência prática dessa decisão

---
## Acurácia

A acurácia mede a fração total de previsões corretas:

$$
\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Ela é útil como visão geral, mas falha em dois cenários comuns:

- quando as classes são desbalanceadas
- quando os custos dos erros são muito diferentes

### Exemplo clássico

Se apenas 1% das transações é fraude, um modelo que prevê "não é fraude" para tudo terá 99% de acurácia e ainda assim será inútil.

!!! warning "Acurácia alta não prova qualidade"
    Se a métrica principal da aula virar só acurácia, o aluno aprende uma heurística ruim.
    Acurácia é resumo.
    Ela não substitui a análise dos tipos de erro.

---
## Precisão

A precisão responde:

**das previsões positivas, quantas estavam certas?**

$$
\text{Precisão} = \frac{TP}{TP + FP}
$$

Ela é importante quando **falso positivo custa caro**.

Exemplos:

- filtro de spam que apaga e-mails importantes
- sistema antifraude que bloqueia compra legítima
- triagem automática que gera muitos alarmes falsos

Precisão alta significa que, quando o modelo acusa positivo, ele costuma estar certo.

---
## Recall

O recall responde:

**dos positivos reais, quantos o modelo conseguiu encontrar?**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Ele é importante quando **falso negativo custa caro**.

Exemplos:

- deixar passar uma fraude real
- não detectar uma doença
- não identificar um defeito crítico

Recall alto significa que o modelo está deixando poucos casos positivos escaparem.

---
## Trade-off entre Precisão e Recall

Em muitos problemas, aumentar uma tende a piorar a outra.

Se você deixa o classificador mais conservador:

- ele faz menos previsões positivas
- a precisão pode subir
- o recall pode cair

Se você deixa o classificador mais sensível:

- ele marca mais casos como positivos
- o recall pode subir
- a precisão pode cair

Isso não é defeito do modelo.
É parte natural do problema.

Em boa parte dos casos, esse trade-off aparece quando você muda o threshold de decisão.

---
## F1-Score

O F1-score combina precisão e recall em um único número:

$$
F1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}
$$

Ele é útil quando você quer um balanço entre as duas métricas.

### Quando usar

- quando precisão e recall importam ao mesmo tempo
- quando você quer comparar modelos sem favorecer só um tipo de erro

### Quando não usar como única métrica

- quando um tipo de erro é claramente mais grave que o outro

Se falso negativo é muito pior, olhe recall primeiro.
Se falso positivo é muito pior, olhe precisão primeiro.

---
## A classe positiva precisa ser explícita

Métricas como precisão e recall dependem de qual classe você está chamando de **positiva**.

No dataset desta aula, por padrão:

- `0 = malignant`
- `1 = benign`

Se você usa as funções padrão do `sklearn` sem ajustar isso, a leitura de precisão e recall passa a ser feita em relação à classe `1`.

!!! warning "Erro conceitual comum"
    Em contexto médico, muita gente fala de recall como se estivesse medindo a detecção de casos graves, mas o código às vezes está medindo recall da classe oposta.
    Se você não explicita a classe positiva, sua interpretação pode ficar errada.

---
## O que observar no notebook

Quando rodar a prática, responda mentalmente a estas perguntas:

1. O modelo está errando mais por falso positivo ou por falso negativo?
2. A acurácia está escondendo algum comportamento relevante?
3. Se o problema fosse médico, qual erro seria menos aceitável?
4. As métricas calculadas estão mesmo usando a classe de interesse como positiva?

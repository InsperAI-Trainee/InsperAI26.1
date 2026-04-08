# O Problema de Classificação

Na aula anterior, o target era um número contínuo.
Aqui, o target representa uma **categoria**.

Exemplos:

- `spam` ou `não spam`
- `fraude` ou `normal`
- `doente` ou `saudável`

Isso muda duas coisas centrais:

- o modelo agora precisa **separar classes**, não ajustar uma reta a valores contínuos
- a avaliação deixa de ser "o erro médio foi pequeno?" e passa a ser "que tipo de erro o modelo está cometendo?"

---
## Regressão vs Classificação

| Tipo de problema | Saída esperada | Exemplo |
|---|---|---|
| Regressão | valor contínuo | preço de uma casa |
| Classificação binária | uma entre duas classes | spam ou não spam |
| Classificação multiclasse | uma entre várias classes | espécie de flor |

Nesta aula, o foco é **classificação binária**. Ou seja, o modelo classifica a entrada entre **dois** rótulos.

---
## O que significa errar em classificação

Em regressão, o modelo tenta prever um valor contínuo. Portanto, errar em problemas de regressão quer dizer "quão longe da amostra real um erro de `+10` e um erro de `+100` diferem em magnitude.

Em classificação, o foco costuma ser outro: **que lado da fronteira o modelo escolheu**.

Isso produz quatro casos possíveis:

- verdadeiro positivo
- verdadeiro negativo
- falso positivo
- falso negativo

Esses quatro casos são a base de praticamente toda avaliação de classificadores.

### Exemplo

Imagine um modelo que detecta fraude:

- **verdadeiro positivo**: marcou fraude e era fraude
- **verdadeiro negativo**: liberou e era normal
- **falso positivo**: bloqueou uma operação legítima
- **falso negativo**: deixou passar uma fraude real

??? question "Pergunta: considerando os 4 casos, por que analisar somente o percentual de acerto do modelo pode ser simplista?"
    O percentual de acerto (**acurácia**) junta todos os acertos e erros em um único número e esconde **que tipo de erro** o modelo está cometendo.
    
    Um modelo pode ter acurácia alta e ainda assim falhar justamente no erro mais grave do problema. Por exemplo, pense o que poderia acontecer se um modelo de classificação de tumores (benigno ou maligno) devolver um falso negativo.

---
## Fechamento

Até aqui, a estrutura do problema está clara:

- temos classes, não valores contínuos
- o modelo precisa produzir uma decisão
- errar em problemas de regressão é diferente de errar em problemas de classificação

Mas como de fato isso funciona?

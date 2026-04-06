# Aula 1 — Workflow de ML e Regressão Linear

Nesta aula, você vai compreender o fluxo de trabalho de um projeto de IA e implementar o seu primeiro modelo de IA! O fio condutor é o **California Housing Dataset**, um conjunto de dados real, com problemas reais, que vai nos forçar a tomar decisões antes mesmo de treinar qualquer modelo.

O objetivo não é só fazer o modelo funcionar. É entender **por que** cada etapa existe, o que cada saída significa e o que acontece quando você pula uma etapa do workflow.

---
## Workflow de ML

Em qualquer projeto de IA, o fluxo básico costuma ser:

1. Entender o problema e o dataset.
2. Explorar os dados e procurar sinais de erro, assimetria e outliers.
3. Separar treino e teste.
4. Treinar um primeiro baseline.
5. Avaliar com métricas e gráficos.
6. Ajustar os dados ou o modelo e comparar os resultados.

É exatamente essa sequência que vamos seguir nesta aula. Cada uma dessas etapas será explicada.

---
## Material

Esta aula é composta por um handout (esta página) e uma atividade prática em um notebook.

Sequência recomendada:

1. Leia esta visão geral e a página de [Regressão Linear](teoria.md).
2. Abra o notebook do assignment e execute a atividade guiada.
3. Use [Exploração dos Dados](exploracao.md), [Outliers e Dados Suspeitos](outliers.md) e [Treinando o Modelo](treinamento.md) em paralelo, na mesma ordem do notebook.

---
## Vídeos

Antes de começar, assista as aulas do curso de ML do GOAT **Andrew Ng**.
> - [Machine Learning Specialization - vídeos 9 a 14](https://www.youtube.com/watch?v=dLc-lfEEYss&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=9)

Alternativamente, caso você tenha tempo, recomendamos fortemente assistir assistir essa masterclass do GOAT que vai do conceito mais fundamental da IA até a matemática do treinamento de modelos.

> - [Stanford CS229: Machine Learning - Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8)


Caso não ache necessário ver um vídeo grande assim, temos essa outra opção do StatQuest

> - [ Linear Regression, Clearly Explained!!! ](https://www.youtube.com/watch?v=7ArmBVF2dCs)

Caso você ainda ache muito grande, primeiro, largue os vídeos curtos, sério, isso está acabando com sua vida. Segundo, veja esse vídeo do 3-Minute Data Science

> - [Linear Regression in 3 Minutes ](https://www.youtube.com/watch?v=3dhcmeOTZ_Q)

---

## O Dataset

O **California Housing Dataset** é derivado do censo de 1990 do estado da Califórnia. Cada linha representa um **bloco censitário**, a menor unidade geográfica do censo americano, tipicamente com 600 a 3.000 habitantes.

| Coluna | Descrição |
|---|---|
| `MedInc` | Renda mediana dos domicílios (dezenas de milhares de USD) |
| `HouseAge` | Idade mediana das casas do bloco |
| `AveRooms` | Média de cômodos por domicílio |
| `AveBedrms` | Média de quartos por domicílio |
| `Population` | População total do bloco |
| `AveOccup` | Média de moradores por domicílio |
| `Latitude` | Latitude do bloco |
| `Longitude` | Longitude do bloco |
| `MedHouseVal` | **Target** — Valor mediano das casas (centenas de milhares de USD) |

!!! warning "Dados do censo de 1990"
    O dataset tem mais de 30 anos. Valores absolutos de preço e renda estão desatualizados — mas os **padrões e relações entre variáveis** continuam sendo um excelente campo de aprendizado.

---

## Referências
- **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (Aurélien Géron) - Capítulo 4

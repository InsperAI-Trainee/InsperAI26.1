# Aula 1 — Workflow de ML e Regressão Linear

Nesta aula saímos da teoria e colocamos a mão no dado. O fio condutor é o **California Housing Dataset**, um conjunto real, com problemas reais, que vai nos forçar a tomar decisões antes mesmo de treinar qualquer modelo.

O objetivo não é só fazer o modelo funcionar. É entender **por que** cada etapa existe e o que acontece quando você a pula.

---
## Como usar os materiais

Esta aula foi organizada para que o **handout** e o **notebook** tenham papéis diferentes e complementares:

| Material | Papel na aula |
|---|---|
| Handout | Explica a teoria, mostra a intuição matemática e documenta o estudo de caso |
| [Notebook da prática](pratica.md) | Guia a atividade em código: exploração, investigação de outliers, treino e comparação de modelos |

Sequência recomendada:

1. Leia esta visão geral e a página de [Regressão Linear](teoria.md).
2. Abra o [Notebook da prática](pratica.md) e execute a atividade guiada.
3. Volte às páginas de exploração, outliers e treinamento sempre que quiser interpretar melhor o que apareceu no notebook.
4. Feche a aula com [Atividades](atividades.md) e [Resumo](resumo.md).

---
## Vídeos prévios
Caso você ache que faz sentido, e tenha tempo disponível, é fortemente recomendado que você assista esse vídeo antes da aula, ele é do nosso mestre todo poderoso **Andrew Ng**, usaremos muitas coisas ensinadas por ele!

> - [Stanford CS229: Machine Learning - Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8)

Caso não ache necessário ver um vídeo grande assim, temos essa outra opção do StatQuest

> - [ Linear Regression, Clearly Explained!!! ](https://www.youtube.com/watch?v=7ArmBVF2dCs)

Caso você ainda ache muito grande, primeiro, largue os vídeos curtos, sério, isso está acabando com sua vida, segundo, veja esse vídeo do 3-Minute Data Science

> - [Linear Regression in 3 Minutes ](https://www.youtube.com/watch?v=3dhcmeOTZ_Q)

---
## O que vamos estudar

| Seção | Conteúdo |
|---|---|
| 📖 Teoria | Equação da reta, resíduos, MSE, notação vetorial e R² |
| 💻 Prática guiada | Notebook principal da aula com baseline, análise de outliers e comparação antes/depois da limpeza |
| 🔍 Exploração | Carregamento, estatísticas descritivas e visualizações do dataset |
| ⚠️ Outliers | Identificação de dados suspeitos que podem prejudicar o modelo |
| 🤖 Treinamento | Divisão treino/teste, `LinearRegression` e avaliação com R² |
| 🔗 Leitura Complementar | Análise completa de Gabriel Valentim (NeroAI) |
| ✏️ Atividades | Exercícios pós-notebook para fixar e estender o que foi feito na prática |

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

!!! info "Referência"
    Este material acompanha o capítulo 4 do livro **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (Aurélien Géron).

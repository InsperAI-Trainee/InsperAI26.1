# Aula 01 — Regressão Linear

Nesta aula saímos da teoria e colocamos a mão no dado. O fio condutor é o **California Housing Dataset** — um conjunto real, com problemas reais, que vai nos forçar a tomar decisões antes mesmo de treinar qualquer modelo.

O objetivo não é só fazer o modelo funcionar. É entender **por que** cada etapa existe e o que acontece quando você a pula.

---

## O que vamos estudar

| Seção | Conteúdo |
|---|---|
| 📖 Teoria | Equação da reta, resíduos, MSE, notação vetorial e R² |
| 🔍 Exploração | Carregamento, estatísticas descritivas e visualizações do dataset |
| ⚠️ Outliers | Identificação de dados suspeitos que podem prejudicar o modelo |
| 🤖 Treinamento | Divisão treino/teste, `LinearRegression` e avaliação com R² |
| 🔗 Leitura Complementar | Análise completa de Gabriel Valentim (NeroAI) |
| ✏️ Atividades | Exercícios para fixar os conceitos |

---

## O Dataset

O **California Housing Dataset** é derivado do censo de 1990 do estado da Califórnia. Cada linha representa um **bloco censitário** — a menor unidade geográfica do censo americano, tipicamente com 600 a 3.000 habitantes.

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
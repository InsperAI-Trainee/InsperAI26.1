# Regressão Linear com o Dataset California Housing

Bem-vindo ao material de apoio desta aula. Aqui você encontrará o conteúdo completo abordado em sala, com os códigos, gráficos e explicações organizados por seção.

---

## O que vamos estudar

| Seção | Conteúdo |
|---|---|
| 🔍 Exploração | Carregamento, estatísticas descritivas e visualizações do dataset |
| ⚠️ Outliers | Identificação de dados suspeitos que podem prejudicar o modelo |
| 🤖 Treinamento | Divisão treino/teste, `LinearRegression` e avaliação com R² |
| ✏️ Atividades | Exercícios para fixar os conceitos |

---

## Dataset

Usamos o **California Housing Dataset**, derivado do censo de 1990 do estado da Califórnia.  
Cada linha representa um **bloco censitário** — a menor unidade geográfica do censo americano.

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

---

!!! info "Referência"
    Este material acompanha o capítulo 4 do livro **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (Aurélien Géron).

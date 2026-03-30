# 📋 Resumo da Aula

Referência completa de tudo que foi visto na Aula 01. Use esta página para revisar antes de exercícios, provas ou projetos.

---

## Tabela de Fórmulas

| Conceito | Fórmula | O que significa |
|---|---|---|
| **Regressão simples** | $\hat{y} = mx + b$ | Previsão com uma feature |
| **Regressão múltipla** | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ | Previsão com várias features (produto escalar) |
| **Forma matricial** | $\hat{\mathbf{y}} = X\mathbf{w} + b$ | Previsão para todo o dataset de uma vez |
| **Resíduo** | $r_i = y_i - \hat{y}_i$ | Erro individual de cada amostra |
| **MSE** | $\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Média dos erros ao quadrado: usada no treino |
| **RMSE** | $\sqrt{MSE}$ | Erro médio na unidade original do problema |
| **R²** | $1 - \dfrac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Fração da variância do target explicada pelo modelo |
| **Equação Normal** | $\hat{\mathbf{w}} = (X^TX)^{-1}X^T\mathbf{y}$ | Solução fechada: encontra os pesos ótimos direto |

!!! tip "Como interpretar o R²"
    - **R² < 0.5** → modelo fraco, não captura os padrões principais
    - **R² entre 0.5 e 0.8** → razoável, mas há espaço para melhorar
    - **R² > 0.8** → bom ajuste, mas sempre verifique se não é overfitting

---

## Pipeline Completo

```python
# 1. Carregar o dataset
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame

# 2. Separar features e target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 3. Dividir treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Treinar o modelo
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

# 5. Inspecionar coeficientes
print("Intercepto:", model.intercept_)
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat}: {coef:+.4f}")

# 6. Avaliar
from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print(f"R² Teste: {r2:.4f}")
```

---

## Outliers — Sinais de Alerta

| Variável | Threshold suspeito | Causa provável |
|---|---|---|
| `MedHouseVal` | Pico exato em 5.0 | Preço censurado acima de USD 500k |
| `AveRooms` | > 50 cômodos/domicílio | Hotéis, repúblicas |
| `AveOccup` | > 20 pessoas/domicílio | Prisões, hospitais, quartéis |

!!! warning "Efeito no modelo"
    Outliers elevam o MSE de forma desproporcional, ao elevar o erro ao quadrado, um ponto muito distante domina a função de custo e força o modelo a distorcer os coeficientes para tentar acomodá-lo.

---
## Glossário

### Dados

**Amostra (sample)**
: Uma linha do dataset. No California Housing, cada amostra é um bloco censitário.

**Feature**
: Uma coluna de entrada usada para fazer previsões. Ex: `MedInc`, `AveRooms`.

**Target**
: A variável que queremos prever. No nosso caso: `MedHouseVal`.

**Outlier**
: Ponto do dataset com valor muito distante dos demais. Pode representar um erro de coleta ou um caso genuinamente anômalo — como um presídio num dataset residencial.

---

### Modelo

**Parâmetros**
: Os valores que o modelo aprende durante o treino, os pesos $\mathbf{w}$ e o intercepto $b$.

**Intercepto ($b$)**
: Valor previsto pelo modelo quando todas as features são zero. Representa o "ponto de partida" da reta.

**Peso ($w_i$)**
: Coeficiente associado a uma feature. Indica quanto o valor previsto muda ao aumentar aquela feature em 1 unidade, mantendo as demais fixas.

**Produto escalar**
: Operação entre dois vetores que multiplica elemento a elemento e soma tudo. É exatamente o que acontece quando o modelo faz uma previsão: $\hat{y} = \mathbf{w}^T\mathbf{x} + b$.

**Equação Normal**
: Solução matemática fechada para encontrar os pesos ótimos da regressão linear sem iterações. Funciona bem para datasets pequenos, mas é computacionalmente caro para datasets grandes ($O(n^3)$).

---

### Treino e Avaliação

**Train/test split**
: Divisão do dataset em dois conjuntos: treino (o modelo aprende) e teste (avaliamos a performance real). O modelo nunca vê os dados de teste durante o treino.

**Resíduo**
: Diferença entre o valor real e o previsto para uma amostra: $r_i = y_i - \hat{y}_i$. Resíduos grandes indicam que o modelo está errando naquele ponto.

**MSE (Mean Squared Error)**
: Média dos resíduos ao quadrado. Penaliza erros grandes de forma desproporcional e é a métrica minimizada durante o treino.

**RMSE (Root Mean Squared Error)**
: Raiz quadrada do MSE. Devolve o erro para a unidade original do problema — mais fácil de interpretar que o MSE.

**R² (Coeficiente de Determinação)**
: Fração da variância do target explicada pelo modelo. Vai de 0 a 1 — quanto maior, melhor.

---

### Problemas Comuns

**Overfitting**
: Quando o modelo memoriza os dados de treino e perde capacidade de generalizar. Sinal clássico: R² alto no treino, baixo no teste.

**Underfitting**
: Quando o modelo é simples demais para capturar os padrões dos dados. R² baixo tanto no treino quanto no teste.

**Multicolinearidade**
: Quando duas ou mais features são altamente correlacionadas entre si. Isso torna a matriz $X^TX$ quase singular e distorce os coeficientes do modelo.

---

## Fixando o Conteúdo

<quiz>
O R² de um modelo de regressão linear pode ser negativo.

- [x] Verdadeiro
> Correto! R² negativo significa que o modelo é pior do que simplesmente prever sempre a média do target — algo que só acontece quando os parâmetros estão muito errados ou o modelo foi avaliado em dados muito diferentes dos de treino.
- [ ] Falso
> Incorreto. R² pode sim ser negativo, indicando que o modelo performa pior do que um preditor trivial (a média).
</quiz>

<quiz>
Elevar os resíduos ao quadrado no cálculo do MSE serve para:

- [ ] Facilitar o cálculo computacional
> Não é essa a razão principal.
- [ ] Deixar o erro na mesma unidade do target
> Isso é o que o RMSE faz, não o MSE.
- [x] Penalizar erros grandes de forma mais intensa e evitar que erros positivos e negativos se cancelem
> Correto! Ao elevar ao quadrado, um erro de 10 pesa 4× mais que um erro de 5 — e erros de sinais opostos não se cancelam.
- [ ] Normalizar os resíduos entre 0 e 1
> Isso não é o que acontece com o quadrado.
</quiz>

<quiz>
Qual é a principal razão para separar os dados em treino e teste antes de treinar o modelo?

- [ ] Reduzir o tempo de treinamento
> Não é essa a razão. O split pode até aumentar o tempo se o conjunto de treino ficar pequeno.
- [ ] Aumentar o R² do modelo
> O split não aumenta o R² — pelo contrário, o R² no teste tende a ser menor do que no treino.
- [x] Avaliar a capacidade do modelo de generalizar para dados que ele nunca viu
> Correto! Se avaliássemos no mesmo conjunto de treino, mediríamos apenas a memorização — não a generalização.
- [ ] Garantir que o modelo não use features irrelevantes
> Isso é feito por outras técnicas, como seleção de features ou regularização.
</quiz>

<quiz>
Um modelo com R² = 0.91 no treino e R² = 0.43 no teste está provavelmente sofrendo de:

- [ ] Underfitting
> Underfitting acontece quando o modelo é simples demais e performa mal em treino e teste ao mesmo tempo.
- [x] Overfitting
> Correto! A grande diferença entre R² de treino e teste é o sinal clássico de overfitting — o modelo memorizou os dados de treino e não generalizou.
- [ ] Alta variância no dataset
> Alta variância nos dados pode dificultar o ajuste, mas não explica a diferença tão grande entre treino e teste.
- [ ] Outliers no conjunto de teste
> Outliers podem reduzir o R² no teste, mas não explicam sozinhos uma queda tão drástica.
</quiz>

<quiz>
A Equação Normal $(X^TX)^{-1}X^Ty$ falha quando:

- [ ] O dataset tem mais de 10.000 amostras
> O tamanho do dataset não impede a Equação Normal de funcionar — ela apenas fica mais lenta.
- [x] Duas ou mais features são linearmente dependentes (redundantes)
> Correto! Se duas features são redundantes, $X^TX$ se torna singular (não invertível) e a solução fechada não existe.
- [ ] O target tem distribuição assimétrica
> A distribuição do target não afeta a invertibilidade de $X^TX$.
- [ ] O modelo está em overfitting
> Overfitting não quebra a Equação Normal — é um problema de generalização, não de cálculo.
</quiz>

<quiz>
No California Housing Dataset, um bloco com `AveOccup = 312` provavelmente representa:

- [ ] Um bairro residencial de alta renda
> Bairros residenciais têm em média 2–4 pessoas por domicílio, muito abaixo de 312.
- [ ] Um erro de coleta de dados
> Pode ser, mas há uma explicação mais provável.
- [x] Um estabelecimento institucional como prisão, hospital ou quartel
> Correto! Centenas de pessoas em poucos "domicílios" é o padrão típico de blocos com prisões, hospitais ou bases militares.
- [ ] Um condomínio com muitas unidades
> Condomínios têm muitas unidades separadas — o que aumentaria o denominador e manteria `AveOccup` baixo.
</quiz>

---

!!! info "Referência"
    Este material acompanha o capítulo 4 do livro **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (Aurélien Géron).
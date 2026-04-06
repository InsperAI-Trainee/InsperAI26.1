# Exploração dos Dados

Esta página acompanha as etapas **1. Inspeção inicial** e **2. EDA curta** do notebook da prática.

O papel do handout aqui é dizer **o que fazer** e **o que observar**.  
O papel do notebook é produzir as tabelas, gráficos e respostas.

!!! tip "Como usar esta página"
    Deixe o notebook aberto e avance por ele na mesma ordem dos tópicos abaixo.  
    Sempre que aparecer uma visualização ou uma estatística, volte aqui para interpretar o que vale a pena notar.

---

## 1. Inspeção inicial

No notebook, execute a parte que carrega o `California Housing Dataset` e depois rode os comandos de inspeção básica:

```python
df.head()
df.info()
df.describe().T.round(2)
```

Nesta etapa, o objetivo não é tirar conclusões profundas ainda. É só construir um mapa inicial do dataset.

### O que observar

- Quantas linhas e colunas existem.
- Se há valores faltantes.
- Quais colunas parecem ter grande dispersão.
- Quais máximos estão muito distantes do percentil de 75%.
- Qual coluna é o target: `MedHouseVal`.

### Perguntas para responder no notebook

1. Há valores faltantes?
2. Quais colunas já parecem ter valores extremos?
3. Em quais variáveis a média parece bem distante da mediana?

!!! note "Leitura rápida do `.describe()`"
    - **mean vs 50%**: diferença grande sugere assimetria.
    - **max muito longe do 75%**: sinal clássico de outliers.
    - **std alto**: indica grande dispersão.

---

## 2. EDA curta

No notebook, a ideia é fazer uma exploração curta, suficiente para levantar os sinais mais importantes para o restante da aula.

As visualizações centrais desta etapa são:

- distribuição do target `MedHouseVal`
- distribuição de variáveis como `AveOccup` e `AveRooms`
- mapa de `Longitude` vs `Latitude`, colorido por `MedHouseVal`

Você não precisa esgotar o dataset. Precisa apenas descobrir os padrões que vão afetar o modelo.

### O que observar

- Se existe concentração artificial de valores em `MedHouseVal`.
- Se `AveOccup` e `AveRooms` têm caudas longas ou valores implausíveis.
- Se há padrão geográfico forte no preço das casas.

### Perguntas para responder no notebook

1. O histograma de `MedHouseVal` parece natural ou há algum teto visível?
2. `AveOccup` e `AveRooms` parecem compatíveis com bairros residenciais comuns?
3. Os blocos mais caros parecem distribuídos ao acaso ou concentrados em certas regiões?

!!! warning "Sinal importante"
    Se você encontrar um pico muito forte em `MedHouseVal = 5.0`, guarde isso.  
    Esse detalhe vai voltar na próxima página, quando investigarmos dados suspeitos.

---

## 2.2 Correlações com o Target

Ainda na exploração, calcule as correlações das features com `MedHouseVal`.

O objetivo aqui não é escolher automaticamente as melhores variáveis, e sim ganhar intuição:

- quais variáveis parecem mais associadas ao preço
- quais relações são positivas
- quais relações são negativas

### O que observar

- Se `MedInc` aparece entre as maiores correlações positivas.
- Se latitude e longitude parecem carregar informação geográfica relevante.
- Se variáveis com correlação fraca ainda podem ser úteis quando combinadas com outras.

### Perguntas para responder no notebook

1. Quais features parecem mais relevantes para prever `MedHouseVal`?
2. Alguma correlação te surpreendeu?
3. Correlação fraca significa feature inútil?

!!! note "Limite da correlação"
    Correlação mede associação linear isolada.  
    Um modelo com múltiplas features pode extrair valor mesmo de variáveis que, sozinhas, parecem pouco informativas.

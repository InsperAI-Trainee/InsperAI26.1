# ✏️ Atividades

Coloque em prática o que foi visto em aula. Tente resolver cada exercício antes de olhar a dica.

---

## Atividade 1 — Explorando as Estatísticas

Rode `.describe()` no dataset e responda:

1. Qual variável tem o maior valor de `max` em relação ao seu `75%`? O que isso sugere?
2. A média (`mean`) de `AveRooms` é maior ou menor que a mediana (`50%`)? O que isso diz sobre a distribuição?
3. Qual coluna tem o menor desvio padrão (`std`) relativo à sua média?

??? tip "Dica"
    Para a questão 1, calcule a razão `max / 75%` para cada coluna. Razões muito altas indicam presença de outliers extremos.

---

## Atividade 2 — Investigando Outliers

Complete o código abaixo para identificar blocos com `AveRooms` acima do percentil 99:

```python
threshold = df["AveRooms"].quantile(___)

extremos = df[df["AveRooms"] > threshold]
print(f"Blocos com AveRooms no top 1%: {len(extremos)}")
extremos[["AveRooms", "AveBedrms", "Population", "MedHouseVal"]].head(10)
```

Após identificar, reflita: faz sentido um domicílio ter 50, 100 ou 141 cômodos?  
Que tipo de estabelecimento poderia gerar esses valores?

??? tip "Dica"
    Substitua `___` por `0.99`. Lembre que `quantile(0.99)` retorna o valor abaixo do qual estão 99% dos dados.

---

## Atividade 3 — Impacto do Tamanho do Conjunto de Teste

Treine dois modelos com diferentes proporções de `test_size` e compare os R²:

```python
for test_size in [0.1, 0.3]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=___, random_state=42
    )
    m = LinearRegression().fit(X_tr, y_tr)
    r2 = r2_score(y_te, m.predict(X_te))
    print(f"test_size={test_size} → R² Teste = {r2:.4f} "
          f"| Treino={len(X_tr):,} | Teste={len(X_te):,}")
```

Responda: o R² mudou muito? Por quê ele pode variar com o tamanho do conjunto de teste?

??? tip "Dica"
    Substitua `___` por `test_size`. Conjuntos de teste menores têm mais variância (o R² pode oscilar mais entre diferentes) `random_state`.

---

## Atividade 4 — Analisando os Coeficientes

Após treinar o modelo, imprima os coeficientes e responda:

1. Qual feature tem o **maior coeficiente positivo**? Isso faz sentido economicamente?
2. `Latitude` tem coeficiente positivo ou negativo? Por quê?
3. `AveBedrms` tem sinal esperado ou surpreendente? Pesquise o conceito de **multicolinearidade** para entender.

---

## Atividade 5 — Desafio

Filtre o dataset removendo:

- Blocos com `AveOccup > 20`
- Blocos com `MedHouseVal == 5.0` (preço censurado)

Retreine o modelo com os dados filtrados e compare o R² com o modelo original.

```python
df_limpo = df[
    (df["AveOccup"] <= ___) &
    (df["MedHouseVal"] < ___)
].copy()

print(f"Dataset original: {len(df):,} linhas")
print(f"Dataset limpo:    {len(df_limpo):,} linhas")

X2 = df_limpo.drop(columns=["MedHouseVal"])
y2 = df_limpo["MedHouseVal"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = LinearRegression().fit(X_tr2, y_tr2)
r2_limpo = r2_score(y_te2, model2.predict(X_te2))

print(f"\nR² (dados brutos): {r2_test:.4f}")
print(f"R² (dados limpos): {r2_limpo:.4f}")
```

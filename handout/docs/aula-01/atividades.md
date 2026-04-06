# Atividades

Estas atividades foram pensadas para serem feitas **depois** do notebook do assignment da aula 01 no **GitHub Classroom**.
Elas assumem que você já executou a prática principal e já tem, pelo menos, um modelo bruto e um modelo limpo para comparar.

Tente resolver cada exercício antes de olhar a dica.

---

## Atividade 1 — Revisando a Inspeção Inicial

Volte ao output de `.describe()` gerado no notebook e responda:

1. Quais colunas mais claramente sugerem presença de outliers?
2. O target `MedHouseVal` já dava pistas do teto artificial antes mesmo do gráfico? Onde?
3. `AveRooms` e `AveOccup` parecem simétricas ou assimétricas? O que isso sugere sobre médias e medianas?

??? tip "Dica"
    Compare `max`, `75%`, `mean` e `50%`. Valores máximos muito distantes e médias bem acima da mediana costumam ser sinais fortes de assimetria e extremos.

---

## Atividade 2 — Estendendo a Análise de Outliers

No notebook, a prática principal focou em `MedHouseVal == 5.0` e `AveOccup > 20`.
Agora investigue um terceiro sinal: valores extremos de `AveRooms`.

Complete o código abaixo:

```python
threshold = df["AveRooms"].quantile(___)

extremos = df[df["AveRooms"] > threshold]
print(f"Blocos com AveRooms no top 1%: {len(extremos)}")
extremos[["AveRooms", "AveBedrms", "Population", "MedHouseVal"]].head(10)
```

Depois responda:

1. Faz sentido um domicílio ter 50, 100 ou 141 cômodos?
2. Esses casos parecem mais próximos de erro de medição, uso institucional ou outra coisa?
3. Você incluiria esse filtro no modelo limpo da aula 01? Por quê?

??? tip "Dica"
    Substitua `___` por `0.99`. Lembre que `quantile(0.99)` retorna o valor abaixo do qual estão 99% dos dados.

---

## Atividade 3 — Sensibilidade ao Tamanho do Teste

Repita o treino com diferentes proporções de `test_size` e compare os resultados. Faça isso primeiro com os dados brutos e, se tiver tempo, repita com os dados limpos.

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
    Substitua `___` por `test_size`. Conjuntos de teste menores tendem a produzir métricas mais instáveis, porque cada amostra individual pesa mais no resultado final.

---

## Atividade 4 — Comparando Coeficientes Bruto vs Limpo

Monte uma tabela comparando os coeficientes do modelo bruto com os do modelo limpo.

Se você padronizou os nomes no notebook, pode usar algo próximo disto:

```python
comparacao = pd.DataFrame({
    "feature": X.columns,
    "coef_bruto": model_bruto.coef_,
    "coef_limpo": model_limpo.coef_,
})

comparacao["delta_abs"] = (
    comparacao["coef_limpo"] - comparacao["coef_bruto"]
).abs()

comparacao.sort_values("delta_abs", ascending=False).head(10)
```

Depois responda:

1. Qual feature tem o **maior coeficiente positivo**? Isso faz sentido economicamente?
2. Quais coeficientes mais mudaram depois da limpeza?
3. `AveBedrms` tem sinal esperado ou surpreendente? Pesquise o conceito de **multicolinearidade** para interpretar esse comportamento.

---

## Atividade 5 — Diagnóstico com Resíduos

No notebook, você comparou métricas agregadas. Agora olhe para os erros ponto a ponto do modelo limpo.

Complete:

```python
residuos = y_test_limpo - y_pred_test_limpo

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(residuos, bins=40)
axes[0].set_title("Distribuição dos resíduos")

axes[1].scatter(y_pred_test_limpo, residuos, alpha=0.2, s=10)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Resíduo vs valor predito")

plt.tight_layout()
plt.show()
```

Depois responda:

1. Os resíduos parecem centrados em zero?
2. Existe padrão visual claro de subestimação ou superestimação?
3. O gráfico sugere que a relação é bem capturada por um modelo linear simples?

# ⚠️ Outliers e Dados Suspeitos

Os gráficos da exploração já deixaram pistas. Agora vamos investigar formalmente os pontos que podem **prejudicar o treinamento do modelo**.

!!! danger "Por que isso importa?"
    Um modelo de regressão linear minimiza o erro quadrático. Outliers têm erros grandes — e ao elevar ao quadrado, eles dominam a função de custo. O modelo acaba "distorcendo" os coeficientes para tentar acomodar esses pontos anômalos.

---

## Problema 1 — Teto Artificial no Preço

O valor máximo de `MedHouseVal` é exatamente **5.0** (USD 500.000). Isso não é coincidência: o dataset original truncou todos os preços acima desse valor.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição do preço
ax = axes[0]
ax.hist(df["MedHouseVal"], bins=80, color="#4a90d9",
        edgecolor="white", linewidth=0.4)
ax.axvline(5.0, color="red", linestyle="--", linewidth=2,
           label="Teto artificial (USD 500k)")
ax.set_title("Distribuição do Preço\n(note o pico no limite máximo)",
             fontweight="bold")
ax.set_xlabel("Valor Mediano das Casas (100k USD)")
ax.set_ylabel("Frequência")
ax.legend()

# Média de cômodos
ax = axes[1]
ax.hist(df["AveRooms"], bins=100, color="#f5a623",
        edgecolor="white", linewidth=0.4)
p99 = df["AveRooms"].quantile(0.99)
ax.axvline(p99, color="red", linestyle="--", linewidth=2,
           label=f"P99 = {p99:.1f} cômodos")
ax.set_title("Média de Cômodos por Domicílio\n(outliers à direita)",
             fontweight="bold")
ax.set_xlabel("Média de Cômodos (AveRooms)")
ax.set_ylabel("Frequência")
ax.set_xlim(0, 15)
ax.legend()

plt.suptitle("Sinais de dados suspeitos",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

![Outliers de preço e cômodos](img/outliers_preco_comodos.png)

!!! warning "Consequência para o modelo"
    Todos os blocos com preço real acima de USD 500k aparecem como se valessem exatamente USD 500k. O modelo vai aprender que "casas muito boas custam USD 500k" — quando na verdade ele simplesmente não sabe o preço real delas.

---

## Problema 2 — Ocupação Anômala (Prisões, Hotéis, Hospitais)

`AveOccup` mede a **média de pessoas por domicílio** no bloco. Uma residência típica tem 2–4 moradores. Mas alguns blocos têm valores absurdos:

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df["AveOccup"], bins=200, color="#7b68ee",
        edgecolor="white", linewidth=0.3)
p99 = df[df["AveOccup"] <= 20]["AveOccup"].quantile(0.99)
ax.axvline(p99, color="red", linestyle="--", linewidth=2,
           label=f"P99 (normais) = {p99:.1f} pessoas/dom.")
ax.set_xlim(0, 20)
ax.set_title("Ocupação Média por Domicílio (AveOccup)\n"
             "Valores muito altos → blocos institucionais?",
             fontweight="bold")
ax.set_xlabel("Média de pessoas por domicílio")
ax.set_ylabel("Frequência")
ax.legend()
plt.tight_layout()
plt.show()

# Listando os casos extremos
suspeitos = df[df["AveOccup"] > 20].sort_values("AveOccup", ascending=False)
print(f"Blocos com AveOccup > 20: {len(suspeitos)}")
suspeitos[["Latitude", "Longitude", "AveOccup",
           "AveRooms", "Population", "MedHouseVal"]].head(10)
```

![Distribuição de ocupação com outliers](img/outliers_ocupacao.png)

!!! example "Exemplos reais de blocos suspeitos"
    | Tipo | AveOccup esperado | Por quê distorce? |
    |---|---|---|
    | **Presídio** | 500–1.000+ | Centenas de presos num "domicílio" |
    | **Hotel** | 100–500 | Quartos de hotel contam como domicílios |
    | **Hospital / Asilo** | 50–200 | Leitos por "unidade residencial" |
    | **Base militar** | 200–800 | Quartel conta como domicílio |

"""
Gera gráficos de latitude/longitude melhor formatados (mapas)
Dataset: California Housing Dataset
Saída: img/
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path

PALETTE = {
    "azul":     "#2D6BE4",
    "vermelho": "#E4452D",
    "verde":    "#2DBE6C",
    "cinza":    "#9BA3AF",
    "fundo":    "#F8F9FC",
    "texto":    "#1E1E2E",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["fundo"],
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#D0D4DC",
    "axes.labelcolor":   PALETTE["texto"],
    "xtick.color":       PALETTE["texto"],
    "ytick.color":       PALETTE["texto"],
    "text.color":        PALETTE["texto"],
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       150,
    "savefig.facecolor": PALETTE["fundo"],
})

OUT = Path("img")
OUT.mkdir(exist_ok=True)

# Carregar dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target


# ── 1. Mapa geográfico com preços ─────────────────────────────────────────────
def plot_mapa_geo():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter com tamanho proporcional à população e cor pelo preço
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseVal"],
        cmap="RdYlGn",  # Red-Yellow-Green para interpretabilidade (verde = caro)
        alpha=0.6,
        s=df["Population"] / 100,  # tamanho ∝ população
        linewidths=0,
        edgecolors="none",
    )

    # Colorbar com label melhor
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Valor mediano das casas\n(centenas de milhares USD)",
                   fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Título e labels
    ax.set_title(
        "Distribuição Geográfica dos Blocos Censitários da Califórnia\n"
        "(tamanho do ponto ∝ população  |  cor = preço mediano)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")

    # Grid discreto para referência
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    # Melhorar aspecto dos ticks
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "mapa_geo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ mapa_geo.png (melhorado)")


# ── 2. Mapa mostrando outliers ───────────────────────────────────────────────
def plot_mapa_suspeitos():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Usar Z-score para identificar outliers em preço
    from scipy import stats
    z_scores = np.abs(stats.zscore(df["MedHouseVal"]))
    eh_suspeito = z_scores > 2.5

    # Scatter para dados normais
    normais = df[~eh_suspeito]
    ax.scatter(
        normais["Longitude"],
        normais["Latitude"],
        c=normais["MedHouseVal"],
        cmap="Blues",
        alpha=0.5,
        s=normais["Population"] / 100,
        linewidths=0,
        label="Blocos normais",
        edgecolors="none",
    )

    # Scatter para dados suspeitos (outliers)
    suspeitos = df[eh_suspeito]
    ax.scatter(
        suspeitos["Longitude"],
        suspeitos["Latitude"],
        c=suspeitos["MedHouseVal"],
        cmap="Reds",
        alpha=0.8,
        s=suspeitos["Population"] / 100,
        linewidths=1.5,
        edgecolors=PALETTE["vermelho"],
        marker="D",  # diamante para destaque
        label=f"Outliers ({len(suspeitos)} blocos)",
    )

    # Título e labels
    ax.set_title(
        "Identificação de Outliers Geográficos\n"
        "(blocos com preços anormalmente altos ou baixos)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")

    # Grid e legenda
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    # Melhorar aspecto dos ticks
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "mapa_suspeitos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ mapa_suspeitos.png (melhorado)")


# ── 3. Heatmap 2D: correlação latitude/longitude com preço ──────────────────
def plot_heatmap_geo():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Criar bins 2D para visualizar densidade e preço médio
    lat_bins = 30
    lon_bins = 30

    lat_edges = np.linspace(df["Latitude"].min(), df["Latitude"].max(), lat_bins + 1)
    lon_edges = np.linspace(df["Longitude"].min(), df["Longitude"].max(), lon_bins + 1)

    heatmap = np.zeros((lat_bins, lon_bins))
    for i in range(lat_bins):
        for j in range(lon_bins):
            mask = (
                (df["Latitude"] >= lat_edges[i]) &
                (df["Latitude"] < lat_edges[i + 1]) &
                (df["Longitude"] >= lon_edges[j]) &
                (df["Longitude"] < lon_edges[j + 1])
            )
            if mask.sum() > 0:
                heatmap[i, j] = df[mask]["MedHouseVal"].mean()
            else:
                heatmap[i, j] = np.nan

    im = ax.imshow(
        heatmap[::-1],  # Inverter para latitude crescer de baixo para cima
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        cmap="RdYlGn",
        aspect="auto",
        origin="lower",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Preço mediano (centenas de milhares USD)",
                   fontsize=11, fontweight="bold")

    # Título e labels
    ax.set_title(
        "Mapa de Calor: Preços Medianos por Região Geográfica",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")

    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "mapa_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ mapa_heatmap.png (novo)")


# ── 4. Matriz de Correlação com valores positivos e negativos ─────────────────
def plot_correlacao_heatmap():
    fig, ax = plt.subplots(figsize=(11, 9))

    # Calcular matriz de correlação
    corr_matrix = df.corr()

    # Criar heatmap com paleta divergente (azul = negativo, vermelho = positivo)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Labels das variáveis
    labels = corr_matrix.columns
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Adicionar valores de correlação nas células
    for i in range(len(labels)):
        for j in range(len(labels)):
            corr_val = corr_matrix.iloc[i, j]
            # Cor do texto: branco se fundo escuro, preto se fundo claro
            text_color = "white" if abs(corr_val) > 0.5 else "black"
            ax.text(j, i, f"{corr_val:.2f}",
                   ha="center", va="center", color=text_color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Correlação de Pearson",
                   fontsize=11, fontweight="bold")

    # Título
    ax.set_title(
        "Matriz de Correlação — California Housing Dataset\n"
        "(vermelho = correlação positiva  |  azul = correlação negativa)",
        fontsize=13, fontweight="bold", pad=15
    )

    plt.tight_layout()
    plt.savefig(OUT / "correlacao_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ correlacao_heatmap.png (novo)")


# ── Execução ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Gerando gráficos de latitude/longitude...\n")
    plot_mapa_geo()
    plot_mapa_suspeitos()
    plot_heatmap_geo()
    plot_correlacao_heatmap()
    print("\nTodos os gráficos gerados em ./img/")

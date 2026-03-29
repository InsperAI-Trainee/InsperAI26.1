"""
Gera todos os gráficos da Aula 01 — Regressão Linear (Teoria)
Exemplo fio condutor: área do imóvel vs preço de venda
Saída: img/
"""

import numpy as np
import matplotlib.pyplot as plt
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
    "axes.facecolor":    PALETTE["fundo"],
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
rng = np.random.default_rng(42)

# Dados base — mesmos em todos os gráficos para consistência visual
AREA  = rng.uniform(40, 180, 40)
PRECO = 3500 * AREA + 80_000 + rng.normal(0, 40_000, 40)


# ── 1. Intuição: com e sem padrão ────────────────────────────────────────────
def plot_intuicao():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Sem correlação — área e preço embaralhados independentemente
    x1 = rng.uniform(40, 180, 40)
    y1 = rng.permutation(PRECO)  # preços sem relação com a área
    axes[0].scatter(x1, y1 / 1000, color=PALETTE["cinza"], alpha=0.7, s=55,
                    edgecolors="white", linewidth=0.5)
    axes[0].set_title("Sem padrão aparente", fontweight="bold", pad=10)
    axes[0].set_xlabel("Área (m²)")
    axes[0].set_ylabel("Preço (R$ mil)")

    # Com correlação — os dados reais
    axes[1].scatter(AREA, PRECO / 1000, color=PALETTE["azul"], alpha=0.75, s=55,
                    edgecolors="white", linewidth=0.5)
    axes[1].set_title("Com padrão: imóveis maiores tendem a custar mais", fontweight="bold", pad=10)
    axes[1].set_xlabel("Área (m²)")
    axes[1].set_ylabel("Preço (R$ mil)")

    fig.suptitle("O primeiro passo: existe um padrão nos dados?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "intuicao_dispersao.png")
    plt.close()
    print("✔ intuicao_dispersao.png")


# ── 2. Equação da reta ────────────────────────────────────────────────────────
def plot_equacao_reta():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.linspace(30, 190, 300)
    m, b = 3500, 80_000
    y = (m * x + b) / 1000

    ax.plot(x, y, color=PALETTE["azul"], linewidth=2.5,
            label=r"$\hat{y} = 3500x + 80000$")

    # Intercepto
    ax.scatter([0], [b / 1000], color=PALETTE["vermelho"], zorder=5, s=90, clip_on=False)
    ax.annotate("intercepto (b)\npreço base estimado\nquando área → 0",
                xy=(0, b / 1000), xytext=(38, 55),
                arrowprops=dict(arrowstyle="->", color=PALETTE["vermelho"]),
                color=PALETTE["vermelho"], fontsize=9)

    # Triângulo de inclinação
    x0, x1s = 80, 110
    y0, y1s = (m * x0 + b) / 1000, (m * x1s + b) / 1000
    ax.annotate("", xy=(x1s, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-", color=PALETTE["verde"], lw=1.5))
    ax.annotate("", xy=(x1s, y1s), xytext=(x1s, y0),
                arrowprops=dict(arrowstyle="-", color=PALETTE["verde"], lw=1.5))
    ax.text(95, y0 - 20, "Δx = 30 m²", ha="center", color=PALETTE["verde"], fontsize=9)
    ax.text(x1s + 3, (y0 + y1s) / 2, "Δy = R$105k", va="center",
            color=PALETTE["verde"], fontsize=9)
    ax.text(128, 390,
            "inclinação (m):\ncada m² adiciona\nR$ 3.500 ao preço",
            color=PALETTE["verde"], fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc=PALETTE["fundo"],
                      ec=PALETTE["verde"], lw=1))

    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Preço previsto (R$ mil)")
    ax.set_title(r"A equação da reta: $\hat{y} = mx + b$", fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 780)

    plt.tight_layout()
    plt.savefig(OUT / "equacao_reta.png")
    plt.close()
    print("✔ equacao_reta.png")


# ── 3. Retas ruins vs ajustada ───────────────────────────────────────────────
def plot_reta_ajuste():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    configs = [
        (8000,  0,       "Reta ruim\n(superestima tudo)"),
        (1000,  300_000, "Reta razoável\n(subestima imóveis grandes)"),
        (3500,  80_000,  "Reta ajustada\n(melhor equilíbrio)"),
    ]

    xline = np.linspace(35, 185, 200)
    for ax, (m, b, title) in zip(axes, configs):
        ax.scatter(AREA, PRECO / 1000, color=PALETTE["azul"], alpha=0.7,
                   s=45, edgecolors="white", linewidth=0.5)
        ax.plot(xline, (m * xline + b) / 1000, color=PALETTE["vermelho"], linewidth=2)
        ax.set_title(title, fontweight="bold", fontsize=9.5, pad=8)
        ax.set_xlabel("Área (m²)")
        ax.set_ylabel("Preço (R$ mil)")

    fig.suptitle("O modelo busca a reta que melhor representa os dados",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "reta_ajuste.png")
    plt.close()
    print("✔ reta_ajuste.png")


# ── 4. Resíduos ───────────────────────────────────────────────────────────────
def plot_residuos():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    area_s  = np.array([50, 65, 80, 95, 110, 130, 150, 170], dtype=float)
    preco_s = np.array([265, 290, 370, 410, 465, 545, 600, 680], dtype=float)
    m, b    = 3.5, 90  # em R$ mil
    pred_s  = m * area_s + b

    ax.scatter(area_s, preco_s, color=PALETTE["azul"], zorder=5, s=70,
               label="Preço real", edgecolors="white")
    xline = np.linspace(45, 175, 200)
    ax.plot(xline, m * xline + b, color=PALETTE["vermelho"], linewidth=2,
            label=r"Previsão ($\hat{y}$)")

    for xi, yr, yp in zip(area_s, preco_s, pred_s):
        ax.plot([xi, xi], [yr, yp], color=PALETTE["verde"],
                linewidth=1.5, linestyle="--", zorder=3)

    idx     = 4
    mid     = (preco_s[idx] + pred_s[idx]) / 2
    residuo = preco_s[idx] - pred_s[idx]
    ax.annotate(f"resíduo = R$ {residuo:.0f}k\n(real − previsto)",
                xy=(area_s[idx], mid), xytext=(area_s[idx] + 12, mid + 12),
                arrowprops=dict(arrowstyle="->", color=PALETTE["verde"]),
                color=PALETTE["verde"], fontsize=9)

    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Preço (R$ mil)")
    ax.set_title("Resíduos: o quanto o modelo erra em cada imóvel",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "residuos.png")
    plt.close()
    print("✔ residuos.png")


# ── 5. Curva MSE ──────────────────────────────────────────────────────────────
def plot_mse_curva():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    area_n  = AREA / 1000
    preco_n = PRECO / 1_000_000

    slopes   = np.linspace(0, 800, 400)
    mse_vals = [np.mean((preco_n - m * area_n) ** 2) for m in slopes]
    m_opt    = slopes[np.argmin(mse_vals)]

    axes[0].plot(slopes, mse_vals, color=PALETTE["azul"], linewidth=2.5)
    axes[0].axvline(m_opt, color=PALETTE["vermelho"], linestyle="--", linewidth=1.5,
                    label=f"mínimo ≈ m={m_opt:.0f}")
    axes[0].scatter([m_opt], [min(mse_vals)], color=PALETTE["vermelho"], zorder=5, s=80)
    axes[0].set_xlabel("Valor de m testado")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Custo (MSE) para cada inclinação testada",
                      fontweight="bold", pad=10)
    axes[0].legend(fontsize=9)

    xline = np.linspace(35, 185, 200)
    axes[1].scatter(AREA, PRECO / 1000, color=PALETTE["azul"], alpha=0.7,
                    s=50, edgecolors="white", linewidth=0.5)
    axes[1].plot(xline, (m_opt * 1000 * xline / 1000),
                 color=PALETTE["vermelho"], linewidth=2,
                 label="Melhor reta encontrada")
    axes[1].set_xlabel("Área (m²)")
    axes[1].set_ylabel("Preço (R$ mil)")
    axes[1].set_title("Reta com menor MSE", fontweight="bold", pad=10)
    axes[1].legend(fontsize=9)

    fig.suptitle("A função de custo guia o modelo até a melhor reta",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "mse_curva.png")
    plt.close()
    print("✔ mse_curva.png")


# ── 6. Avaliação R² e RMSE ───────────────────────────────────────────────────
def plot_avaliacao():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng2 = np.random.default_rng(7)

    configs = [
        (20_000, "Modelo com R² alto\n(erro pequeno, reta bem ajustada)"),
        (90_000, "Modelo com R² baixo\n(erro grande, reta mal ajustada)"),
    ]

    xline = np.linspace(35, 185, 200)
    for ax, (noise, title) in zip(axes, configs):
        area_i  = rng2.uniform(40, 180, 40)
        preco_i = 3500 * area_i + 80_000 + rng2.normal(0, noise, 40)
        m_fit   = np.polyfit(area_i, preco_i, 1)
        pred_i  = np.polyval(m_fit, area_i)

        ss_res = np.sum((preco_i - pred_i) ** 2)
        ss_tot = np.sum((preco_i - preco_i.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((preco_i - pred_i) ** 2)) / 1000

        ax.scatter(area_i, preco_i / 1000, color=PALETTE["azul"], alpha=0.7,
                   s=50, edgecolors="white", linewidth=0.5)
        ax.plot(xline, np.polyval(m_fit, xline) / 1000,
                color=PALETTE["vermelho"], linewidth=2)
        ax.set_title(f"{title}\nR² = {r2:.2f}  |  RMSE = R$ {rmse:.0f}k",
                     fontweight="bold", fontsize=9.5, pad=8)
        ax.set_xlabel("Área (m²)")
        ax.set_ylabel("Preço (R$ mil)")

    fig.suptitle("Avaliando a qualidade do modelo",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "avaliacao_modelo.png")
    plt.close()
    print("✔ avaliacao_modelo.png")


# ── Execução ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_intuicao()
    plot_equacao_reta()
    plot_reta_ajuste()
    plot_residuos()
    plot_mse_curva()
    plot_avaliacao()
    print("\nTodos os gráficos gerados em ./img/")
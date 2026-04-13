import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path

# ── Estilo global ──────────────────────────────────────────────────────────────
PALETTE = {
    "indigo":     "#3F51B5",
    "deep_orange":"#FF5722",
    "indigo_light":"#C5CAE9",
    "orange_light":"#FFCCBC",
    "bg":         "#FAFAFA",
    "text":       "#212121",
    "gray":       "#9E9E9E",
    "grid":       "#E0E0E0",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["gray"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUT = Path("/home/claude/imgs")
OUT.mkdir(exist_ok=True)

def save(name):
    path = OUT / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓  {path}")


# ── 1. Neurônio biológico (diagrama esquemático) ───────────────────────────────
def plot_neuronio():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # Dendritos (entradas)
    dendrite_ys = [0.8, 1.6, 2.4, 3.2]
    for y in dendrite_ys:
        ax.annotate("", xy=(3.2, 2.0), xytext=(1.0, y),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["indigo"], lw=1.5))
        ax.text(0.55, y, f"$x_{dendrite_ys.index(y)+1}$", fontsize=13,
                ha="center", va="center", color=PALETTE["indigo"])

    # Soma
    soma = plt.Circle((3.8, 2.0), 0.6, color=PALETTE["indigo"], zorder=5)
    ax.add_patch(soma)
    ax.text(3.8, 2.0, "Σ", fontsize=18, ha="center", va="center",
            color="white", fontweight="bold", zorder=6)

    # Axônio
    ax.annotate("", xy=(6.5, 2.0), xytext=(4.4, 2.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["deep_orange"], lw=2.5))

    # Função de ativação
    act = plt.Rectangle((6.5, 1.55), 1.2, 0.9, color=PALETTE["deep_orange"],
                         zorder=5, linewidth=0)
    ax.add_patch(act)
    ax.text(7.1, 2.0, "$f(z)$", fontsize=13, ha="center", va="center",
            color="white", fontweight="bold", zorder=6)

    # Saída
    ax.annotate("", xy=(9.2, 2.0), xytext=(7.7, 2.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["deep_orange"], lw=2.5))
    ax.text(9.5, 2.0, "$\\hat{y}$", fontsize=15, ha="center", va="center",
            color=PALETTE["deep_orange"])

    # Labels
    ax.text(1.0, 0.2, "Entradas\n(dendritos)", fontsize=9, ha="center",
            color=PALETTE["gray"])
    ax.text(3.8, 1.1, "Soma\n(corpo celular)", fontsize=9, ha="center",
            color=PALETTE["gray"])
    ax.text(7.1, 1.2, "Ativação\n(limiar)", fontsize=9, ha="center",
            color=PALETTE["gray"])
    ax.text(9.3, 1.4, "Saída\n(axônio)", fontsize=9, ha="center",
            color=PALETTE["gray"])

    # Pesos
    for i, y in enumerate(dendrite_ys):
        mid_x = (1.0 + 3.2) / 2 - 0.1
        mid_y = (y + 2.0) / 2
        ax.text(mid_x + 0.2, mid_y + 0.15, f"$w_{i+1}$", fontsize=10,
                color=PALETTE["indigo_light"], ha="center")

    ax.set_title("Modelo do Neurônio Artificial (Perceptron)", fontsize=14,
                 pad=12, color=PALETTE["text"])
    save("01_neuronio_artificial")


# ── 2. Funções de ativação ─────────────────────────────────────────────────────
def plot_ativacoes():
    z = np.linspace(-5, 5, 300)

    sigmoid = 1 / (1 + np.exp(-z))
    tanh    = np.tanh(z)
    relu    = np.maximum(0, z)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    configs = [
        (sigmoid, "Sigmoid  $\\sigma(z)$",   PALETTE["indigo"],     (-0.1, 1.1)),
        (tanh,    "Tanh  $\\tanh(z)$",        PALETTE["deep_orange"],(-1.2, 1.2)),
        (relu,    "ReLU  $\\max(0, z)$",      "#43A047",             (-0.3, 5.3)),
    ]

    for ax, (y, title, color, ylim) in zip(axes, configs):
        ax.plot(z, y, color=color, lw=2.5)
        ax.axhline(0, color=PALETTE["gray"], lw=0.8)
        ax.axvline(0, color=PALETTE["gray"], lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$z$", fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True)

    fig.suptitle("Funções de Ativação", fontsize=14, y=1.02)
    plt.tight_layout()
    save("02_funcoes_ativacao")


# ── 3. Problema do XOR ─────────────────────────────────────────────────────────
def plot_xor():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    points = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([0, 1, 1, 0])
    colors_pts = [PALETTE["indigo"] if l == 1 else PALETTE["deep_orange"] for l in labels]
    markers    = ["o" if l == 1 else "s" for l in labels]

    # Painel esquerdo: XOR (não separável linearmente)
    ax = axes[0]
    for (x1, x2), c, m in zip(points, colors_pts, markers):
        ax.scatter(x1, x2, color=c, marker=m, s=180, zorder=5, edgecolors="white", lw=1.5)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("$x_1$", fontsize=12); ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("XOR — não linearmente separável", fontsize=12)
    ax.grid(True)
    # Tentativa de reta (falha)
    x_line = np.linspace(-0.5, 1.5, 100)
    ax.plot(x_line, -x_line + 1, color=PALETTE["gray"], lw=1.5, linestyle="--",
            label="qualquer reta falha")
    ax.legend(fontsize=9)

    # Painel direito: AND (separável linearmente)
    labels_and = np.array([0, 0, 0, 1])
    colors_and = [PALETTE["indigo"] if l == 1 else PALETTE["deep_orange"] for l in labels_and]
    markers_and = ["o" if l == 1 else "s" for l in labels_and]
    ax2 = axes[1]
    for (x1, x2), c, m in zip(points, colors_and, markers_and):
        ax2.scatter(x1, x2, color=c, marker=m, s=180, zorder=5, edgecolors="white", lw=1.5)
    ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-0.5, 1.5)
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xlabel("$x_1$", fontsize=12); ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_title("AND — linearmente separável", fontsize=12)
    ax2.grid(True)
    ax2.plot(x_line, -x_line + 1.5, color="#43A047", lw=2, linestyle="-",
             label="reta separa perfeitamente")
    ax2.legend(fontsize=9)

    p1 = mpatches.Patch(color=PALETTE["indigo"],     label="Saída = 1")
    p2 = mpatches.Patch(color=PALETTE["deep_orange"],label="Saída = 0")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Limitação do Perceptron — o Problema do XOR", fontsize=14, y=1.02)
    plt.tight_layout()
    save("03_xor_problema")


# ── 4. Diagrama MLP simples ────────────────────────────────────────────────────
def plot_mlp():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    layers = {
        "Entrada\n(2 neurônios)":  (1.5, [2.0, 5.0]),
        "Camada Oculta\n(3 neurônios)": (4.5, [1.5, 3.5, 5.5]),
        "Saída\n(1 neurônio)":    (7.5, [3.5]),
    }
    layer_colors = [PALETTE["indigo"], PALETTE["deep_orange"], "#43A047"]

    node_positions = {}
    for (label, (x, ys)), color in zip(layers.items(), layer_colors):
        for y in ys:
            circle = plt.Circle((x, y), 0.35, color=color, zorder=5)
            ax.add_patch(circle)
            node_positions.setdefault(x, []).append(y)
        ax.text(x, 0.3, label, ha="center", fontsize=9, color=PALETTE["gray"])

    # Conexões
    xs = [1.5, 4.5, 7.5]
    for i in range(len(xs) - 1):
        for y1 in node_positions[xs[i]]:
            for y2 in node_positions[xs[i+1]]:
                ax.plot([xs[i]+0.35, xs[i+1]-0.35], [y1, y2],
                        color=PALETTE["indigo_light"], lw=0.8, zorder=1, alpha=0.7)

    ax.set_title("Rede Neural Multicamada (MLP) — Forward Pass", fontsize=14,
                 pad=12, color=PALETTE["text"])

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(
        layer_colors, ["Camada de Entrada", "Camada Oculta", "Camada de Saída"])]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    save("04_mlp_diagrama")


# ── 5. Gradient Descent — revisão visual ──────────────────────────────────────
def plot_gradient_descent():
    w = np.linspace(-3, 3, 300)
    loss = w**2 + 1  # parábola simples como superfície de perda

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(w, loss, color=PALETTE["indigo"], lw=2.5, label="$L(w)$")

    # Passos do gradient descent
    w_curr = 2.5
    lr = 0.3
    steps = []
    for _ in range(6):
        grad = 2 * w_curr
        w_next = w_curr - lr * grad
        steps.append((w_curr, w_curr**2 + 1, w_next, w_next**2 + 1))
        w_curr = w_next

    for i, (x0, y0, x1, y1) in enumerate(steps):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["deep_orange"], lw=1.8))
        ax.scatter(x0, y0, color=PALETTE["deep_orange"], s=60, zorder=5)

    ax.scatter(steps[-1][2], steps[-1][3], color="#43A047", s=100, zorder=6,
               label="Convergindo para mínimo")

    ax.set_xlabel("$w$ (peso)", fontsize=12)
    ax.set_ylabel("$L(w)$ (perda)", fontsize=12)
    ax.set_title("Gradient Descent — descendo a superfície de perda", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True)
    save("05_gradient_descent")


# ── 6. Backpropagation — fluxo do gradiente ───────────────────────────────────
def plot_backprop():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")

    boxes = [
        (1.2, 2.0, "Entrada\n$x$",     PALETTE["indigo"]),
        (3.5, 2.0, "Camada 1\n$h^{(1)}$", PALETTE["indigo"]),
        (5.9, 2.0, "Camada 2\n$h^{(2)}$", PALETTE["indigo"]),
        (8.3, 2.0, "Saída\n$\\hat{y}$", PALETTE["deep_orange"]),
        (9.9, 2.0, "Perda\n$L$",       "#C62828"),
    ]

    for x, y, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.55), 1.4, 1.1,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=4)

    # Forward (setas para a direita)
    xs = [b[0] for b in boxes]
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i+1] - 0.7, 2.2), xytext=(xs[i] + 0.7, 2.2),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["indigo"], lw=1.8))

    # Backward (setas para a esquerda)
    for i in range(len(xs) - 1, 0, -1):
        ax.annotate("", xy=(xs[i-1] + 0.7, 1.8), xytext=(xs[i] - 0.7, 1.8),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["deep_orange"], lw=1.8))

    ax.text(5.5, 3.1, "Forward pass  →", fontsize=10, color=PALETTE["indigo"], ha="center")
    ax.text(5.5, 0.8, "←  Backpropagation (gradientes)", fontsize=10,
            color=PALETTE["deep_orange"], ha="center")

    ax.set_title("Fluxo de informação: Forward Pass e Backpropagation", fontsize=13,
                 pad=10)
    save("06_backpropagation")


# ── Execução ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Gerando gráficos da Aula 04 — Redes Neurais...\n")
    plot_neuronio()
    plot_ativacoes()
    plot_xor()
    plot_mlp()
    plot_gradient_descent()
    plot_backprop()
    print(f"\nPronto! {len(list(OUT.glob('*.png')))} imagens salvas em {OUT}/")

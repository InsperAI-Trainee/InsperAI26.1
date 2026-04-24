"""
Gera os gráficos didáticos da Aula 06 — Visão Computacional e CNNs.

Saída: handout/docs/aula-06/imgs/*.png
Paleta Indigo / Deep Orange (Material Design) — herdada da Aula 04.

Uso:
    python handout/docs/aula-06/gerar_graficos.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.signal import convolve2d

# ── Estilo global ──────────────────────────────────────────────────────────────
PALETTE = {
    "indigo":       "#3F51B5",
    "deep_orange":  "#FF5722",
    "indigo_light": "#C5CAE9",
    "orange_light": "#FFCCBC",
    "green":        "#43A047",
    "bg":           "#FAFAFA",
    "text":         "#212121",
    "gray":         "#9E9E9E",
    "grid":         "#E0E0E0",
    "red":          "#C62828",
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

OUT = Path(__file__).parent / "imgs"
OUT.mkdir(exist_ok=True)


def save(name: str) -> None:
    path = OUT / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓  {path.name}")


def _draw_grid_with_values(ax, matrix, cell_colors=None, text_color="white",
                            fontsize=10, highlight=None, highlight_color=None):
    """Desenha uma matriz como uma grade de células com valores numéricos."""
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            color = cell_colors[i, j] if cell_colors is not None else PALETTE["indigo_light"]
            if highlight is not None and (i, j) in highlight:
                color = highlight_color or PALETTE["deep_orange"]
            rect = plt.Rectangle((j, rows - 1 - i), 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            val = matrix[i, j]
            if isinstance(val, (int, np.integer)):
                txt = str(val)
            else:
                txt = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            ax.text(j + 0.5, rows - 1 - i + 0.5, txt, ha="center", va="center",
                    color=text_color, fontsize=fontsize, fontweight="bold")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")


# ── 01. Imagem como matriz ─────────────────────────────────────────────────────
def plot_imagem_como_matriz():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Esquerda — grayscale 8x8 com valores
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, size=(6, 6))
    # Normalizar intensidade para a cor
    norm_gray = gray / 255.0
    cell_colors = np.empty(gray.shape, dtype=object)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # Indigo escala
            v = norm_gray[i, j]
            cell_colors[i, j] = (
                0.247 + (1 - 0.247) * (1 - v),
                0.318 + (1 - 0.318) * (1 - v),
                0.710 + (1 - 0.710) * (1 - v),
            )
    ax = axes[0]
    _draw_grid_with_values(ax, gray, cell_colors=cell_colors, text_color="white",
                            fontsize=11)
    ax.set_title("Imagem grayscale → matriz 2D\n(cada célula = intensidade de 0 a 255)",
                 fontsize=12, pad=12)

    # Direita — RGB como 3 matrizes empilhadas
    ax2 = axes[1]
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis("off")

    channel_colors = [PALETTE["red"], PALETTE["green"], PALETTE["indigo"]]
    channel_labels = ["R", "G", "B"]

    for k, (color, label) in enumerate(zip(channel_colors, channel_labels)):
        offset_x = 1.2 + k * 0.9
        offset_y = 4.8 - k * 0.9
        rect = plt.Rectangle((offset_x, offset_y), 4, 2.6, facecolor=color,
                              edgecolor="white", linewidth=2, alpha=0.85)
        ax2.add_patch(rect)
        ax2.text(offset_x + 0.25, offset_y + 2.25, label, fontsize=16,
                 fontweight="bold", color="white")

    ax2.annotate("", xy=(7.5, 4.5), xytext=(6.2, 4.5),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.8))
    ax2.text(8.9, 4.5, "tensor\n$H \\times W \\times 3$", fontsize=12,
             ha="center", va="center", color=PALETTE["text"])

    ax2.text(2.8, 1.4, "3 canais de cor", ha="center", fontsize=11,
             color=PALETTE["gray"])
    ax2.set_title("Imagem RGB → tensor 3D\n(3 canais: vermelho, verde, azul)",
                  fontsize=12, pad=12)

    fig.suptitle("Imagens como dados numéricos", fontsize=14, y=1.02)
    plt.tight_layout()
    save("01_imagem_como_matriz")


# ── 02. MLP achata imagem e perde estrutura ────────────────────────────────────
def plot_mlp_flatten_perde_estrutura():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    # Matriz 4×4 à esquerda
    for i in range(4):
        for j in range(4):
            val = (i * 4 + j + 1)
            rect = plt.Rectangle((0.5 + j * 0.7, 4.4 - i * 0.7), 0.7, 0.7,
                                  facecolor=PALETTE["indigo_light"],
                                  edgecolor="white", linewidth=1)
            ax.add_patch(rect)
            ax.text(0.85 + j * 0.7, 4.75 - i * 0.7, str(val),
                    ha="center", va="center", fontsize=9, color=PALETTE["text"])

    # Destacar vizinhos no 2D
    for (ii, jj) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        rect = plt.Rectangle((0.5 + jj * 0.7, 4.4 - ii * 0.7), 0.7, 0.7,
                              facecolor="none",
                              edgecolor=PALETTE["deep_orange"], linewidth=2.5)
        ax.add_patch(rect)

    ax.text(2.1, 5.7, "Imagem 2D — vizinhança preservada",
            ha="center", fontsize=11, color=PALETTE["text"])
    ax.text(2.1, 0.9, "Pixels próximos estão **próximos**",
            ha="center", fontsize=9.5, color=PALETTE["gray"])

    # Seta → flatten
    ax.annotate("", xy=(5.5, 3.3), xytext=(3.8, 3.3),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=2))
    ax.text(4.65, 3.65, "flatten()", fontsize=10, ha="center",
            color=PALETTE["text"], fontweight="bold")

    # Vetor 1×16 à direita
    for k in range(16):
        rect = plt.Rectangle((5.8 + k * 0.45, 3.0), 0.45, 0.65,
                              facecolor=PALETTE["indigo_light"],
                              edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(6.03 + k * 0.45, 3.32, str(k + 1),
                ha="center", va="center", fontsize=8, color=PALETTE["text"])
        # Destacar os mesmos (1,1),(1,2),(2,1),(2,2) que viraram índices 6,7,10,11 (1-indexed)
        if (k + 1) in [6, 7, 10, 11]:
            rect_hl = plt.Rectangle((5.8 + k * 0.45, 3.0), 0.45, 0.65,
                                     facecolor="none",
                                     edgecolor=PALETTE["deep_orange"], linewidth=2.5)
            ax.add_patch(rect_hl)

    ax.text(9.4, 4.3, "Vetor achatado — vizinhança perdida",
            ha="center", fontsize=11, color=PALETTE["text"])

    # Anotação destacando que (6,7) são vizinhos no vetor, mas (7,10) NÃO são no vetor
    ax.annotate("vizinhos no\n2D, separados\nno vetor",
                xy=(5.8 + 9 * 0.45 + 0.2, 3.0), xytext=(11.2, 1.6),
                fontsize=9, color=PALETTE["deep_orange"], ha="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["deep_orange"], lw=1.3))

    ax.set_title("Flatten quebra a estrutura 2D da imagem", fontsize=14, pad=16)
    plt.tight_layout()
    save("02_mlp_flatten_perde_estrutura")


# ── 03. Explosão de parâmetros em MLP ─────────────────────────────────────────
def plot_explosao_parametros():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    scenarios = [
        ("28×28\ngrayscale\n(MNIST)",   28 * 28,         128),
        ("64×64×3\ncolorido\n(Signs)",  64 * 64 * 3,     128),
        ("224×224×3\ncolorido\n(ImageNet)", 224 * 224 * 3, 128),
    ]

    labels = [s[0] for s in scenarios]
    params = [s[1] * s[2] + s[2] for s in scenarios]  # W + b da primeira camada
    colors = [PALETTE["indigo"], PALETTE["deep_orange"], PALETTE["red"]]

    bars = ax.bar(labels, params, color=colors, edgecolor="white", linewidth=2)

    for bar, p in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.02,
                f"{p:,}".replace(",", "."),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=PALETTE["text"])

    ax.set_ylabel("Parâmetros na 1ª camada densa\n(MLP com 128 neurônios)", fontsize=11)
    ax.set_title("Explosão de parâmetros ao alimentar um MLP com imagens",
                 fontsize=13, pad=14)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save("03_explosao_parametros")


# ── 04. Invariância por translação ─────────────────────────────────────────────
def plot_invariancia_translacao():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    shape_rows, shape_cols = 8, 8

    def make_frame(cx, cy):
        img = np.zeros((shape_rows, shape_cols))
        # Cruz 3x3 em (cx, cy)
        for di, dj in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            i, j = cx + di, cy + dj
            if 0 <= i < shape_rows and 0 <= j < shape_cols:
                img[i, j] = 1
        return img

    positions = [(1, 1), (3, 4), (5, 6)]
    captions = ["Posição A", "Posição B", "Posição C"]

    for ax, (cx, cy), cap in zip(axes, positions, captions):
        img = make_frame(cx, cy)
        ax.imshow(img, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(shape_cols))
        ax.set_yticks(range(shape_rows))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="white", linewidth=1)
        ax.set_title(cap, fontsize=11, pad=8)

    fig.suptitle("Mesmo objeto em posições diferentes — para um MLP, são entradas completamente distintas",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    save("04_invariancia_translacao")


# ── 05. Kernel deslizando sobre matriz ─────────────────────────────────────────
def plot_kernel_deslizando():
    # Entrada 5×5 com valores, kernel 3×3 deslizando → saída 3×3
    # Vamos mostrar 4 snapshots: posições (0,0), (0,1), (1,1), (2,2)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    rng = np.random.default_rng(7)
    inp = rng.integers(0, 9, size=(5, 5))
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])  # Sobel-like simples

    snapshots = [(0, 0), (0, 1), (1, 1), (2, 2)]

    for ax, (ki, kj) in zip(axes, snapshots):
        # Desenhar entrada 5×5
        ax.set_xlim(-0.2, 5.8)
        ax.set_ylim(-0.2, 5.3)
        ax.set_aspect("equal")
        ax.axis("off")

        for i in range(5):
            for j in range(5):
                inside_kernel = (ki <= i < ki + 3) and (kj <= j < kj + 3)
                color = PALETTE["orange_light"] if inside_kernel else PALETTE["indigo_light"]
                rect = plt.Rectangle((j, 4 - i), 1, 1, facecolor=color,
                                      edgecolor="white", linewidth=1.2)
                ax.add_patch(rect)
                ax.text(j + 0.5, 4 - i + 0.5, str(inp[i, j]),
                        ha="center", va="center", fontsize=9,
                        color=PALETTE["text"])

        # Contorno do kernel
        rect = plt.Rectangle((kj, 4 - ki - 2), 3, 3, facecolor="none",
                              edgecolor=PALETTE["deep_orange"], linewidth=2.8)
        ax.add_patch(rect)

        # Calcular o valor de saída
        window = inp[ki:ki+3, kj:kj+3]
        out_val = int((window * kernel).sum())
        ax.text(2.5, -0.1, f"saída[{ki},{kj}] = {out_val}",
                ha="center", fontsize=10, color=PALETTE["deep_orange"],
                fontweight="bold")
        ax.set_title(f"passo {snapshots.index((ki, kj)) + 1}",
                     fontsize=11, pad=4)

    fig.suptitle("Kernel 3×3 deslizando sobre entrada 5×5 (stride 1)",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    save("05_kernel_deslizando")


# ── 06. Kernel de edge detection (Sobel) ───────────────────────────────────────
def plot_kernel_edge_detection():
    # Gera uma imagem sintética com formas geométricas, aplica Sobel X e mostra o feature map
    img = np.zeros((40, 40))
    # quadrado à esquerda
    img[10:28, 6:18] = 1.0
    # círculo à direita
    yy, xx = np.ogrid[:40, :40]
    circle_mask = (yy - 18) ** 2 + (xx - 30) ** 2 <= 7 ** 2
    img[circle_mask] = 1.0

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    feature_map = convolve2d(img, sobel_x, mode="same", boundary="fill", fillvalue=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Imagem de entrada", fontsize=11)
    axes[0].axis("off")

    # Kernel ilustrado
    ax = axes[1]
    ax.set_xlim(-0.2, 3.3); ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(3):
        for j in range(3):
            v = sobel_x[i, j]
            color = PALETTE["indigo"] if v > 0 else (PALETTE["deep_orange"] if v < 0 else PALETTE["gray"])
            rect = plt.Rectangle((j, 2 - i), 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, 2 - i + 0.5, str(v),
                    ha="center", va="center", fontsize=13, color="white",
                    fontweight="bold")
    ax.set_title("Kernel (Sobel X)\ndetecta bordas verticais", fontsize=11, pad=6)

    axes[2].imshow(feature_map, cmap="RdBu_r", vmin=-4, vmax=4)
    axes[2].set_title("Feature map resultante\n(bordas verticais destacadas)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle("O kernel aprende a destacar padrões — aqui, bordas verticais",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    save("06_kernel_edge_detection")


# ── 07. Stride e padding ───────────────────────────────────────────────────────
def plot_stride_padding():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

    # Painel esquerdo: padding "valid" vs "same"
    ax = axes[0]
    ax.set_xlim(-0.5, 11); ax.set_ylim(-0.5, 6)
    ax.set_aspect("equal"); ax.axis("off")

    # Valid — 5×5 sem padding, saída menor
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((j, 4 - i), 1, 1,
                                  facecolor=PALETTE["indigo_light"],
                                  edgecolor="white", linewidth=1)
            ax.add_patch(rect)
    ax.text(2.5, 5.4, 'padding "valid"\nsaída = 3×3', ha="center",
            fontsize=10, color=PALETTE["text"])

    # Same — 5×5 com borda de zeros, saída mantém
    offset_x = 6
    for i in range(7):
        for j in range(7):
            is_border = (i == 0 or i == 6 or j == 0 or j == 6)
            color = PALETTE["orange_light"] if is_border else PALETTE["indigo_light"]
            rect = plt.Rectangle((offset_x + j, 5 - i), 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1)
            ax.add_patch(rect)
            if is_border:
                ax.text(offset_x + j + 0.5, 5 - i + 0.5, "0",
                        ha="center", va="center", fontsize=8,
                        color=PALETTE["deep_orange"], fontweight="bold")
    ax.text(offset_x + 3.5, 6.4, 'padding "same"\nsaída = 5×5',
            ha="center", fontsize=10, color=PALETTE["text"])

    ax.set_title('Padding: "valid" (sem borda) vs "same" (borda de zeros)',
                 fontsize=11.5, pad=12)

    # Painel direito: stride 1 vs stride 2 visual
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 11); ax2.set_ylim(-0.5, 6)
    ax2.set_aspect("equal"); ax2.axis("off")

    # Stride 1 — kernel avança de 1 em 1
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((j, 4 - i), 1, 1,
                                  facecolor=PALETTE["indigo_light"],
                                  edgecolor="white", linewidth=1)
            ax2.add_patch(rect)
    # Dois kernels 3×3 em posições próximas
    for dx, dy, col in [(0, 0, PALETTE["deep_orange"]),
                       (1, 0, PALETTE["green"])]:
        rect = plt.Rectangle((dx, 4 - dy - 2), 3, 3, facecolor="none",
                              edgecolor=col, linewidth=2.5)
        ax2.add_patch(rect)
    ax2.text(2.5, 5.4, "stride = 1\n(janelas se sobrepõem)",
             ha="center", fontsize=10, color=PALETTE["text"])

    # Stride 2 — kernel pula de 2 em 2
    offset_x = 6
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((offset_x + j, 4 - i), 1, 1,
                                  facecolor=PALETTE["indigo_light"],
                                  edgecolor="white", linewidth=1)
            ax2.add_patch(rect)
    for dx, dy, col in [(0, 0, PALETTE["deep_orange"]),
                       (2, 0, PALETTE["green"])]:
        rect = plt.Rectangle((offset_x + dx, 4 - dy - 2), 3, 3, facecolor="none",
                              edgecolor=col, linewidth=2.5)
        ax2.add_patch(rect)
    ax2.text(offset_x + 2.5, 5.4, "stride = 2\n(janelas pulam 2 colunas)",
             ha="center", fontsize=10, color=PALETTE["text"])

    ax2.set_title("Stride: passo com que o kernel avança",
                  fontsize=11.5, pad=12)

    plt.tight_layout()
    save("07_stride_padding")


# ── 08. Depth e múltiplos filtros ──────────────────────────────────────────────
def plot_depth_multiplos_filtros():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Painel 1: N kernels independentes → N feature maps
    ax = axes[0]
    ax.set_xlim(0, 11); ax.set_ylim(0, 7)
    ax.axis("off")

    # Entrada (cubo)
    input_color = PALETTE["indigo"]
    rect = plt.Rectangle((0.5, 2.5), 1.8, 2.5, facecolor=input_color,
                          edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(1.4, 1.9, "entrada\n$H \\times W \\times C$",
            ha="center", fontsize=10, color=PALETTE["text"])

    # 4 kernels
    kernel_colors = [PALETTE["deep_orange"], "#F9A825", PALETTE["green"], "#6A1B9A"]
    for k, col in enumerate(kernel_colors):
        y = 5.5 - k * 1.2
        rect = plt.Rectangle((4.0, y - 0.25), 0.5, 0.5, facecolor=col,
                              edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.annotate("", xy=(3.8, y), xytext=(2.4, 3.7),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1))
        # Feature map gerado
        fm = plt.Rectangle((6.3, y - 0.45), 1.4, 0.9, facecolor=col,
                            edgecolor="white", linewidth=1.5, alpha=0.7)
        ax.add_patch(fm)
        ax.annotate("", xy=(6.2, y), xytext=(4.6, y),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1.2))

    ax.text(4.25, 6.5, "N kernels", ha="center",
            fontsize=11, fontweight="bold", color=PALETTE["text"])
    ax.text(7.0, 6.5, "N feature\nmaps", ha="center",
            fontsize=11, fontweight="bold", color=PALETTE["text"])
    ax.text(8.5, 3.7, "→ profundidade\n= N",
            fontsize=10, color=PALETTE["deep_orange"], fontweight="bold")

    ax.set_title("Cada kernel produz 1 feature map\n(N kernels → profundidade N na saída)",
                 fontsize=11.5, pad=10)

    # Painel 2: um kernel em camada intermediária atua sobre todos os canais de entrada
    ax2 = axes[1]
    ax2.set_xlim(0, 11); ax2.set_ylim(0, 7)
    ax2.axis("off")

    # 3 canais empilhados
    for c, col in enumerate([PALETTE["red"], PALETTE["green"], PALETTE["indigo"]]):
        offset = c * 0.3
        rect = plt.Rectangle((1.0 + offset, 2.2 + offset), 2.2, 2.5, facecolor=col,
                              edgecolor="white", linewidth=2, alpha=0.75)
        ax2.add_patch(rect)
    ax2.text(2.1, 1.5, "entrada\n3 canais (RGB)",
             ha="center", fontsize=10, color=PALETTE["text"])

    # Kernel 3D tocando todos os canais
    for c, col in enumerate([PALETTE["deep_orange"], "#F9A825", "#F57F17"]):
        offset = c * 0.18
        rect = plt.Rectangle((5.6 + offset, 3.2 + offset), 1.1, 1.1, facecolor=col,
                              edgecolor="white", linewidth=2, alpha=0.9)
        ax2.add_patch(rect)
    ax2.text(6.2, 1.9, "1 kernel 3D\n$f \\times f \\times 3$",
             ha="center", fontsize=10, color=PALETTE["text"])

    # Seta do kernel para o feature map
    ax2.annotate("", xy=(9.3, 3.5), xytext=(7.1, 3.5),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1.8))

    # Feature map (1 único)
    fm = plt.Rectangle((9.3, 2.6), 1.3, 1.8, facecolor=PALETTE["deep_orange"],
                        edgecolor="white", linewidth=1.8, alpha=0.8)
    ax2.add_patch(fm)
    ax2.text(9.95, 1.9, "1 feature\nmap",
             ha="center", fontsize=10, color=PALETTE["text"])

    ax2.text(5.5, 6.3,
             "Um kernel em camada intermediária\nabrange **todos** os canais de entrada",
             ha="center", fontsize=11, color=PALETTE["text"])

    ax2.set_title("O kernel é, na verdade, um tensor 3D",
                  fontsize=11.5, pad=10)

    plt.tight_layout()
    save("08_depth_multiplos_filtros")


# ── 09. Max pooling 4×4 → 2×2 ──────────────────────────────────────────────────
def plot_max_pooling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    inp = np.array([[1, 3, 2, 4],
                    [5, 6, 1, 2],
                    [7, 2, 8, 3],
                    [1, 4, 5, 9]])

    # Colorir quadrantes
    quadrant_colors = [PALETTE["indigo_light"], PALETTE["orange_light"],
                       "#C8E6C9", "#F8BBD0"]

    ax = axes[0]
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_aspect("equal"); ax.axis("off")
    for qi in range(2):
        for qj in range(2):
            qc = quadrant_colors[qi * 2 + qj]
            for i_in in range(2):
                for j_in in range(2):
                    ii = qi * 2 + i_in
                    jj = qj * 2 + j_in
                    rect = plt.Rectangle((jj, 3 - ii), 1, 1, facecolor=qc,
                                          edgecolor="white", linewidth=1.5)
                    ax.add_patch(rect)
                    val = inp[ii, jj]
                    is_max = (val == inp[qi*2:qi*2+2, qj*2:qj*2+2].max())
                    ax.text(jj + 0.5, 3 - ii + 0.5, str(val),
                            ha="center", va="center", fontsize=13,
                            fontweight="bold" if is_max else "normal",
                            color=PALETTE["deep_orange"] if is_max else PALETTE["text"])
    ax.set_title("Entrada 4×4\n(valores em laranja = máximo do quadrante)",
                 fontsize=11, pad=10)

    # Saída 2×2
    ax2 = axes[1]
    ax2.set_xlim(0, 2.5); ax2.set_ylim(0, 2.5)
    ax2.set_aspect("equal"); ax2.axis("off")
    out = np.zeros((2, 2), dtype=int)
    for qi in range(2):
        for qj in range(2):
            out[qi, qj] = inp[qi*2:qi*2+2, qj*2:qj*2+2].max()

    for qi in range(2):
        for qj in range(2):
            qc = quadrant_colors[qi * 2 + qj]
            rect = plt.Rectangle((qj, 1 - qi), 1, 1, facecolor=qc,
                                  edgecolor="white", linewidth=1.5)
            ax2.add_patch(rect)
            ax2.text(qj + 0.5, 1 - qi + 0.5, str(out[qi, qj]),
                     ha="center", va="center", fontsize=18,
                     fontweight="bold", color=PALETTE["deep_orange"])
    ax2.set_title("Saída 2×2\n(max pool 2×2, stride 2)",
                  fontsize=11, pad=10)

    fig.suptitle("Max pooling — reduz dimensão mantendo a informação mais forte",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save("09_max_pooling")


# ── 10. Arquitetura CNN típica (diagrama horizontal) ───────────────────────────
def plot_arquitetura_cnn_tipica():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis("off")

    # Blocos: (x, w, h, cor, label_top, label_bottom)
    blocks = [
        (0.3,  1.8, 3.6, PALETTE["indigo"],      "Entrada",         "$64\\times64\\times3$"),
        (2.6,  1.4, 3.0, PALETTE["deep_orange"], "Conv2D(8)",       "$62\\times62\\times8$"),
        (4.6,  1.2, 2.4, PALETTE["green"],       "MaxPool 2×2",     "$31\\times31\\times8$"),
        (6.5,  1.0, 2.0, PALETTE["deep_orange"], "Conv2D(16)",      "$29\\times29\\times16$"),
        (8.2,  0.8, 1.6, PALETTE["green"],       "MaxPool 2×2",     "$14\\times14\\times16$"),
        (9.8,  0.35,3.2, PALETTE["gray"],        "Flatten",         "$3136$"),
        (10.9, 0.35,2.2, PALETTE["indigo"],      "Dense(64)",       "$64$"),
        (12.3, 0.35,1.4, PALETTE["indigo"],      "Dense(6)",        "$6$ classes"),
        (13.8, 0.35,1.0, PALETTE["red"],         "softmax",         "probs"),
    ]

    for x, w, h, color, label_top, label_bottom in blocks:
        y = (6 - h) / 2 + 0.2
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                              edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h + 0.25, label_top,
                ha="center", fontsize=9, fontweight="bold", color=PALETTE["text"])
        ax.text(x + w / 2, y - 0.35, label_bottom,
                ha="center", fontsize=8, color=PALETTE["gray"])

    # Setas entre blocos
    for i in range(len(blocks) - 1):
        x_a = blocks[i][0] + blocks[i][1]
        x_b = blocks[i + 1][0]
        ax.annotate("", xy=(x_b - 0.02, 3), xytext=(x_a + 0.02, 3),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1))

    ax.set_title("Arquitetura típica de uma CNN — exemplo para entrada 64×64×3 (Signs Dataset)",
                 fontsize=13, pad=14)
    plt.tight_layout()
    save("10_arquitetura_cnn_tipica")


# ── Execução ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Gerando gráficos da Aula 06 — Visão Computacional e CNNs...\n")
    plot_imagem_como_matriz()
    plot_mlp_flatten_perde_estrutura()
    plot_explosao_parametros()
    plot_invariancia_translacao()
    plot_kernel_deslizando()
    plot_kernel_edge_detection()
    plot_stride_padding()
    plot_depth_multiplos_filtros()
    plot_max_pooling()
    plot_arquitetura_cnn_tipica()
    n = len(list(OUT.glob("*.png")))
    print(f"\nPronto! {n} imagens salvas em {OUT}/")

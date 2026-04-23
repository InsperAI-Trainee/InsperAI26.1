"""
Gera os gráficos didáticos da Aula 07 — ResNet e Transfer Learning.

Saída: handout/docs/aula-07/imgs/*.png
Paleta Indigo / Deep Orange (Material Design) — herdada da Aula 04 e Aula 06.

Uso:
    python handout/docs/aula-07/gerar_graficos.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Estilo global ──────────────────────────────────────────────────────────────
PALETTE = {
    "indigo":       "#3F51B5",
    "deep_orange":  "#FF5722",
    "indigo_light": "#C5CAE9",
    "orange_light": "#FFCCBC",
    "green":        "#43A047",
    "green_light":  "#C8E6C9",
    "bg":           "#FAFAFA",
    "text":         "#212121",
    "gray":         "#9E9E9E",
    "gray_dark":    "#616161",
    "grid":         "#E0E0E0",
    "red":          "#C62828",
    "amber":        "#F9A825",
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


# ── 01. Degradation problem — redes mais profundas piorando ────────────────────
def plot_degradation_problem():
    fig, ax = plt.subplots(figsize=(10, 5.2))

    epochs = np.linspace(0, 60, 200)

    # Rede rasa (20 camadas) — converge razoavelmente bem
    err_20 = 0.25 + 0.22 * np.exp(-epochs / 15) + 0.008 * np.sin(epochs * 0.8)

    # Rede profunda sem atalhos (56 camadas) — estagna em valor maior
    err_56 = 0.32 + 0.18 * np.exp(-epochs / 22) + 0.012 * np.sin(epochs * 0.6)

    ax.plot(epochs, err_20, color=PALETTE["indigo"], lw=2.5,
            label="Plain-20 (20 camadas)")
    ax.plot(epochs, err_56, color=PALETTE["deep_orange"], lw=2.5,
            label="Plain-56 (56 camadas)")

    ax.fill_between(epochs, err_20, err_56, where=(err_56 > err_20),
                    color=PALETTE["deep_orange"], alpha=0.1)

    # Seta mostrando o "gap" inesperado
    ax.annotate("", xy=(55, 0.36), xytext=(55, 0.29),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["red"], lw=1.5))
    ax.text(57, 0.325, "gap inesperado:\nmais camadas,\nerro maior",
            fontsize=10, color=PALETTE["red"], va="center")

    ax.set_xlabel("Épocas de treino", fontsize=11)
    ax.set_ylabel("Erro de treino", fontsize=11)
    ax.set_title("Degradation problem — redes profundas sem atalhos\ntreinam PIOR que redes rasas",
                 fontsize=13, pad=12)
    ax.set_xlim(0, 75)
    ax.set_ylim(0.2, 0.55)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save("01_degradation_problem")


# ── 02. Bloco residual (identity) ──────────────────────────────────────────────
def plot_bloco_residual():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    # Caminho principal: x → conv → ReLU → conv → (+x) → ReLU → y
    boxes = [
        (1.0,  "x",         PALETTE["indigo"],      "entrada"),
        (3.0,  "Conv 3×3",  PALETTE["deep_orange"], "$F$"),
        (5.0,  "ReLU",      PALETTE["amber"],       ""),
        (7.0,  "Conv 3×3",  PALETTE["deep_orange"], "$F$"),
        (9.2,  "+",         PALETTE["green"],       "soma"),
        (10.8, "ReLU",      PALETTE["amber"],       ""),
    ]

    for x, label, color, sub in boxes:
        if label == "+":
            circle = plt.Circle((x, 2.8), 0.4, facecolor=color, edgecolor="white",
                                 linewidth=2, zorder=4)
            ax.add_patch(circle)
            ax.text(x, 2.8, "+", ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white", zorder=5)
        else:
            w = 1.1 if label in ("ReLU",) else 1.3
            if label == "x":
                w = 0.8
            rect = plt.Rectangle((x - w/2, 2.3), w, 1.0, facecolor=color,
                                  edgecolor="white", linewidth=2, zorder=4)
            ax.add_patch(rect)
            ax.text(x, 2.8, label, ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold", zorder=5)
        if sub:
            ax.text(x, 1.6, sub, ha="center", fontsize=9, color=PALETTE["gray"])

    # Setas principais entre blocos
    connections = [
        (1.4, 3.0 - 0.65),
        (3.0 + 0.65, 5.0 - 0.55),
        (5.0 + 0.55, 7.0 - 0.65),
        (7.0 + 0.65, 9.2 - 0.4),
        (9.2 + 0.4, 10.8 - 0.55),
    ]
    for x_a, x_b in connections:
        ax.annotate("", xy=(x_b, 2.8), xytext=(x_a, 2.8),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

    # Skip connection (atalho)
    arc_x = np.linspace(1.0, 9.2, 60)
    arc_y = 2.8 + 1.4 * np.sin(np.pi * (arc_x - 1.0) / (9.2 - 1.0))
    ax.plot(arc_x, arc_y, color=PALETTE["red"], lw=2.8, zorder=3)
    ax.annotate("", xy=(9.0, 3.15), xytext=(8.7, 3.25),
                arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=2.8))

    ax.text(5.1, 4.7, "atalho (skip connection) — copia $x$ e soma lá na frente",
            fontsize=11, color=PALETTE["red"], ha="center", fontweight="bold")

    # Saída
    ax.annotate("", xy=(11.8, 2.8), xytext=(11.35, 2.8),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))
    ax.text(11.9, 2.8, "$y$", ha="left", va="center", fontsize=13,
            color=PALETTE["indigo"], fontweight="bold")

    # Equação
    ax.text(6.0, 0.6, r"$y = \mathrm{ReLU}(F(x) + x)$", fontsize=14,
            ha="center", color=PALETTE["text"])

    ax.set_title("Bloco residual (identity block) — a rede aprende o resíduo $F(x)$",
                 fontsize=13, pad=12)
    plt.tight_layout()
    save("02_bloco_residual")


# ── 03. Plain vs ResNet ────────────────────────────────────────────────────────
def plot_plain_vs_resnet():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def draw_stack(ax, title, with_skips=False):
        ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
        blocks_y = [6.2, 4.6, 3.0, 1.4]
        labels = ["Conv + ReLU", "Conv + ReLU", "Conv + ReLU", "Conv + ReLU"]
        for y, label in zip(blocks_y, labels):
            rect = plt.Rectangle((1.5, y - 0.4), 3, 0.8, facecolor=PALETTE["deep_orange"],
                                  edgecolor="white", linewidth=2, zorder=4)
            ax.add_patch(rect)
            ax.text(3, y, label, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold", zorder=5)

        # setas forward
        for i in range(len(blocks_y) - 1):
            y_a = blocks_y[i] - 0.4
            y_b = blocks_y[i + 1] + 0.4
            ax.annotate("", xy=(3, y_b), xytext=(3, y_a),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

        # Entrada e saída
        ax.text(3, 7.6, "entrada", ha="center", fontsize=10, color=PALETTE["indigo"],
                fontweight="bold")
        ax.annotate("", xy=(3, 6.6), xytext=(3, 7.4),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))
        ax.text(3, 0.6, "saída", ha="center", fontsize=10, color=PALETTE["indigo"],
                fontweight="bold")
        ax.annotate("", xy=(3, 0.8), xytext=(3, 1.0),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

        # Atalhos (se with_skips)
        if with_skips:
            for i in range(len(blocks_y) - 1):
                y_top = blocks_y[i] + 0.4
                y_bot = blocks_y[i + 1] - 0.4
                arc_x = np.linspace(4.7, 5.8, 30)
                arc_y = np.linspace(y_top, y_bot, 30)
                ax.plot([4.7, 5.4, 5.4, 4.7],
                        [y_top, y_top, y_bot, y_bot],
                        color=PALETTE["red"], lw=2.5)
                ax.annotate("", xy=(4.7, y_bot), xytext=(4.9, y_bot),
                            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=2.5))
            ax.text(5.5, 4.0, "atalhos", fontsize=10, color=PALETTE["red"],
                    rotation=90, ha="center", va="center", fontweight="bold")

        ax.set_title(title, fontsize=12.5, pad=8)

    draw_stack(axes[0], "Plain network — convs empilhadas")
    draw_stack(axes[1], "ResNet — convs + atalhos", with_skips=True)

    fig.suptitle("A mesma pilha, com e sem atalhos — só a ResNet treina bem quando fica profunda",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save("03_plain_vs_resnet")


# ── 04. Transfer learning — conceito ───────────────────────────────────────────
def plot_transfer_learning_conceito():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    # Modelo original (ImageNet)
    rect_orig = plt.Rectangle((0.8, 1.6), 5.0, 2.8,
                               facecolor=PALETTE["indigo"],
                               edgecolor="white", linewidth=2)
    ax.add_patch(rect_orig)
    ax.text(3.3, 4.7, "Modelo pré-treinado em ImageNet\n(MobileNetV2, ResNet50, VGG, ...)",
            ha="center", fontsize=10, color=PALETTE["text"], fontweight="bold")

    # Head original (vai ser removido)
    rect_head_orig = plt.Rectangle((4.3, 2.2), 1.1, 1.6,
                                    facecolor=PALETTE["red"],
                                    edgecolor="white", linewidth=2, alpha=0.5)
    ax.add_patch(rect_head_orig)
    ax.text(4.85, 3.0, "head\n1000 classes\n(ImageNet)",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(4.85, 1.3, "— remover —", ha="center", fontsize=9,
            color=PALETTE["red"], fontweight="bold")

    ax.text(3.0, 0.7, "features ricas, aprendidas em milhões de imagens",
            ha="center", fontsize=9, color=PALETTE["gray"], style="italic")

    # Seta
    ax.annotate("", xy=(7.5, 3.0), xytext=(6.0, 3.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=2))
    ax.text(6.75, 3.4, "reaproveitar\nbase", ha="center", fontsize=9,
            color=PALETTE["text"])

    # Modelo novo com head novo
    rect_base = plt.Rectangle((7.8, 1.6), 4.2, 2.8,
                               facecolor=PALETTE["indigo"],
                               edgecolor="white", linewidth=2)
    ax.add_patch(rect_base)
    ax.text(9.9, 4.7, "Base reutilizada\n(pesos do ImageNet)",
            ha="center", fontsize=10, color=PALETTE["text"], fontweight="bold")

    # Head novo
    rect_head_novo = plt.Rectangle((11.0, 2.2), 1.1, 1.6,
                                    facecolor=PALETTE["green"],
                                    edgecolor="white", linewidth=2)
    ax.add_patch(rect_head_novo)
    ax.text(11.55, 3.0, "head\nnovo\n(seu problema)",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(11.55, 1.3, "+ treinar", ha="center", fontsize=9,
            color=PALETTE["green"], fontweight="bold")

    ax.text(10.0, 0.7, "sua tarefa (poucas imagens)",
            ha="center", fontsize=9, color=PALETTE["gray"], style="italic")

    ax.set_title("Transfer learning — reaproveita a base, troca só a cabeça",
                 fontsize=13, pad=12)
    plt.tight_layout()
    save("04_transfer_learning_conceito")


# ── 05. Feature extraction vs Fine-tuning ──────────────────────────────────────
def plot_feature_extraction_vs_fine_tuning():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def draw_model(ax, title, fine_tune_last_k=0):
        ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")

        # 5 blocos convolucionais (base) + 1 head
        base_labels = ["Conv block 1", "Conv block 2", "Conv block 3",
                       "Conv block 4", "Conv block 5"]
        n_base = len(base_labels)

        for idx, label in enumerate(base_labels):
            y = 6.8 - idx * 1.0
            # Quantos dos últimos blocos estão destravados?
            is_trainable = idx >= n_base - fine_tune_last_k
            color = PALETTE["deep_orange"] if is_trainable else PALETTE["gray"]
            rect = plt.Rectangle((1.5, y - 0.35), 3, 0.7, facecolor=color,
                                  edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(3, y, label, ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold")
            # Ícone de cadeado ou engrenagem
            status = "■ treinável" if is_trainable else "■ congelado"
            stat_color = PALETTE["deep_orange"] if is_trainable else PALETTE["gray_dark"]
            ax.text(5.2, y, status, ha="left", va="center", fontsize=8.5,
                    color=stat_color)

        # Head (sempre treinável)
        rect_head = plt.Rectangle((1.5, 1.1), 3, 0.7, facecolor=PALETTE["green"],
                                   edgecolor="white", linewidth=2)
        ax.add_patch(rect_head)
        ax.text(3, 1.45, "Head novo (Dense + sigmoid)",
                ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        ax.text(5.2, 1.45, "■ treinável", ha="left", va="center", fontsize=8.5,
                color=PALETTE["green"])

        ax.text(3, 7.55, "entrada", ha="center", fontsize=9, color=PALETTE["indigo"],
                fontweight="bold")
        ax.text(3, 0.55, "saída (prob)", ha="center", fontsize=9, color=PALETTE["indigo"],
                fontweight="bold")

        ax.set_title(title, fontsize=12.5, pad=8)

    draw_model(axes[0], "Feature extraction\n(base toda congelada)", fine_tune_last_k=0)
    draw_model(axes[1], "Fine-tuning\n(últimos blocos destravados)", fine_tune_last_k=2)

    plt.tight_layout()
    save("05_feature_extraction_vs_fine_tuning")


# ── 06. Matriz 2×2: quando usar cada estratégia ────────────────────────────────
def plot_matriz_quando_usar():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    # Títulos dos eixos
    ax.text(5, 6.5, "Similaridade com ImageNet →",
            ha="center", fontsize=11, color=PALETTE["text"], fontweight="bold")
    ax.text(0.25, 3.25, "Tamanho do seu dataset →", rotation=90,
            va="center", ha="center", fontsize=11, color=PALETTE["text"],
            fontweight="bold")

    # Labels das colunas
    ax.text(3.0, 6.0, "Baixa", ha="center", fontsize=10, color=PALETTE["gray_dark"])
    ax.text(7.0, 6.0, "Alta", ha="center", fontsize=10, color=PALETTE["gray_dark"])

    # Labels das linhas
    ax.text(0.8, 4.4, "Grande", ha="center", fontsize=10, color=PALETTE["gray_dark"],
            rotation=90)
    ax.text(0.8, 1.9, "Pequeno", ha="center", fontsize=10, color=PALETTE["gray_dark"],
            rotation=90)

    cells = [
        # (x, y, w, h, fill_color, title, body)
        (1.4, 3.3, 3.6, 2.3, PALETTE["indigo_light"],
         "Fine-tuning agressivo",
         "Você tem dados para suportar.\nDestrave quase tudo e treine\ncom lr baixo."),
        (5.2, 3.3, 3.6, 2.3, PALETTE["green_light"],
         "Fine-tuning parcial",
         "Dataset grande + features\njá úteis. Destrave as últimas\ncamadas."),
        (1.4, 0.6, 3.6, 2.3, PALETTE["orange_light"],
         "Cuidado",
         "Dataset pequeno + dados\nmuito diferentes. Considere\nobter mais dados."),
        (5.2, 0.6, 3.6, 2.3, PALETTE["green_light"],
         "Feature extraction",
         "Pouco dado + domínio\nsemelhante. Congele tudo,\ntreine só o head.\n(default recomendado)"),
    ]

    for (x, y, w, h, color, title, body) in cells:
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                              edgecolor=PALETTE["gray"], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center",
                fontsize=11, color=PALETTE["text"], fontweight="bold")
        ax.text(x + w/2, y + h/2 - 0.25, body, ha="center", va="center",
                fontsize=9, color=PALETTE["text"])

    ax.set_title("Qual estratégia escolher? — tamanho do dataset × similaridade",
                 fontsize=12.5, pad=12)
    plt.tight_layout()
    save("06_matriz_quando_usar")


# ── 07. Pipeline MobileNetV2 ───────────────────────────────────────────────────
def plot_mobilenet_pipeline():
    fig, ax = plt.subplots(figsize=(17, 4.2))
    ax.set_xlim(0, 20); ax.set_ylim(0, 5); ax.axis("off")

    blocks = [
        (0.3,  2.2, "Imagem RGB",        PALETTE["indigo"],      "$224\\times224\\times3$"),
        (2.9,  2.4, "preprocess_input",  PALETTE["amber"],       "escala p/ $[-1, 1]$"),
        (5.7,  2.7, "MobileNetV2 base",  PALETTE["gray"],        "congelada — weights='imagenet'"),
        (8.8,  3.2, "GlobalAvgPooling2D", PALETTE["green"],       "pooling final"),
        (12.4, 2.6, "Dense(1, sigmoid)", PALETTE["deep_orange"], "probabilidade"),
        (15.4, 2.0, "saída",             PALETTE["red"],         "alpaca\nou não"),
    ]

    for x, w, label, color, sub in blocks:
        h = 2.0
        rect = plt.Rectangle((x, 1.3), w, h, facecolor=color,
                              edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, 2.6, label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(x + w/2, 0.9, sub, ha="center", va="top", fontsize=9,
                color=PALETTE["gray_dark"])

    # Setas
    for i in range(len(blocks) - 1):
        x_a = blocks[i][0] + blocks[i][1]
        x_b = blocks[i + 1][0]
        ax.annotate("", xy=(x_b - 0.05, 2.3), xytext=(x_a + 0.05, 2.3),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

    ax.set_title("Pipeline completo — MobileNetV2 como extrator de features + head binário",
                 fontsize=12.5, pad=12)
    plt.tight_layout()
    save("07_mobilenet_pipeline")


# ── 08. Hierarquia de features — rasas / médias / profundas ────────────────────
def plot_hierarquia_features():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    panels = [
        {
            "ax": axes[0],
            "title": "Camadas rasas",
            "subtitle": "bordas, cores, gradientes",
            "color": PALETTE["indigo_light"],
            "border": PALETTE["indigo"],
            "kind": "edges",
        },
        {
            "ax": axes[1],
            "title": "Camadas médias",
            "subtitle": "texturas, padrões, partes",
            "color": PALETTE["orange_light"],
            "border": PALETTE["deep_orange"],
            "kind": "textures",
        },
        {
            "ax": axes[2],
            "title": "Camadas profundas",
            "subtitle": "objetos, conceitos inteiros",
            "color": PALETTE["green_light"],
            "border": PALETTE["green"],
            "kind": "objects",
        },
    ]

    rng = np.random.default_rng(7)

    for p in panels:
        ax = p["ax"]
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

        # Fundo
        bg = plt.Rectangle((0.3, 0.3), 9.4, 9.4, facecolor=p["color"],
                            edgecolor=p["border"], linewidth=2)
        ax.add_patch(bg)

        if p["kind"] == "edges":
            # Linhas em diversas orientações — detectores de bordas
            for _ in range(18):
                theta = rng.uniform(0, np.pi)
                cx, cy = rng.uniform(1.5, 8.5, size=2)
                length = rng.uniform(1.2, 2.2)
                dx, dy = np.cos(theta) * length / 2, np.sin(theta) * length / 2
                ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                        color=p["border"], lw=2.3, solid_capstyle="round", alpha=0.85)
        elif p["kind"] == "textures":
            # Grades / padrões repetitivos
            for gx in np.linspace(1.2, 8.8, 9):
                ax.plot([gx, gx], [1.3, 8.7], color=p["border"], lw=1.2, alpha=0.55)
            for gy in np.linspace(1.3, 8.7, 9):
                ax.plot([1.2, 8.8], [gy, gy], color=p["border"], lw=1.2, alpha=0.55)
            # Pontos em cima — textura tipo "tecido"
            for _ in range(22):
                cx, cy = rng.uniform(1.5, 8.5, size=2)
                ax.plot(cx, cy, "o", color=p["border"], ms=5, alpha=0.75)
        elif p["kind"] == "objects":
            # Silhueta simplificada (olhos + corpo) — "rosto" abstrato
            body = plt.Circle((5, 4.4), 2.4, facecolor=p["border"],
                               edgecolor="white", linewidth=2, alpha=0.85)
            ax.add_patch(body)
            eye_l = plt.Circle((4.1, 4.9), 0.42, facecolor="white", zorder=5)
            eye_r = plt.Circle((5.9, 4.9), 0.42, facecolor="white", zorder=5)
            ax.add_patch(eye_l); ax.add_patch(eye_r)
            ax.plot([4.1, 5.9], [3.6, 3.6], color="white", lw=2.2,
                    solid_capstyle="round", zorder=5)
            # Orelhas
            ax.add_patch(plt.Circle((3.2, 6.8), 0.7, facecolor=p["border"],
                                     edgecolor="white", linewidth=2, alpha=0.85))
            ax.add_patch(plt.Circle((6.8, 6.8), 0.7, facecolor=p["border"],
                                     edgecolor="white", linewidth=2, alpha=0.85))
            ax.text(5, 1.2, "objeto reconhecível", ha="center", fontsize=9,
                    color=PALETTE["text"], style="italic")

        ax.set_title(p["title"], fontsize=12.5, fontweight="bold",
                     color=p["border"], pad=8)
        ax.text(5, -0.4, p["subtitle"], ha="center", fontsize=10,
                color=PALETTE["gray_dark"], transform=ax.transData)

    fig.suptitle("Hierarquia de features em uma CNN — do concreto (bordas) ao abstrato (objetos)",
                 fontsize=13, y=1.02)
    # Texto abaixo explicando a implicação
    fig.text(0.5, -0.05,
             "Camadas rasas são genéricas e reutilizáveis entre domínios — é por isso que transfer learning funciona.",
             ha="center", fontsize=10.5, color=PALETTE["text"], style="italic")

    plt.tight_layout()
    save("08_hierarquia_features")


# ── 09. Timeline das arquiteturas de CNN ───────────────────────────────────────
def plot_timeline_arquiteturas():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    # (ano, profundidade, top-5 acc ImageNet %, nome, cor)
    # Valores aproximados — finalidade didática.
    archs = [
        (1998,   5,   None, "LeNet-5",              PALETTE["gray"]),
        (2012,   8,   84.7, "AlexNet",              PALETTE["indigo"]),
        (2014,  19,   92.7, "VGG-19",               PALETTE["deep_orange"]),
        (2014,  22,   93.3, "Inception (GoogLeNet)", PALETTE["amber"]),
        (2015, 152,   96.4, "ResNet-152",           PALETTE["green"]),
    ]

    # Marcadores
    for (year, depth, acc, name, color) in archs:
        size = 240 if acc is None else 260 + (acc - 85) * 30
        ax.scatter(year, depth, s=max(size, 180), color=color,
                   edgecolors="white", linewidths=2, zorder=5, alpha=0.92)

    # Linha de tendência — ordenada por ano
    years = [a[0] for a in archs]
    depths = [a[1] for a in archs]
    ax.plot(years, depths, color=PALETTE["gray"], lw=1.3,
            linestyle="--", alpha=0.5, zorder=1)

    # Callouts: posicionados manualmente pra não colidir.
    # Cada callout ((x_texto, y_texto), (x_marcador, y_marcador))
    callouts = [
        # LeNet-5 — à direita do ponto, mesmo nível
        ((2001.5,   5), (1998,   5), "LeNet-5\n(sem ImageNet)", PALETTE["gray"]),
        # AlexNet — acima-à-direita
        ((2009.2,  35), (2012,   8), "AlexNet\nTop-5: 84.7%",   PALETTE["indigo"]),
        # VGG-19 — abaixo-à-esquerda
        ((2010.8, 55), (2014,  19), "VGG-19\nTop-5: 92.7%",     PALETTE["deep_orange"]),
        # Inception — abaixo-à-direita
        ((2016.5, 40), (2014,  22), "Inception\n(GoogLeNet)\nTop-5: 93.3%", PALETTE["amber"]),
        # ResNet-152 — à esquerda do ponto
        ((2011.8, 150), (2015, 152), "ResNet-152\nTop-5: 96.4%",  PALETTE["green"]),
    ]

    for ((tx, ty), (mx, my), text, color) in callouts:
        ax.annotate(text,
                    xy=(mx, my), xytext=(tx, ty),
                    fontsize=10, fontweight="bold", color=color,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=color,
                                    lw=1, alpha=0.6,
                                    connectionstyle="arc3,rad=0"),
                    zorder=4)

    # Destaque do salto ResNet
    ax.annotate("",
                xy=(2015, 148), xytext=(2014.1, 28),
                arrowprops=dict(arrowstyle="->", color=PALETTE["red"],
                                 lw=1.8, connectionstyle="arc3,rad=-0.3"),
                zorder=2)
    ax.text(2013.2, 95, "salto destravado\npela skip connection",
            fontsize=10, color=PALETTE["red"], ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["bg"],
                      edgecolor=PALETTE["red"], linewidth=1, alpha=0.9))

    ax.set_xlabel("Ano", fontsize=11)
    ax.set_ylabel("Profundidade (nº de camadas)", fontsize=11)
    ax.set_title("Linha do tempo das CNNs — 1998 → 2015 (tamanho do ponto ~ Top-5 acc)",
                 fontsize=13, pad=12)
    ax.set_xlim(1996, 2019)
    ax.set_ylim(-15, 180)
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save("09_timeline_arquiteturas")


# ── 10. Trade-off: acurácia × tamanho ──────────────────────────────────────────
def plot_tradeoff_acuracia_vs_tamanho():
    fig, ax = plt.subplots(figsize=(11, 6))

    # (params em milhões, Top-1 acc ImageNet %, nome, cor, destaque?)
    models = [
        ( 3.5, 71.8, "MobileNetV2",  PALETTE["deep_orange"], True),
        ( 5.3, 74.4, "MobileNetV3-L", PALETTE["amber"],       False),
        ( 6.8, 78.0, "EfficientNet-B0", PALETTE["green"],     False),
        (25.6, 76.0, "ResNet-50",    PALETTE["indigo"],      True),
        (44.5, 77.4, "ResNet-101",   PALETTE["indigo_light"], False),
        (138.0, 72.0, "VGG-16",      PALETTE["gray"],        False),
    ]

    for (params, acc, name, color, highlight) in models:
        size = 400 if highlight else 230
        edge = PALETTE["text"] if highlight else "white"
        ewidth = 2.5 if highlight else 2
        ax.scatter(params, acc, s=size, color=color, edgecolors=edge,
                   linewidths=ewidth, zorder=5, alpha=0.92)

        # Offsets escolhidos pra não sair da área visível
        if name == "MobileNetV2":
            dx, dy, ha = 14, 8, "left"
        elif name == "MobileNetV3-L":
            dx, dy, ha = 14, -4, "left"
        elif name == "EfficientNet-B0":
            dx, dy, ha = 14, 0, "left"
        elif name == "ResNet-50":
            dx, dy, ha = 0, 18, "center"
        elif name == "ResNet-101":
            dx, dy, ha = 0, 18, "center"
        elif name == "VGG-16":
            dx, dy, ha = -14, 0, "right"
        else:
            dx, dy, ha = 10, 10, "left"

        ax.annotate(name, (params, acc),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=10,
                    fontweight="bold" if highlight else "normal",
                    color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Parâmetros (milhões, escala log)", fontsize=11)
    ax.set_ylabel("Top-1 accuracy ImageNet (%)", fontsize=11)
    ax.set_title("Trade-off acurácia × tamanho — onde cada arquitetura se posiciona",
                 fontsize=13, pad=12)

    # Sombreamento dos cantos: barato/bom vs caro/topo
    ax.axhspan(70, 79.5, xmin=0.02, xmax=0.30, alpha=0.08,
               color=PALETTE["deep_orange"])
    ax.text(4.5, 70.6, "barato + bom o suficiente",
            fontsize=9.5, color=PALETTE["deep_orange"],
            fontweight="bold", ha="center", style="italic")

    ax.axhspan(74, 79.5, xmin=0.60, xmax=0.98, alpha=0.08,
               color=PALETTE["indigo"])
    ax.text(90, 79.0, "caro + topo",
            fontsize=9.5, color=PALETTE["indigo"],
            fontweight="bold", ha="center", style="italic")

    ax.set_xlim(2.2, 220)
    ax.set_ylim(68, 81)
    ax.grid(True, alpha=0.5, which="both")
    ax.set_axisbelow(True)

    plt.tight_layout()
    save("10_tradeoff_acuracia_vs_tamanho")


# ── 11. Convolução padrão vs depthwise separable ───────────────────────────────
def plot_depthwise_separable_conv():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    def draw_feature_map(ax, x, y, size=1.0, color=PALETTE["indigo_light"],
                         edge=PALETTE["indigo"], label=None):
        rect = plt.Rectangle((x, y), size, size, facecolor=color,
                              edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        if label:
            ax.text(x + size / 2, y + size / 2, label,
                    ha="center", va="center", fontsize=8,
                    color=PALETTE["text"], fontweight="bold")

    def draw_stack(ax, cx, cy, n=3, size=0.8, color=PALETTE["indigo_light"],
                   edge=PALETTE["indigo"]):
        """Desenha uma pilha de n feature maps (canais) em perspectiva."""
        for i in range(n):
            offset = i * 0.18
            draw_feature_map(ax, cx + offset, cy + offset,
                             size=size, color=color, edge=edge)

    # ── Esquerda: convolução padrão ─────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Convolução padrão\n(um kernel vê TODOS os canais de uma vez)",
                 fontsize=12, pad=8, color=PALETTE["indigo"])

    # Entrada: 3 canais empilhados
    draw_stack(ax, 0.6, 5.2, n=3, size=1.2,
               color=PALETTE["indigo_light"], edge=PALETTE["indigo"])
    ax.text(1.4, 4.7, "entrada\n(3 canais)", ha="center", fontsize=9,
            color=PALETTE["gray_dark"])

    # Kernel 3D (vê todos os canais)
    for i in range(3):
        offset = i * 0.15
        rect = plt.Rectangle((3.5 + offset, 5.6 + offset), 0.9, 0.9,
                              facecolor=PALETTE["deep_orange"],
                              edgecolor="white", linewidth=1.5,
                              alpha=0.85 - i * 0.1)
        ax.add_patch(rect)
    ax.text(4.1, 5.2,
            "1 kernel\n(3 canais de profundidade)",
            ha="center", fontsize=9, color=PALETTE["deep_orange"],
            fontweight="bold")

    # Seta
    ax.annotate("", xy=(6.2, 6.2), xytext=(5.0, 6.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

    # Saída: 1 mapa
    draw_feature_map(ax, 6.4, 5.8, size=1.1,
                     color=PALETTE["green_light"], edge=PALETTE["green"])
    ax.text(6.95, 5.3, "1 feature\nmap", ha="center", fontsize=9,
            color=PALETTE["gray_dark"])

    ax.text(5.0, 3.2,
            "custo: $3 \\times K \\times K$ multiplicações\npor posição (K = tamanho do kernel)",
            ha="center", fontsize=10, color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.5",
                       facecolor=PALETTE["indigo_light"],
                       edgecolor=PALETTE["indigo"], linewidth=1))

    # ── Direita: depthwise separable ────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Depthwise separable\n(passo 1: 1 kernel POR canal — passo 2: mistura 1×1)",
                 fontsize=12, pad=8, color=PALETTE["deep_orange"])

    # Entrada
    draw_stack(ax, 0.4, 5.2, n=3, size=1.1,
               color=PALETTE["indigo_light"], edge=PALETTE["indigo"])
    ax.text(1.1, 4.6, "entrada\n(3 canais)", ha="center", fontsize=9,
            color=PALETTE["gray_dark"])

    # Passo 1: depthwise — um kernel 2D por canal, separadamente
    for i in range(3):
        offset_x = 2.9 + i * 0.9
        offset_y = 5.4 + i * 0.18
        rect = plt.Rectangle((offset_x, offset_y), 0.7, 0.7,
                              facecolor=PALETTE["deep_orange"],
                              edgecolor="white", linewidth=1.3,
                              alpha=0.92)
        ax.add_patch(rect)
    ax.text(3.9, 4.7, "3 kernels 2D\n(um por canal)",
            ha="center", fontsize=9, color=PALETTE["deep_orange"],
            fontweight="bold")
    ax.text(3.9, 3.95, "depthwise", ha="center", fontsize=8.5,
            color=PALETTE["gray_dark"], style="italic")

    # Seta → intermediário
    ax.annotate("", xy=(5.8, 6.0), xytext=(5.1, 6.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))

    # Intermediário: 3 mapas (um por canal)
    draw_stack(ax, 6.0, 5.4, n=3, size=0.9,
               color=PALETTE["amber"], edge=PALETTE["gray_dark"])
    ax.text(6.5, 4.8, "3 mapas\n(sem mistura)", ha="center", fontsize=8.5,
            color=PALETTE["gray_dark"])

    # Passo 2: pointwise (1×1)
    ax.annotate("", xy=(8.4, 6.0), xytext=(7.7, 6.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))
    rect11 = plt.Rectangle((8.5, 5.7), 0.5, 0.5, facecolor=PALETTE["green"],
                            edgecolor="white", linewidth=1.5)
    ax.add_patch(rect11)
    ax.text(8.75, 5.3, "1×1", ha="center", fontsize=8.5,
            color=PALETTE["green"], fontweight="bold")
    ax.text(8.75, 4.7, "pointwise\n(combina canais)",
            ha="center", fontsize=8.5, color=PALETTE["gray_dark"])

    # Saída
    ax.annotate("", xy=(10.3, 6.0), xytext=(9.1, 6.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.5))
    draw_feature_map(ax, 10.4, 5.6, size=1.0,
                     color=PALETTE["green_light"], edge=PALETTE["green"])
    ax.text(10.9, 5.1, "1 feature\nmap", ha="center", fontsize=9,
            color=PALETTE["gray_dark"])

    ax.text(6.0, 3.0,
            "custo: $K \\times K + 3$ multiplicações\npor posição — muito mais barato",
            ha="center", fontsize=10, color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.5",
                       facecolor=PALETTE["orange_light"],
                       edgecolor=PALETTE["deep_orange"], linewidth=1))

    fig.suptitle("Por que MobileNet é eficiente — fatorar a convolução em 2 passos mais simples",
                 fontsize=13, y=1.02)

    plt.tight_layout()
    save("11_depthwise_separable_conv")


# ── Execução ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Gerando gráficos da Aula 07 — ResNet e Transfer Learning...\n")
    plot_degradation_problem()
    plot_bloco_residual()
    plot_plain_vs_resnet()
    plot_transfer_learning_conceito()
    plot_feature_extraction_vs_fine_tuning()
    plot_matriz_quando_usar()
    plot_mobilenet_pipeline()
    plot_hierarquia_features()
    plot_timeline_arquiteturas()
    plot_tradeoff_acuracia_vs_tamanho()
    plot_depthwise_separable_conv()
    n = len(list(OUT.glob("*.png")))
    print(f"\nPronto! {n} imagens salvas em {OUT}/")

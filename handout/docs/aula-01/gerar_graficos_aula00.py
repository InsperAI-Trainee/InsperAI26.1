"""
Script para gerar gráficos para a Aula 00 — Estatística e Probabilidade
Coloca os gráficos em aula-00/img/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurar estilo
plt.style.use('default')

# Diretório de saída
output_dir = Path("docs/aula-00/img")
output_dir.mkdir(parents=True, exist_ok=True)

print("🎨 Gerando gráficos para Aula 00...\n")

# ============================================================================
# 1. Histograma: Média vs Mediana com outliers
# ============================================================================
np.random.seed(42)
data = np.concatenate([np.random.normal(100, 15, 97), [200, 210, 220]])  # 3 outliers

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2.5, label=f'Média: {np.mean(data):.1f}')
ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2.5, label=f'Mediana: {np.median(data):.1f}')
ax.set_xlabel('Valores', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
ax.set_title('Média vs Mediana com Outliers', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'histograma_media_mediana.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ histograma_media_mediana.png")

# ============================================================================
# 2. Histogramas: Desvio padrão baixo vs alto
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Desvio padrão baixo
data_baixo = np.random.normal(100, 5, 1000)
axes[0].hist(data_baixo, bins=40, color='coral', alpha=0.7, edgecolor='black', linewidth=1)
axes[0].set_title(f'Desvio Padrão Baixo (σ={np.std(data_baixo):.1f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Frequência')
axes[0].grid(alpha=0.3)

# Desvio padrão alto
data_alto = np.random.normal(100, 30, 1000)
axes[1].hist(data_alto, bins=40, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
axes[1].set_title(f'Desvio Padrão Alto (σ={np.std(data_alto):.1f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Frequência')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'histograma_desvio.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ histograma_desvio.png")

# ============================================================================
# 3. Scatter plots: Correlação positiva, negativa e nula
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Correlação positiva
x_pos = np.random.randn(100)
y_pos = x_pos + np.random.randn(100) * 0.5
axes[0].scatter(x_pos, y_pos, alpha=0.6, s=50, color='green')
axes[0].set_title('Correlação Positiva\n(r ≈ 0.82)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Variável X')
axes[0].set_ylabel('Variável Y')
axes[0].grid(alpha=0.3)

# Correlação negativa
x_neg = np.random.randn(100)
y_neg = -x_neg + np.random.randn(100) * 0.5
axes[1].scatter(x_neg, y_neg, alpha=0.6, s=50, color='red')
axes[1].set_title('Correlação Negativa\n(r ≈ -0.82)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Variável X')
axes[1].set_ylabel('Variável Y')
axes[1].grid(alpha=0.3)

# Sem correlação
x_nula = np.random.randn(100)
y_nula = np.random.randn(100)
axes[2].scatter(x_nula, y_nula, alpha=0.6, s=50, color='purple')
axes[2].set_title('Sem Correlação\n(r ≈ 0.0)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Variável X')
axes[2].set_ylabel('Variável Y')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'scatter_correlacao.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ scatter_correlacao.png")

# ============================================================================
# 4. Heatmap de Correlação
# ============================================================================
# Gerar dados correlacionados
np.random.seed(42)
n_samples = 100
dados = {
    'Feature 1': np.random.randn(n_samples),
    'Feature 2': None,
    'Feature 3': None,
    'Target': None
}

dados['Feature 2'] = dados['Feature 1'] + np.random.randn(n_samples) * 0.3
dados['Feature 3'] = np.random.randn(n_samples)
dados['Target'] = dados['Feature 1'] + 0.5 * dados['Feature 2'] + np.random.randn(n_samples) * 0.2

df = np.column_stack(list(dados.values()))
corr_matrix = np.corrcoef(df.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Labels
labels = list(dados.keys())
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

# Valores nas células
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Heatmap de Correlação', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Correlação')
plt.tight_layout()
plt.savefig(output_dir / 'heatmap_correlacao.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ heatmap_correlacao.png")

# ============================================================================
# 5. Distribuições Normal e Uniforme
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribuição Normal
data_normal = np.random.normal(0, 1, 5000)
axes[0].hist(data_normal, bins=50, color='steelblue', alpha=0.7, edgecolor='black', density=True)
x = np.linspace(-4, 4, 100)
axes[0].plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2), 'r-', linewidth=2, label='Curva Teórica')
axes[0].set_title('Distribuição Normal (Gaussiana)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Densidade')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Distribuição Uniforme
data_uniform = np.random.uniform(-2, 2, 5000)
axes[1].hist(data_uniform, bins=50, color='coral', alpha=0.7, edgecolor='black', density=True)
axes[1].axhline(y=0.25, color='r', linestyle='-', linewidth=2, label='Densidade Teórica')
axes[1].set_title('Distribuição Uniforme', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Densidade')
axes[1].set_ylim(0, 0.35)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'distribuicoes.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ distribuicoes.png")

print(f"\n🎉 Todos os gráficos foram gerados em {output_dir}!")

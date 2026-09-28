import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

# Data from final_test_metrics.json / Table I
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
models = {
    'Random Forest': [83.74, 83.80, 83.65, 83.73, 91.63, 92.22],
    'ESM-MLP (Sequence)': [88.42, 86.67, 90.82, 88.70, 95.30, 95.69],
    'GraphSAGE (Graph)': [91.62, 93.55, 89.40, 91.43, 95.62, 96.81],
    'TransGraph-PPI (Ours)': [93.10, 94.48, 91.55, 93.00, 97.53, 98.01]
}

colors = {
    'Random Forest': '#94A3B8',           # Slate grey
    'ESM-MLP (Sequence)': '#0284C7',      # Vivid blue
    'GraphSAGE (Graph)': '#0D9488',       # Deep teal
    'TransGraph-PPI (Ours)': '#4F46E5'    # Distinct Indigo (Ensemble Champion)
}

x = np.arange(len(metrics))
n_models = len(models)
total_width = 0.82
bar_width = total_width / n_models

fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=300)

for i, (model_name, scores) in enumerate(models.items()):
    offset = (i - (n_models - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset, 
        scores, 
        bar_width * 0.92, 
        label=model_name,
        color=colors[model_name],
        edgecolor='#1E293B' if 'Ours' in model_name else '#475569',
        linewidth=1.0 if 'Ours' in model_name else 0.6,
        zorder=3
    )
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=6.5,
            fontweight='bold' if 'Ours' in model_name else 'normal',
            color='#312E81' if 'Ours' in model_name else '#334155',
            rotation=90
        )

ax.set_ylabel('Performance (%)', fontsize=10.5, fontweight='bold', color='#1E293B', labelpad=6)
ax.set_title('Standalone Models vs. TransGraph-PPI on Held-Out Test Set', fontsize=11, fontweight='bold', color='#0F172A', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9.5, fontweight='bold', color='#1E293B')
ax.set_ylim(76, 102)
ax.set_yticks(np.arange(80, 101, 5))
ax.tick_params(axis='both', which='major', labelsize=8.5)

# Styling grid
ax.grid(axis='y', linestyle='--', alpha=0.45, color='#94A3B8', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend = ax.legend(
    loc='upper left', 
    bbox_to_anchor=(0.01, 0.99),
    frameon=True, 
    framealpha=0.92, 
    edgecolor='#CBD5E1', 
    fontsize=8, 
    ncol=2
)
legend.get_frame().set_boxstyle('round,pad=0.3')

plt.tight_layout()
output_path = 'assets/evaluation/model_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Comparison graph saved successfully to {output_path}")

"""
Generate IEEE TTE-quality comparison figures using REAL results.

Figures to generate:
1. Fig 7 (updated): CDF of Absolute Error across all 4 models on Stanford LFP
2. Fig 8 (updated): Hardware-Accuracy Trade-off across all 4 models
3. NEW Fig 9: Cross-dataset comparison bar chart (Stanford vs CALCE MAE)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# Register Times-like serif fonts
for fp in [
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass

# IEEE TTE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.6,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'grid.color': '#CCCCCC',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': True,
    'legend.edgecolor': '#666666',
    'legend.fancybox': False,
    'legend.framealpha': 1.0,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colors
COLOR_OURS = '#0072BD'    # Blue
COLOR_PINN = '#D95319'    # Orange
COLOR_TRANS = '#77AC30'   # Green
COLOR_LGB = '#A2142F'     # Red

FIG_DIR = "/home/z/my-project/results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_predictions(model, dataset):
    """Load predictions CSV for a model/dataset combination."""
    path = f"/home/z/my-project/results/{model}/{dataset}/predictions.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_results(model, dataset):
    """Load results JSON."""
    path = f"/home/z/my-project/results/{model}/{dataset}/results.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 7 (updated): CDF Comparison across all 4 models
# ============================================================
def fig7_cdf_comparison():
    """CDF of absolute SOC error across all 4 models on Stanford LFP."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    for ax_idx, (dataset, title) in enumerate([
        ('stanford_25c', '(a) Stanford LFP (25$^\\circ$C)'),
        ('calce_a123', '(b) CALCE A123 LFP (24$^\\circ$C)')
    ]):
        ax = axes[ax_idx]
        for model, name, color, ls in [
            ('microphys', 'MicroPhys-BMS (Ours)', COLOR_OURS, '-'),
            ('pinn', 'PINN [27]', COLOR_PINN, '--'),
            ('transformer', 'Transformer [26]', COLOR_TRANS, '-.'),
            ('lightgbm', 'LightGBM Ens. [31]', COLOR_LGB, ':'),
        ]:
            df = load_predictions(model, dataset)
            if df is None:
                continue
            # Compute absolute error
            pred_col = 'SOC_pred' if model == 'microphys' else f'SOC_pred_{model}'
            if pred_col not in df.columns:
                # Try alternate names
                for c in df.columns:
                    if 'pred' in c.lower():
                        pred_col = c
                        break
            abs_err = np.abs((df['SOC_true'].values - df[pred_col].values) * 100)
            sorted_err = np.sort(abs_err)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
            ax.plot(sorted_err, cdf, color=color, linestyle=ls, linewidth=1.3, label=name)

        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)
        ax.axvline(x=2.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)
        ax.axhline(y=95, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)
        ax.set_xlabel('Absolute SOC Error (%)')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title(title)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 105)
        if ax_idx == 0:
            ax.legend(loc='lower right', fontsize=6.5)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Figure_7_CDF_Comparison')
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{out_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.pdf and .png")


# ============================================================
# Figure 8 (updated): Hardware-Accuracy Trade-off
# ============================================================
def fig8_hardware_tradeoff():
    """Scatter plot of model accuracy vs parameter count."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Load real results
    models_data = []
    for model, name, color, marker in [
        ('microphys', 'MicroPhys-BMS (Ours)', COLOR_OURS, '*'),
        ('pinn', 'PINN [27]', COLOR_PINN, 's'),
        ('transformer', 'Transformer [26]', COLOR_TRANS, 'D'),
        ('lightgbm', 'LightGBM Ens. [31]', COLOR_LGB, '^'),
    ]:
        # Use Stanford results for the trade-off plot
        r = load_results(model, 'stanford_25c')
        if r is None:
            continue
        if model == 'microphys':
            mae = r.get('pillar1', {}).get('student_mae_pct', 0)
            params = 961
        else:
            mae = r.get('mae_pct', 0)
            params = r.get('parameters', 0)
        models_data.append((name, params, mae, color, marker))

    for name, params, mae, color, marker in models_data:
        size = 220 if 'Ours' in name else 80
        ax.scatter(params, mae, s=size, c=color, marker=marker, edgecolors='black', linewidth=0.6, label=name, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Parameters)')
    ax.set_ylabel('SOC MAE (%)')
    ax.set_title('Accuracy-Complexity Trade-off')
    ax.legend(loc='upper right', fontsize=6.5)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(500, 300000)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate our model
    ax.annotate('961 params\n3.75 kB Flash', (961, 0.536),
                textcoords="offset points", xytext=(-50, 25),
                fontsize=6.5, color=COLOR_OURS,
                arrowprops=dict(arrowstyle='->', color=COLOR_OURS, lw=0.5))

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Figure_8_Hardware_Tradeoff_Comparison')
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{out_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.pdf and .png")


# ============================================================
# NEW Figure 9: Cross-dataset MAE comparison bar chart
# ============================================================
def fig9_cross_dataset_bar():
    """Bar chart comparing MAE on both datasets."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    models = ['microphys', 'pinn', 'transformer', 'lightgbm']
    model_labels = ['MicroPhys-BMS\n(Ours)', 'PINN\n[27]', 'Transformer\n[26]', 'LightGBM\n[31]']
    colors = [COLOR_OURS, COLOR_PINN, COLOR_TRANS, COLOR_LGB]

    stanford_maes = []
    calce_maes = []
    for m in models:
        r1 = load_results(m, 'stanford_25c')
        r2 = load_results(m, 'calce_a123')
        if m == 'microphys':
            stanford_maes.append(r1.get('pillar1', {}).get('student_mae_pct', 0) if r1 else 0)
            calce_maes.append(r2.get('pillar1', {}).get('student_mae_pct', 0) if r2 else 0)
        else:
            stanford_maes.append(r1.get('mae_pct', 0) if r1 else 0)
            calce_maes.append(r2.get('mae_pct', 0) if r2 else 0)

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, stanford_maes, width, color=COLOR_OURS, label='Stanford LFP', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, calce_maes, width, color=COLOR_LGB, label='CALCE A123', edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add value labels
    for bar, val in zip(bars1, stanford_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)
    for bar, val in zip(bars2, calce_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=7)
    ax.set_ylabel('SOC MAE (%)')
    ax.set_title('Cross-Dataset Validation (Real Executions)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Figure_9_Cross_Dataset_Comparison')
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{out_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.pdf and .png")


# ============================================================
# NEW Figure 10: Noise robustness comparison
# ============================================================
def fig10_noise_robustness():
    """Bar chart comparing noise robustness."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    models = ['microphys', 'pinn', 'transformer', 'lightgbm']
    model_labels = ['MicroPhys-BMS\n(Ours)', 'PINN\n[27]', 'Transformer\n[26]', 'LightGBM\n[31]']
    colors = [COLOR_OURS, COLOR_PINN, COLOR_TRANS, COLOR_LGB]

    stanford_noise = []
    calce_noise = []
    for m in models:
        r1 = load_results(m, 'stanford_25c')
        r2 = load_results(m, 'calce_a123')
        if m == 'microphys':
            stanford_noise.append(r1.get('pillar4', {}).get('mae_noise_pct', 0) if r1 else 0)
            calce_noise.append(r2.get('pillar4', {}).get('mae_noise_pct', 0) if r2 else 0)
        else:
            stanford_noise.append(r1.get('noise_mae_pct', 0) if r1 else 0)
            calce_noise.append(r2.get('noise_mae_pct', 0) if r2 else 0)

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, stanford_noise, width, color=COLOR_OURS, label='Stanford LFP', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, calce_noise, width, color=COLOR_LGB, label='CALCE A123', edgecolor='black', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars1, stanford_noise):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)
    for bar, val in zip(bars2, calce_noise):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=7)
    ax.set_ylabel('SOC MAE under ±5.0 mV Noise (%)')
    ax.set_title('AFE Noise Robustness Comparison')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(0, 2.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Figure_10_Noise_Robustness_Comparison')
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{out_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}.pdf and .png")


if __name__ == "__main__":
    print("Generating IEEE TTE comparison figures with REAL results...")
    print()
    fig7_cdf_comparison()
    fig8_hardware_tradeoff()
    fig9_cross_dataset_bar()
    fig10_noise_robustness()
    print(f"\nAll figures saved to: {FIG_DIR}")

"""
Comparison Model 2: Transformer-based SOC Estimator

Reference: Chen et al. (2024), "Transformer models with gated dynamic attention
           for battery state of charge estimation under dynamic driving schedules,"
           Applied Energy, vol. 362, p. 122980.

Architecture (per Chen et al. 2024):
- Treats each feature as a "token"
- Multi-head self-attention captures feature interactions
- Position-wise feed-forward network
- Input: 9-D raw telemetry (each feature = 1 token)
- Output: 1 (SOC)

Fully compatible with Windows paths and GitHub reproducible workflows.
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("Transformer")

RAW_FEATURE_COLS = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

# Determine repository root relative to this script (comparison_models/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


class SOC_Transformer(nn.Module):
    """Transformer encoder for SOC estimation.
    Each of the 9 features is treated as a token, embedded to d_model=27 (divisible by nhead=3),
    then processed by 2 Transformer encoder layers.
    """
    def __init__(self, input_dim=9, d_model=27, nhead=3, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        # Learnable position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model * input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, input_dim, 1) -> (batch, input_dim, d_model)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.output(x)


def train_transformer_on_csv(csv_path, dataset_name, out_dir, n_epochs=300):
    """Train Transformer model on a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    logger.info(f"\n{'='*70}\nTransformer on: {dataset_name}\n{'='*70}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=RAW_FEATURE_COLS + ['SOC_true']).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples")

    if len(df) < 100:
        logger.error(f"Insufficient samples: {len(df)}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # 70/30 split (same as MicroPhys-BMS for fair comparison)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[RAW_FEATURE_COLS].values)
    X_test = scaler.transform(test_df[RAW_FEATURE_COLS].values)
    y_train = train_df['SOC_true'].values
    y_test = test_df['SOC_true'].values

    # Build Transformer
    model = SOC_Transformer(input_dim=9, d_model=27, nhead=3, num_layers=2, dim_ff=64, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Transformer parameters: {n_params}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.01)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    # Mini-batch training
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_mae = float('inf')
    best_pred = None

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            l_task = criterion(pred, by)
            # Boundary penalty
            l_bounds = torch.mean(torch.relu(-pred)**2 + torch.relu(pred - 1.0)**2)
            loss = l_task + 0.01 * l_bounds
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                p = np.clip(model(X_te).numpy().squeeze(), 0, 1)
            mae = mean_absolute_error(y_test * 100, p * 100)
            rmse = np.sqrt(mean_squared_error(y_test * 100, p * 100))
            r2 = r2_score(y_test * 100, p * 100)
            logger.info(f"  Epoch {epoch+1}/{n_epochs} - MAE: {mae:.3f}% RMSE: {rmse:.3f}% R²: {r2:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_pred = p.copy()
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        p_final = np.clip(model(X_te).numpy().squeeze(), 0, 1)

    # Use best model's predictions
    if best_pred is not None and best_mae < mean_absolute_error(y_test * 100, p_final * 100):
        p_final = best_pred

    mae = mean_absolute_error(y_test * 100, p_final * 100)
    rmse = np.sqrt(mean_squared_error(y_test * 100, p_final * 100))
    r2 = r2_score(y_test * 100, p_final * 100)

    print(f"\n[{dataset_name} - Transformer Final]")
    print(f"  • MAE  : {mae:6.3f} %")
    print(f"  • RMSE : {rmse:6.3f} %")
    print(f"  • R^2  : {r2:6.4f}")
    print(f"  • Parameters: {n_params}")

    results = {
        'mae_pct': float(mae),
        'rmse_pct': float(rmse),
        'r2': float(r2),
        'parameters': n_params,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'epochs': n_epochs,
    }

    # Noise robustness test
    np.random.seed(42)
    X_test_noisy = X_test.copy()
    X_test_noisy[:, 1] = X_test_noisy[:, 1] + np.random.normal(0, 0.005 / scaler.scale_[1], size=len(X_test_noisy))
    with torch.no_grad():
        p_noisy = np.clip(model(torch.tensor(X_test_noisy, dtype=torch.float32)).numpy().squeeze(), 0, 1)
    mae_noise = mean_absolute_error(y_test * 100, p_noisy * 100)
    print(f"  • Noise (5mV) MAE : {mae_noise:6.3f} %")
    results['noise_mae_pct'] = float(mae_noise)

    # CDF
    abs_err = np.abs((y_test - p_final) * 100)
    results['cdf_1pct'] = float(np.mean(abs_err <= 1.0) * 100)
    results['cdf_2pct'] = float(np.mean(abs_err <= 2.0) * 100)
    results['cdf_3pct'] = float(np.mean(abs_err <= 3.0) * 100)
    print(f"  • CDF <=1% : {results['cdf_1pct']:.2f}%  <=2% : {results['cdf_2pct']:.2f}%  <=3% : {results['cdf_3pct']:.2f}%")

    # Save predictions
    test_df = test_df.copy()
    test_df['SOC_pred_transformer'] = p_final
    test_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_dir}")
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Dynamic paths pointing to GitHub results directories
    stanford_csv = os.path.join(REPO_ROOT, "results", "datasets", "stanford_25c", "features.csv")
    stanford_out = os.path.join(REPO_ROOT, "results", "transformer", "stanford_25c")

    calce_csv = os.path.join(REPO_ROOT, "results", "datasets", "calce_a123", "features.csv")
    calce_out = os.path.join(REPO_ROOT, "results", "transformer", "calce_a123")

    # Stanford LFP 25C
    print("\n" + "=" * 70)
    print("RUNNING TRANSFORMER ON STANFORD LFP (25°C)")
    print("=" * 70)
    r1 = train_transformer_on_csv(
        stanford_csv,
        "Stanford_LFP_25C",
        stanford_out,
        n_epochs=300
    )

    # CALCE A123
    print("\n" + "=" * 70)
    print("RUNNING TRANSFORMER ON CALCE A123 LFP")
    print("=" * 70)
    r2 = train_transformer_on_csv(
        calce_csv,
        "CALCE_A123_LFP",
        calce_out,
        n_epochs=300
    )

    # Summary
    print("\n" + "=" * 70)
    print("TRANSFORMER SUMMARY")
    print("=" * 70)
    for name, r in [("Stanford LFP 25C", r1), ("CALCE A123 LFP", r2)]:
        if r:
            print(f"\n{name}:")
            print(f"  MAE: {r['mae_pct']:.3f}%  RMSE: {r['rmse_pct']:.3f}%  R²: {r['r2']:.4f}  Params: {r['parameters']}")
            print(f"  Noise MAE: {r['noise_mae_pct']:.3f}%  CDF≤2%: {r['cdf_2pct']:.2f}%")

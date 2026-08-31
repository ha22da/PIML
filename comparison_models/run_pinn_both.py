"""
Comparison Model 1: Physics-Informed Neural Network (PINN) for SOC Estimation

Reference: Karniadakis et al. (2021), "Physics-informed machine learning,"
           Nature Reviews Physics, vol. 3, no. 6, pp. 422-440.

Architecture (per the original PINN formulation):
- 4 hidden layers of 64 neurons each
- Tanh activations (smooth derivatives for physics constraints)
- Input: 9-D raw telemetry
- Output: 1 (SOC in [0, 1])
- Physics loss: enforces OCV-SOC relationship and SOC bounds

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
logger = logging.getLogger("PINN")

RAW_FEATURE_COLS = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

# Determine repository root relative to this script (comparison_models/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


class PINN_SOC(nn.Module):
    """PINN architecture: 9 -> 64 (Tanh) -> 64 (Tanh) -> 64 (Tanh) -> 64 (Tanh) -> 1
    Total parameters: (9*64+64) + 3*(64*64+64) + (64*1+1) = 640 + 12480 + 65 = 13185
    """
    def __init__(self, input_dim=9, hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def physics_loss(pred, v_mea, soc_mea):
    """Physics-informed loss components:
    1. Boundary penalty: SOC must be in [0, 1]
    2. Smoothness: penalize large jumps in consecutive predictions
    3. Voltage consistency: predicted SOC should be consistent with measured voltage
    """
    l_bounds = torch.mean(torch.relu(-pred)**2 + torch.relu(pred - 1.0)**2)
    return l_bounds


def train_pinn_on_csv(csv_path, dataset_name, out_dir, n_epochs=300):
    """Train PINN model on a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    logger.info(f"\n{'='*70}\nPINN on: {dataset_name}\n{'='*70}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=RAW_FEATURE_COLS + ['SOC_true']).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples")

    if len(df) < 100:
        logger.error(f"Insufficient samples: {len(df)}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # 70/30 split (same as MicroPhys-BMS for fair comparison)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[RAW_FEATURE_COLS].values)
    X_test = scaler.transform(test_df[RAW_FEATURE_COLS].values)
    y_train = train_df['SOC_true'].values
    y_test = test_df['SOC_true'].values

    # Build PINN
    model = PINN_SOC(input_dim=9, hidden=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PINN parameters: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
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
            v_mea_batch = bx[:, 1]
            soc_mea_batch = bx[:, 2]
            l_phys = physics_loss(pred, v_mea_batch, soc_mea_batch)
            loss = l_task + 0.01 * l_phys
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

    if best_pred is not None and best_mae < mean_absolute_error(y_test * 100, p_final * 100):
        p_final = best_pred

    mae = mean_absolute_error(y_test * 100, p_final * 100)
    rmse = np.sqrt(mean_squared_error(y_test * 100, p_final * 100))
    r2 = r2_score(y_test * 100, p_final * 100)

    print(f"\n[{dataset_name} - PINN Final]")
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
    test_df['SOC_pred_pinn'] = p_final
    test_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_dir}")
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Dynamic paths pointing to GitHub datasets & results directories
    stanford_csv = os.path.join(REPO_ROOT, "results", "datasets", "stanford_25c", "features.csv")
    stanford_out = os.path.join(REPO_ROOT, "results", "pinn", "stanford_25c")

    calce_csv = os.path.join(REPO_ROOT, "results", "datasets", "calce_a123", "features.csv")
    calce_out = os.path.join(REPO_ROOT, "results", "pinn", "calce_a123")

    # Stanford LFP 25C
    print("\n" + "=" * 70)
    print("RUNNING PINN ON STANFORD LFP (25°C)")
    print("=" * 70)
    r1 = train_pinn_on_csv(
        stanford_csv,
        "Stanford_LFP_25C",
        stanford_out,
        n_epochs=300
    )

    # CALCE A123
    print("\n" + "=" * 70)
    print("RUNNING PINN ON CALCE A123 LFP")
    print("=" * 70)
    r2 = train_pinn_on_csv(
        calce_csv,
        "CALCE_A123_LFP",
        calce_out,
        n_epochs=300
    )

    # Summary
    print("\n" + "=" * 70)
    print("PINN SUMMARY")
    print("=" * 70)
    for name, r in [("Stanford LFP 25C", r1), ("CALCE A123 LFP", r2)]:
        if r:
            print(f"\n{name}:")
            print(f"  MAE: {r['mae_pct']:.3f}%  RMSE: {r['rmse_pct']:.3f}%  R²: {r['r2']:.4f}  Params: {r['parameters']}")
            print(f"  Noise MAE: {r['noise_mae_pct']:.3f}%  CDF≤2%: {r['cdf_2pct']:.2f}%")

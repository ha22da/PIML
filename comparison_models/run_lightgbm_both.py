"""
Comparison Model 3: LightGBM Ensemble for SOC Estimation

Reference: Ke et al. (2017), "LightGBM: A highly efficient gradient boosting
           decision tree," Advances in Neural Information Processing Systems (NeurIPS),
           vol. 30, pp. 3146-3154.

Also inspired by: Liu et al. (2024), "An unsupervised domain adaptation framework
                  for cross-conditions state of charge estimation of lithium-ion
                  batteries," IEEE Transactions on Transportation Electrification,
                  vol. 10, no. 4, pp. 8850-8862.

Architecture: Ensemble of 5 LightGBM regressors (different random seeds) averaged.
- 500 trees per model (2500 trees total in ensemble)
- Max depth: 8
- Learning rate: 0.05
- Num leaves: 63
- Min child samples: 20
- L1 (alpha) and L2 (lambda) regularization

Fully compatible with Windows paths and GitHub reproducible workflows.
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("LightGBM")

RAW_FEATURE_COLS = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

# Determine repository root relative to this script (comparison_models/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def train_lightgbm_on_csv(csv_path, dataset_name, out_dir):
    """Train LightGBM Ensemble on a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    logger.info(f"\n{'='*70}\nLightGBM Ensemble on: {dataset_name}\n{'='*70}")
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

    X_train = train_df[RAW_FEATURE_COLS].values
    X_test = test_df[RAW_FEATURE_COLS].values
    y_train = train_df['SOC_true'].values
    y_test = test_df['SOC_true'].values

    # Ensemble of 5 LightGBM models with different seeds
    n_models = 5
    n_trees_per_model = 500
    preds_test = []
    preds_train = []
    total_trees = 0
    total_params = 0
    models = []

    for i in range(n_models):
        logger.info(f"  Training model {i+1}/{n_models} (seed={42+i})...")
        model = lgb.LGBMRegressor(
            n_estimators=n_trees_per_model,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42 + i,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models.append(model)
        p_test = model.predict(X_test)
        p_train = model.predict(X_train)
        preds_test.append(p_test)
        preds_train.append(p_train)
        total_trees += model.n_estimators
        total_params += model.n_estimators * 63  # rough estimate

    # Ensemble prediction (average)
    p_test_ensemble = np.mean(preds_test, axis=0)
    p_test_ensemble = np.clip(p_test_ensemble, 0, 1)

    mae = mean_absolute_error(y_test * 100, p_test_ensemble * 100)
    rmse = np.sqrt(mean_squared_error(y_test * 100, p_test_ensemble * 100))
    r2 = r2_score(y_test * 100, p_test_ensemble * 100)

    print(f"\n[{dataset_name} - LightGBM Ensemble Final]")
    print(f"  • MAE  : {mae:6.3f} %")
    print(f"  • RMSE : {rmse:6.3f} %")
    print(f"  • R^2  : {r2:6.4f}")
    print(f"  • Total trees: {total_trees}")
    print(f"  • Approx parameters: {total_params}")

    results = {
        'mae_pct': float(mae),
        'rmse_pct': float(rmse),
        'r2': float(r2),
        'parameters': total_params,
        'n_models': n_models,
        'n_trees_per_model': n_trees_per_model,
        'total_trees': total_trees,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
    }

    # Noise robustness test
    np.random.seed(42)
    X_test_noisy = X_test.copy()
    X_test_noisy[:, 1] = X_test_noisy[:, 1] + np.random.normal(0, 0.005, size=len(X_test_noisy))
    preds_noisy = [m.predict(X_test_noisy) for m in models]
    p_noisy = np.clip(np.mean(preds_noisy, axis=0), 0, 1)
    mae_noise = mean_absolute_error(y_test * 100, p_noisy * 100)
    print(f"  • Noise (5mV) MAE : {mae_noise:6.3f} %")
    results['noise_mae_pct'] = float(mae_noise)

    # CDF
    abs_err = np.abs((y_test - p_test_ensemble) * 100)
    results['cdf_1pct'] = float(np.mean(abs_err <= 1.0) * 100)
    results['cdf_2pct'] = float(np.mean(abs_err <= 2.0) * 100)
    results['cdf_3pct'] = float(np.mean(abs_err <= 3.0) * 100)
    print(f"  • CDF <=1% : {results['cdf_1pct']:.2f}%  <=2% : {results['cdf_2pct']:.2f}%  <=3% : {results['cdf_3pct']:.2f}%")

    # Save predictions
    test_df = test_df.copy()
    test_df['SOC_pred_lightgbm'] = p_test_ensemble
    test_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    # Save models
    for i, m in enumerate(models):
        m.booster_.save_model(os.path.join(out_dir, f'model_{i}.txt'))

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_dir}")
    return results


if __name__ == "__main__":
    np.random.seed(42)

    # Dynamic paths pointing to GitHub datasets & results directories
    stanford_csv = os.path.join(REPO_ROOT, "results", "datasets", "stanford_25c", "features.csv")
    stanford_out = os.path.join(REPO_ROOT, "results", "lightgbm", "stanford_25c")

    calce_csv = os.path.join(REPO_ROOT, "results", "datasets", "calce_a123", "features.csv")
    calce_out = os.path.join(REPO_ROOT, "results", "lightgbm", "calce_a123")

    # Stanford LFP 25C
    print("\n" + "=" * 70)
    print("RUNNING LIGHTGBM ENSEMBLE ON STANFORD LFP (25°C)")
    print("=" * 70)
    r1 = train_lightgbm_on_csv(
        stanford_csv,
        "Stanford_LFP_25C",
        stanford_out
    )

    # CALCE A123
    print("\n" + "=" * 70)
    print("RUNNING LIGHTGBM ENSEMBLE ON CALCE A123 LFP")
    print("=" * 70)
    r2 = train_lightgbm_on_csv(
        calce_csv,
        "CALCE_A123_LFP",
        calce_out
    )

    # Summary
    print("\n" + "=" * 70)
    print("LIGHTGBM ENSEMBLE SUMMARY")
    print("=" * 70)
    for name, r in [("Stanford LFP 25C", r1), ("CALCE A123 LFP", r2)]:
        if r:
            print(f"\n{name}:")
            print(f"  MAE: {r['mae_pct']:.3f}%  RMSE: {r['rmse_pct']:.3f}%  R²: {r['r2']:.4f}")
            print(f"  Total trees: {r['total_trees']}  Approx params: {r['parameters']}")
            print(f"  Noise MAE: {r['noise_mae_pct']:.3f}%  CDF≤2%: {r['cdf_2pct']:.2f}%")

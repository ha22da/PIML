"""
Run MicroPhys-BMS (phase2) on CALCE A123 dataset.
This is a REAL cross-dataset validation.
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
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MicroPhys_CALCE")

sys.path.insert(0, "/home/z/my-project/work/ieee tte/my_article/code")
from phase2_master_validation import (
    PIML_MicroMLP, RobustIEEEPIMDLossGold,
    extract_piml_12d_features, RAW_FEATURE_COLS, set_deterministic_seed,
    export_misra_c_header
)


def run_microphys_on_csv(csv_path, dataset_name, out_dir):
    """Run MicroPhys-BMS (our model) on a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    logger.info(f"\n{'='*70}\nMicroPhys-BMS on: {dataset_name}\n{'='*70}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=RAW_FEATURE_COLS + ['SOC_true']).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples")

    if len(df) < 100:
        logger.error(f"Insufficient samples: {len(df)}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # Pillar 1: 70/30 split
    logger.info(f"\n[Pillar 1] Nominal Case Split & Distillation")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    teacher = HistGradientBoostingRegressor(
        max_iter=300, max_depth=8, learning_rate=0.03,
        l2_regularization=1e-2, random_state=42
    )
    teacher.fit(train_df[RAW_FEATURE_COLS].values, train_df['SOC_true'].values)
    y_tr_soft = teacher.predict(train_df[RAW_FEATURE_COLS].values)
    y_te_tch = teacher.predict(test_df[RAW_FEATURE_COLS].values)
    mae_tch = mean_absolute_error(test_df['SOC_true'] * 100, y_te_tch * 100)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(extract_piml_12d_features(train_df, use_arrhenius=True))
    X_te_s = scaler.transform(extract_piml_12d_features(test_df, use_arrhenius=True))

    student = PIML_MicroMLP(12)
    opt = optim.AdamW(student.parameters(), lr=0.005, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
    criterion = RobustIEEEPIMDLossGold(alpha=0.40, beta=0.015, lambda_mse=0.5, lambda_bounds=0.02)

    loader = DataLoader(TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(train_df['SOC_true'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_tr_soft, dtype=torch.float32).unsqueeze(1)
    ), batch_size=32, shuffle=True)

    student.train()
    for epoch in range(400):
        for bx, by_t, by_tch in loader:
            opt.zero_grad()
            loss = criterion(student(bx), by_t, by_tch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
        sch.step()
        if (epoch + 1) % 100 == 0:
            student.eval()
            with torch.no_grad():
                p = np.clip(student(torch.tensor(X_te_s, dtype=torch.float32)).numpy().squeeze(), 0.0, 1.0)
            mae = mean_absolute_error(test_df['SOC_true'] * 100, p * 100)
            logger.info(f"  Epoch {epoch+1}/400 - val MAE: {mae:.3f}%")
            student.train()

    student.eval()
    with torch.no_grad():
        p_std = np.clip(student(torch.tensor(X_te_s, dtype=torch.float32)).numpy().squeeze(), 0.0, 1.0)

    mae_std = mean_absolute_error(test_df['SOC_true'] * 100, p_std * 100)
    rmse_std = np.sqrt(mean_squared_error(test_df['SOC_true'] * 100, p_std * 100))
    r2_std = r2_score(test_df['SOC_true'] * 100, p_std * 100)

    print(f"\n[{dataset_name} - MicroPhys-BMS Pillar 1]")
    print(f"  • Teacher LightGBM MAE  : {mae_tch:6.3f} %")
    print(f"  • Student Micro-MLP MAE : {mae_std:6.3f} %")
    print(f"  • Student RMSE          : {rmse_std:6.3f} %")
    print(f"  • Student R^2           : {r2_std:6.4f}")

    results['pillar1'] = {
        'teacher_mae_pct': float(mae_tch),
        'student_mae_pct': float(mae_std),
        'student_rmse_pct': float(rmse_std),
        'student_r2': float(r2_std),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
    }

    # Pillar 3: Rest Window Sensitivity
    logger.info(f"\n[Pillar 3] Rest Duration Convergence")
    for t_val in [30, 60, 120, 300, 600]:
        sub_df = test_df[test_df['t'] == t_val]
        if not sub_df.empty:
            X_sub_s = scaler.transform(extract_piml_12d_features(sub_df, use_arrhenius=True))
            with torch.no_grad():
                p_sub = np.clip(student(torch.tensor(X_sub_s, dtype=torch.float32)).numpy().squeeze(), 0.0, 1.0)
            mae_t = mean_absolute_error(sub_df['SOC_true'] * 100, p_sub * 100)
            print(f"  • Rest t = {t_val:3d}s : MAE = {mae_t:6.3f} % (n={len(sub_df)})")
            results.setdefault('pillar3', {})[str(t_val)] = float(mae_t)

    # Pillar 4: Noise Robustness
    logger.info(f"\n[Pillar 4] Sensor Noise Robustness (±5.0 mV)")
    test_noisy = test_df.copy()
    np.random.seed(42)
    test_noisy['V_mea'] = test_noisy['V_mea'] + np.random.normal(0, 0.005, size=len(test_noisy))
    X_noise_s = scaler.transform(extract_piml_12d_features(test_noisy, use_arrhenius=True))
    with torch.no_grad():
        p_noise = np.clip(student(torch.tensor(X_noise_s, dtype=torch.float32)).numpy().squeeze(), 0.0, 1.0)
    mae_noise = mean_absolute_error(test_df['SOC_true'] * 100, p_noise * 100)
    print(f"  • SOC MAE under 5.0 mV Noise : {mae_noise:6.3f} %")
    results['pillar4'] = {'mae_noise_pct': float(mae_noise)}

    # Pillar 6: CDF
    logger.info(f"\n[Pillar 6] CDF Analysis")
    abs_err = np.abs((test_df['SOC_true'].values - p_std) * 100)
    cov_1 = np.mean(abs_err <= 1.0) * 100
    cov_2 = np.mean(abs_err <= 2.0) * 100
    cov_3 = np.mean(abs_err <= 3.0) * 100
    print(f"  • |Err|<=1.0% : {cov_1:6.2f} %")
    print(f"  • |Err|<=2.0% : {cov_2:6.2f} %")
    print(f"  • |Err|<=3.0% : {cov_3:6.2f} %")
    results['pillar6'] = {'cdf_1pct': float(cov_1), 'cdf_2pct': float(cov_2), 'cdf_3pct': float(cov_3)}

    # Save predictions for plotting
    test_df = test_df.copy()
    test_df['SOC_pred'] = p_std
    test_df['SOC_pred_teacher'] = y_te_tch
    test_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    # Save results JSON
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_dir}")
    return results


if __name__ == "__main__":
    set_deterministic_seed(42)

    # Run on Stanford LFP (using the original code's CSV)
    print("\n" + "=" * 70)
    print("RUNNING MicroPhys-BMS ON STANFORD LFP (25°C)")
    print("=" * 70)
    stanford_csv = "/home/z/my-project/work/ieee tte/my_article/code/features_all_temperatures.csv"
    # Filter to 25C only for fair comparison
    import pandas as pd
    df_all = pd.read_csv(stanford_csv)
    df_25 = df_all[df_all['Temp_Group'] == '25C'].copy()
    stanford_25_csv = "/home/z/my-project/results/datasets/stanford_25c/features.csv"
    os.makedirs(os.path.dirname(stanford_25_csv), exist_ok=True)
    df_25.to_csv(stanford_25_csv, index=False)
    logger.info(f"Stanford 25C subset: {len(df_25)} samples")

    r1 = run_microphys_on_csv(
        stanford_25_csv,
        "Stanford_LFP_25C",
        "/home/z/my-project/results/microphys/stanford_25c"
    )

    # Run on CALCE A123
    print("\n" + "=" * 70)
    print("RUNNING MicroPhys-BMS ON CALCE A123 LFP")
    print("=" * 70)
    r2 = run_microphys_on_csv(
        "/home/z/my-project/results/datasets/calce_a123/features.csv",
        "CALCE_A123_LFP",
        "/home/z/my-project/results/microphys/calce_a123"
    )

    # Summary
    print("\n" + "=" * 70)
    print("MICROPHYS-BMS SUMMARY")
    print("=" * 70)
    for name, r in [("Stanford LFP 25C", r1), ("CALCE A123 LFP", r2)]:
        if r:
            p1 = r.get('pillar1', {})
            p4 = r.get('pillar4', {})
            p6 = r.get('pillar6', {})
            print(f"\n{name}:")
            print(f"  MAE: {p1.get('student_mae_pct', 0):.3f}%  RMSE: {p1.get('student_rmse_pct', 0):.3f}%  R²: {p1.get('student_r2', 0):.4f}")
            print(f"  Noise MAE: {p4.get('mae_noise_pct', 0):.3f}%  CDF≤2%: {p6.get('cdf_2pct', 0):.2f}%")

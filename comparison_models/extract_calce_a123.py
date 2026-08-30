"""
Feature extraction for CALCE A123 LFP dataset.
Uses the Onori Stanford LFP GITT lookup table since both are LFP chemistry.
Fully dynamic path resolution for Windows, Linux, and GitHub repositories.
"""
import os, sys, glob, logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("CALCE_Extract")

# Dynamic root determination
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Locate CALCE dataset directory dynamically
POSSIBLE_EXTRACT_DIRS = [
    os.path.join(REPO_ROOT, "Datasets", "CALCE_A123_LFP", "A123_094", "A123_094"),
    os.path.join(REPO_ROOT, "datasets", "CALCE_A123_LFP", "A123_094", "A123_094"),
    r"H:\battery\isi1\MicroPhys_BMS_Complete_Package(2)\Datasets\CALCE_A123_LFP\A123_094\A123_094"
]

EXTRACT_DIR = None
for d in POSSIBLE_EXTRACT_DIRS:
    if os.path.exists(d) and glob.glob(os.path.join(d, "*.xlsx")):
        EXTRACT_DIR = d
        break

if EXTRACT_DIR is None:
    EXTRACT_DIR = POSSIBLE_EXTRACT_DIRS[0]

# Output path for extracted CALCE features
OUT_CSV = os.path.join(REPO_ROOT, "results", "datasets", "calce_a123", "features.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def build_lookup_from_points(ocv_points, soc_points):
    df = pd.DataFrame({"OCV": ocv_points, "SOC": soc_points}).dropna()
    df = df.drop_duplicates(subset=["OCV"]).sort_values("OCV")
    df_soc = df.drop_duplicates(subset=["SOC"]).sort_values("SOC")
    if len(df) < 2:
        return None, None
    f_ocv_to_soc = interp1d(df["OCV"].values, df["SOC"].values, bounds_error=False,
                            fill_value=(df["SOC"].min(), df["SOC"].max()))
    f_soc_to_ocv = interp1d(df_soc["SOC"].values, df_soc["OCV"].values, bounds_error=False,
                            fill_value=(df_soc["OCV"].min(), df_soc["OCV"].max()))
    return f_ocv_to_soc, f_soc_to_ocv


def load_onori_lfp_gitt():
    """Load Onori's LFP GITT lookup from baseline features CSV."""
    onori_csv = os.path.join(REPO_ROOT, "baseline", "features_all_temperatures.csv")
    if not os.path.exists(onori_csv):
        logger.error(f"Stanford baseline features not found at: {onori_csv}")
        return None, None
    df = pd.read_csv(onori_csv)
    sub = df[df["Temp_Group"] == "25C"][["OCV_GITT_true", "SOC_true"]].drop_duplicates()
    ocv = sub["OCV_GITT_true"].values
    soc = sub["SOC_true"].values
    f_o2s, f_s2o = build_lookup_from_points(ocv, soc)
    logger.info(f"Loaded Onori LFP GITT: {len(sub)} points, OCV range [{ocv.min():.3f}, {ocv.max():.3f}] V")
    return f_o2s, f_s2o


def extract_features_from_cycling_df(df, time_col, volt_col, curr_col, cap_col,
                                     temp_col, cell_id, temp_group,
                                     f_ocv_to_soc, f_soc_to_ocv,
                                     total_cap=2.5, min_rest_duration=60,
                                     current_threshold=1e-3, t_step=30, t_max=600):
    df = df.copy()
    df["is_rest"] = (df[curr_col].abs() < current_threshold).astype(int)
    df["rest_group"] = (df["is_rest"] != df["is_rest"].shift()).cumsum()

    if temp_col not in df.columns:
        df[temp_col] = 24.0

    records = []
    for gid, grp in df.groupby("rest_group"):
        rest_duration = grp[time_col].iloc[-1] - grp[time_col].iloc[0]
        if grp["is_rest"].iloc[0] == 1 and rest_duration >= min_rest_duration:
            prev_id = gid - 1
            if prev_id in df["rest_group"].values:
                I_m = df[df["rest_group"] == prev_id][curr_col].mean()
            else:
                I_m = 0.0
            if pd.isna(I_m) or abs(I_m) < current_threshold:
                continue

            t_start = grp[time_col].iloc[0]
            grp = grp.copy()
            grp["t_rel"] = grp[time_col] - t_start

            v10_sub = grp[grp["t_rel"] >= 10]
            v30_sub = grp[grp["t_rel"] >= 30]
            if v10_sub.empty or v30_sub.empty:
                continue
            V_10 = float(v10_sub.iloc[0][volt_col])
            V_30 = float(v30_sub.iloc[0][volt_col])
            R = (V_30 - V_10) / I_m

            T_env = float(grp[temp_col].iloc[0])
            cap_at_rest = grp[cap_col].iloc[0]
            soc_true = float(np.clip(1.0 - (cap_at_rest / total_cap), 0.0, 1.0))
            OCV_GITT_true = float(f_soc_to_ocv(soc_true)) if f_soc_to_ocv else float("nan")

            for t_target in range(t_step, t_max + 1, t_step):
                sub_t = grp[grp["t_rel"] >= t_target]
                if sub_t.empty:
                    continue
                row_t = sub_t.iloc[0]
                v_mea = float(row_t[volt_col])
                records.append({
                    "Cell_ID": cell_id,
                    "Temp_Group": temp_group,
                    "t": t_target,
                    "V_mea": v_mea,
                    "SOC_mea": float(f_ocv_to_soc(v_mea)) if f_ocv_to_soc else float("nan"),
                    "V_10": V_10,
                    "I_m": float(I_m),
                    "R": float(R),
                    "I_flag": 1,
                    "T_env": T_env,
                    "dT": float(row_t[temp_col] - T_env),
                    "SOC_true": soc_true,
                    "OCV_GITT_true": OCV_GITT_true,
                })
    return pd.DataFrame(records)


def main():
    if not os.path.exists(EXTRACT_DIR):
        logger.error(f"Dataset directory not found: {EXTRACT_DIR}")
        return

    xlsx_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.xlsx")))
    logger.info(f"Found {len(xlsx_files)} CALCE A123 cycling files in {EXTRACT_DIR}")

    f_o2s, f_s2o = load_onori_lfp_gitt()
    if f_o2s is None:
        logger.error("Failed to load Onori LFP GITT")
        return

    all_features = []
    for fp in xlsx_files:
        fname = os.path.basename(fp)
        cell_id = "A123_094"
        temp_group = "25C"
        try:
            xl = pd.ExcelFile(fp)
            sheet = next((s for s in xl.sheet_names if "Channel" in s), xl.sheet_names[0])
            df = pd.read_excel(fp, sheet_name=sheet)
        except Exception as e:
            logger.warning(f"  Failed: {fname}: {e}")
            continue

        temp_col = next((c for c in df.columns if "Temperature" in c), None)
        if temp_col is None:
            df["T_env"] = 24.0
            temp_col = "T_env"

        feats = extract_features_from_cycling_df(
            df, "Test_Time(s)", "Voltage(V)", "Current(A)", "Discharge_Capacity(Ah)",
            temp_col, cell_id, temp_group, f_o2s, f_s2o,
            total_cap=2.5,
            min_rest_duration=60,
        )
        if not feats.empty:
            feats["Source_File"] = fname
            all_features.append(feats)
            logger.info(f"  {fname}: {len(feats)} samples")

    if not all_features:
        logger.error("No features extracted!")
        return

    final = pd.concat(all_features, ignore_index=True)
    final.to_csv(OUT_CSV, index=False)
    logger.info(f"CALCE A123: saved {len(final)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()

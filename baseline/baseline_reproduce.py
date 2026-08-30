"""
================================================================================
Stanford LFP Battery SOC Estimation Baseline Pipeline (Phase 1)
Reproducing: Che, Xu, Teodorescu, Hu, and Onori (ACS Energy Letters, 2025)
"Enhanced SOC Estimation for LFP Batteries: A Synergistic Approach Using 
Coulomb Counting Reset, Machine Learning, and Relaxation"

Target: Clean 3-Stage Cascaded Random Forest Architecture for SOC/OCV Resetting
================================================================================
"""

import os
import sys
import logging
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase1_Stanford_Baseline")


class GITTLookup:
    """
    Handles bidirectional OCV <-> SOC interpolation from GITT titration data.
    Enforces strict monotonic sorting and boundary clamping.
    """
    
    def __init__(
        self, 
        gitt_file_path: Optional[str] = None, 
        df_gitt: Optional[pd.DataFrame] = None, 
        nominal_capacity: float = 2.5
    ):
        self.nominal_capacity = nominal_capacity
        self.f_ocv_to_soc: Optional[interp1d] = None
        self.f_soc_to_ocv: Optional[interp1d] = None
        
        if gitt_file_path and os.path.exists(gitt_file_path):
            self.fit_from_file(gitt_file_path)
        elif df_gitt is not None:
            self.fit_from_dataframe(df_gitt)

    def fit_from_file(self, file_path: str) -> None:
        """Parses an Excel GITT workbook and extracts equilibrium points."""
        logger.info(f"Loading GITT calibration curve from file: {file_path}")
        xl = pd.ExcelFile(file_path)
        sheet = [s for s in xl.sheet_names if 'Channel' in s or 'Sheet1' in s or 'GITT' in s][0]
        df = pd.read_excel(file_path, sheet_name=sheet)
        self.fit_from_dataframe(df)

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Extracts rest equilibrium points from GITT step data or existing feature tables."""
        if 'Current(A)' in df.columns:
            rest_steps = df[df['Current(A)'].abs() < 1e-3]
            ocv_points = rest_steps.groupby('Step_Index').last()
            cap_col = 'Discharge_Capacity(Ah)' if 'Discharge_Capacity(Ah)' in ocv_points.columns else ocv_points.columns[1]
            volt_col = 'Voltage(V)' if 'Voltage(V)' in ocv_points.columns else ocv_points.columns[0]
            
            total_cap = df[cap_col].max() if df[cap_col].max() > 0 else self.nominal_capacity
            soc_arr = np.clip(1.0 - (ocv_points[cap_col].values / total_cap), 0.0, 1.0)
            ocv_arr = ocv_points[volt_col].values
        elif 'OCV_GITT_true' in df.columns and 'SOC_true' in df.columns:
            clean = df[['OCV_GITT_true', 'SOC_true']].drop_duplicates().sort_values('OCV_GITT_true')
            ocv_arr = clean['OCV_GITT_true'].values
            soc_arr = clean['SOC_true'].values
        else:
            raise ValueError("Invalid DataFrame structure. Must contain raw GITT steps or [OCV_GITT_true, SOC_true].")

        # Build clean monotonic lookup frame
        df_lookup = pd.DataFrame({'OCV': ocv_arr, 'SOC': soc_arr}).dropna()
        df_lookup = df_lookup.drop_duplicates(subset=['OCV']).sort_values('OCV')
        
        # Ensure strict monotonicity for inverse lookup (SOC -> OCV)
        df_lookup_soc = df_lookup.drop_duplicates(subset=['SOC']).sort_values('SOC')
        
        self.f_ocv_to_soc = interp1d(
            df_lookup['OCV'].values, 
            df_lookup['SOC'].values, 
            bounds_error=False, 
            fill_value=(df_lookup['SOC'].min(), df_lookup['SOC'].max())
        )
        self.f_soc_to_ocv = interp1d(
            df_lookup_soc['SOC'].values, 
            df_lookup_soc['OCV'].values, 
            bounds_error=False, 
            fill_value=(df_lookup_soc['OCV'].min(), df_lookup_soc['OCV'].max())
        )
        logger.info(f"GITT Lookup initialized successfully with {len(df_lookup)} equilibrium points.")

    def ocv_to_soc(self, ocv: Union[float, np.ndarray]) -> np.ndarray:
        return np.asarray(self.f_ocv_to_soc(ocv), dtype=np.float64)

    def soc_to_ocv(self, soc: Union[float, np.ndarray]) -> np.ndarray:
        return np.asarray(self.f_soc_to_ocv(soc), dtype=np.float64)


class BatteryFeatureExtractor:
    """
    Extracts the official 9-dimensional relaxation telemetry feature vector F:
    F = [t, V_mea, SOC_mea, V_10, I_m, R, I_flag, T_env, dT]
    """
    
    FEATURE_COLS = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

    def __init__(self, gitt_lookup_map: Dict[str, GITTLookup], nominal_capacity: float = 2.5):
        self.gitt_lookup_map = gitt_lookup_map
        self.nominal_capacity = nominal_capacity

    def extract_from_cycling_dataframe(self, df: pd.DataFrame, cell_id: str, temp_group: str) -> pd.DataFrame:
        """Processes raw battery cycling data containing rest/relaxation steps."""
        gitt = self.gitt_lookup_map.get(temp_group)
        if gitt is None:
            raise KeyError(f"No GITTLookup found for temperature group: {temp_group}")

        df['is_rest'] = (df['Current(A)'].abs() < 1e-3).astype(int)
        df['rest_group'] = (df['is_rest'] != df['is_rest'].shift()).cumsum()
        
        cap_col = 'Discharge_Capacity(Ah)' if 'Discharge_Capacity(Ah)' in df.columns else 'Capacity(Ah)'
        total_cap = df[cap_col].max() if df[cap_col].max() > 0 else self.nominal_capacity
        
        temp_candidate = [c for c in df.columns if 'Temp' in c or 'Aux' in c]
        temp_col = temp_candidate[0] if temp_candidate else None

        records = []
        for group_id, group in df.groupby('rest_group'):
            rest_duration = group['Test_Time(s)'].iloc[-1] - group['Test_Time(s)'].iloc[0]
            
            # Filter for rest segments lasting at least 300 seconds
            if group['is_rest'].iloc[0] == 1 and rest_duration >= 300:
                prev_group_id = group_id - 1
                if prev_group_id in df['rest_group'].values:
                    i_m = df[df['rest_group'] == prev_group_id]['Current(A)'].mean()
                else:
                    i_m = 0.0
                
                # Exclude rests without active operational history
                if abs(i_m) < 1e-3:
                    continue

                t_start = group['Test_Time(s)'].iloc[0]
                t_rel = group['Test_Time(s)'] - t_start
                
                v10_sub = group[t_rel >= 10]
                v30_sub = group[t_rel >= 30]
                if v10_sub.empty or v30_sub.empty:
                    continue

                v_10 = v10_sub.iloc[0]['Voltage(V)']
                v_30 = v30_sub.iloc[0]['Voltage(V)']
                r_val = (v_30 - v_10) / i_m
                t_env = group[temp_col].iloc[0] if temp_col else 25.0

                soc_true = np.clip(1.0 - (group[cap_col].iloc[0] / total_cap), 0.0, 1.0)
                ocv_gitt_true = float(gitt.soc_to_ocv(soc_true))

                # Sample relaxation telemetry at 30-second steps (30s to 600s)
                for t_target in range(30, 601, 30):
                    sub_t = group[t_rel >= t_target]
                    if sub_t.empty:
                        continue
                    row_t = sub_t.iloc[0]
                    v_mea = row_t['Voltage(V)']
                    soc_mea = float(gitt.ocv_to_soc(v_mea))
                    d_t = (row_t[temp_col] - t_env) if temp_col else 0.0

                    records.append({
                        'Cell_ID': cell_id,
                        'Temp_Group': temp_group,
                        't': t_target,
                        'V_mea': v_mea,
                        'SOC_mea': soc_mea,
                        'V_10': v_10,
                        'I_m': i_m,
                        'R': r_val,
                        'I_flag': 1.0 if i_m >= 0 else -1.0,
                        'T_env': t_env,
                        'dT': d_t,
                        'SOC_true': soc_true,
                        'OCV_GITT_true': ocv_gitt_true,
                        'V_diff': ocv_gitt_true - v_mea,
                        'SOC_diff': soc_true - soc_mea
                    })

        return pd.DataFrame(records)


class ThreeStageRFPipeline(BaseEstimator, RegressorMixin):
    """
    Cascaded 3-Stage Random Forest Regressor Pipeline:
      - Sub-model 1: Predicts V_diff = OCV_true - V_mea  --> Generates (OCV_1, SOC_1)
      - Sub-model 2: Predicts SOC_diff = SOC_true - SOC_mea --> Generates (SOC_2, OCV_2)
      - Sub-model 3: Cascaded Fusion                      --> Outputs final (SOC*, OCV*)
    """

    def __init__(
        self,
        gitt_lookup: GITTLookup,
        n_estimators_s1: int = 100,
        n_estimators_s2: int = 100,
        n_estimators_s3: int = 120,
        max_depth: int = 15,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.gitt_lookup = gitt_lookup
        self.n_estimators_s1 = n_estimators_s1
        self.n_estimators_s2 = n_estimators_s2
        self.n_estimators_s3 = n_estimators_s3
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.rf1: Optional[RandomForestRegressor] = None
        self.rf2: Optional[RandomForestRegressor] = None
        self.rf3_soc: Optional[RandomForestRegressor] = None
        self.rf3_ocv: Optional[RandomForestRegressor] = None

    def fit(self, X: pd.DataFrame, y_soc: np.ndarray, y_ocv: np.ndarray) -> "ThreeStageRFPipeline":
        """Trains Sub-models 1, 2, and 3 sequentially."""
        F = X[BatteryFeatureExtractor.FEATURE_COLS].copy()
        v_mea = F['V_mea'].values
        soc_mea = F['SOC_mea'].values

        v_diff = y_ocv - v_mea
        soc_diff = y_soc - soc_mea

        # Stage 1: Train Sub-model 1
        logger.info("Training Sub-model 1 (Voltage Overpotential Correction)...")
        self.rf1 = RandomForestRegressor(
            n_estimators=self.n_estimators_s1,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.rf1.fit(F, v_diff)
        
        v_diff_pred_1 = self.rf1.predict(F)
        ocv_1 = v_mea + v_diff_pred_1
        soc_1 = self.gitt_lookup.ocv_to_soc(ocv_1)

        # Stage 2: Train Sub-model 2
        logger.info("Training Sub-model 2 (Direct SOC Offset Correction)...")
        X_s2 = F.copy()
        X_s2['OCV_1'] = ocv_1
        X_s2['SOC_1'] = soc_1

        self.rf2 = RandomForestRegressor(
            n_estimators=self.n_estimators_s2,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.rf2.fit(X_s2, soc_diff)
        
        soc_diff_pred_2 = self.rf2.predict(X_s2)
        soc_2 = soc_mea + soc_diff_pred_2
        ocv_2 = self.gitt_lookup.soc_to_ocv(soc_2)

        # Stage 3: Train Sub-model 3 (Fusion)
        logger.info("Training Sub-model 3 (Model Fusion Estimators)...")
        X_s3 = X_s2.copy()
        X_s3['SOC_2'] = soc_2
        X_s3['OCV_2'] = ocv_2

        self.rf3_soc = RandomForestRegressor(
            n_estimators=self.n_estimators_s3,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.rf3_soc.fit(X_s3, y_soc)

        self.rf3_ocv = RandomForestRegressor(
            n_estimators=self.n_estimators_s3,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.rf3_ocv.fit(X_s3, y_ocv)
        
        logger.info("Three-stage RF pipeline training completed.")
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Infers estimated SOC* and OCV* across all three sequential stages."""
        F = X[BatteryFeatureExtractor.FEATURE_COLS].copy()
        v_mea = F['V_mea'].values
        soc_mea = F['SOC_mea'].values

        # Stage 1 Inference
        v_diff_pred_1 = self.rf1.predict(F)
        ocv_1 = v_mea + v_diff_pred_1
        soc_1 = self.gitt_lookup.ocv_to_soc(ocv_1)

        # Stage 2 Inference
        X_s2 = F.copy()
        X_s2['OCV_1'] = ocv_1
        X_s2['SOC_1'] = soc_1
        soc_diff_pred_2 = self.rf2.predict(X_s2)
        soc_2 = soc_mea + soc_diff_pred_2
        ocv_2 = self.gitt_lookup.soc_to_ocv(soc_2)

        # Stage 3 Inference
        X_s3 = X_s2.copy()
        X_s3['SOC_2'] = soc_2
        X_s3['OCV_2'] = ocv_2
        soc_star = self.rf3_soc.predict(X_s3)
        ocv_star = self.rf3_ocv.predict(X_s3)

        return soc_star, ocv_star


class ModelEvaluator:
    """Computes exact empirical evaluation metrics for benchmark validation."""
    
    @staticmethod
    def evaluate(
        y_soc_true: np.ndarray, 
        y_soc_pred: np.ndarray, 
        y_ocv_true: np.ndarray, 
        y_ocv_pred: np.ndarray,
        tag: str = "Benchmark Evaluation"
    ) -> Dict[str, float]:
        """Calculates percentage errors for SOC and millivolt errors for OCV."""
        soc_true_pct = y_soc_true * 100.0
        soc_pred_pct = y_soc_pred * 100.0
        
        ocv_true_mv = y_ocv_true * 1000.0
        ocv_pred_mv = y_ocv_pred * 1000.0

        metrics = {
            'SOC_MAE_pct': mean_absolute_error(soc_true_pct, soc_pred_pct),
            'SOC_RMSE_pct': float(np.sqrt(mean_squared_error(soc_true_pct, soc_pred_pct))),
            'SOC_MaxErr_pct': float(np.max(np.abs(soc_true_pct - soc_pred_pct))),
            'OCV_MAE_mV': mean_absolute_error(ocv_true_mv, ocv_pred_mv),
            'OCV_RMSE_mV': float(np.sqrt(mean_squared_error(ocv_true_mv, ocv_pred_mv))),
            'OCV_MaxErr_mV': float(np.max(np.abs(ocv_true_mv - ocv_pred_mv)))
        }

        print(f"\n=======================================================")
        print(f"📊 {tag}")
        print(f"=======================================================")
        print(f" • SOC Mean Absolute Error (MAE)   : {metrics['SOC_MAE_pct']:6.3f} %")
        print(f" • SOC Root Mean Square Error (RMSE): {metrics['SOC_RMSE_pct']:6.3f} %")
        print(f" • SOC Maximum Absolute Error      : {metrics['SOC_MaxErr_pct']:6.3f} %")
        print(f"-------------------------------------------------------")
        print(f" • OCV Mean Absolute Error (MAE)   : {metrics['OCV_MAE_mV']:6.3f} mV")
        print(f" • OCV Root Mean Square Error (RMSE): {metrics['OCV_RMSE_mV']:6.3f} mV")
        print(f" • OCV Maximum Absolute Error      : {metrics['OCV_MaxErr_mV']:6.3f} mV")
        print(f"=======================================================\n")
        return metrics


def run_baseline_pipeline(csv_path: str):
    """
    Executes full validation protocol reproducing Stanford Table 1 & Table 2:
      1. Baseline Naive Lookup Evaluation
      2. Unseen Cells (YX07, YX08) 3-Stage RF Evaluation
    """
    if not os.path.exists(csv_path):
        logger.error(f"Target data file '{csv_path}' not found. Please verify working directory.")
        return

    logger.info(f"Loading extracted feature database from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Filter standard 25°C baseline data
    if 'Temp_Group' in df.columns and '25C' in df['Temp_Group'].values:
        df_eval = df[df['Temp_Group'] == '25C'].copy()
    else:
        df_eval = df.copy()

    # 2. Fit GITT Lookup model
    gitt_lookup = GITTLookup(df_gitt=df_eval)

    # 3. Evaluate Direct Lookup (Naive method without ML correction)
    ModelEvaluator.evaluate(
        y_soc_true=df_eval['SOC_true'].values,
        y_soc_pred=df_eval['SOC_mea'].values,
        y_ocv_true=df_eval['OCV_GITT_true'].values,
        y_ocv_pred=df_eval['V_mea'].values,
        tag="Naive Direct Lookup (Without ML Correction)"
    )

    # 4. Partition into Train (Cells 1-6) and Unseen Test (Cells 7 & 8)
    train_cells = ['YX01', 'YX02', 'YX03', 'YX04', 'YX05', 'YX06']
    train_df = df_eval[df_eval['Cell_ID'].isin(train_cells)].copy()

    if train_df.empty:
        logger.warning("Named Cell IDs (YX01-YX06) not found. Performing 80/20 train/test split.")
        unique_cells = df_eval['Cell_ID'].unique()
        split_idx = int(len(unique_cells) * 0.75)
        train_df = df_eval[df_eval['Cell_ID'].isin(unique_cells[:split_idx])].copy()
        test_cells = unique_cells[split_idx:]
    else:
        test_cells = ['YX07', 'YX08']

    logger.info(f"Training dataset: {len(train_df)} samples across cells: {train_df['Cell_ID'].unique().tolist()}")

    # 5. Fit Three-Stage RF Model
    pipeline = ThreeStageRFPipeline(gitt_lookup=gitt_lookup, random_state=42)
    pipeline.fit(
        X=train_df,
        y_soc=train_df['SOC_true'].values,
        y_ocv=train_df['OCV_GITT_true'].values
    )

    # 6. Evaluate on Unseen Cells
    for cell in test_cells:
        test_df = df_eval[df_eval['Cell_ID'] == cell].copy()
        if test_df.empty:
            logger.warning(f"Unseen Cell '{cell}' has no samples. Skipping.")
            continue
        
        soc_pred, ocv_pred = pipeline.predict(test_df)
        ModelEvaluator.evaluate(
            y_soc_true=test_df['SOC_true'].values,
            y_soc_pred=soc_pred,
            y_ocv_true=test_df['OCV_GITT_true'].values,
            y_ocv_pred=ocv_pred,
            tag=f"Stanford Baseline: Unseen Cell {cell} Benchmark"
        )


if __name__ == "__main__":
    DATA_FILE = "features_all_temperatures.csv"
    run_baseline_pipeline(DATA_FILE)
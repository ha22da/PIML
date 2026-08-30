import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import interp1d
import os

# ۱. بارگذاری داده‌ها
csv_path = r"H:\battery\features_all_temperatures.csv"
gitt_folder = r"H:\battery\ML_SOC_Estimation_ACS_Energy_Letters\Raw_data\2_GITT_test"

df = pd.read_csv(csv_path)

# فقط داده‌های ۲۵ درجه برای ارزیابی استاندارد Unseen Cells
df_25 = df[df['Temp_Group'] == '25C'].copy()

# ساخت تابع GITT دمای ۲۵ درجه
gitt_file = [f for f in os.listdir(gitt_folder) if '25Deg' in f and f.endswith('.xlsx')][0]
gitt_path = os.path.join(gitt_folder, gitt_file)
xl = pd.ExcelFile(gitt_path)
sheet = [s for s in xl.sheet_names if 'Channel' in s][0]
df_gitt = pd.read_excel(gitt_path, sheet_name=sheet)
rest_steps = df_gitt[df_gitt['Current(A)'].abs() < 1e-3]
ocv_points = rest_steps.groupby('Step_Index').last()
df_lookup = pd.DataFrame({'OCV': ocv_points['Voltage(V)'], 'SOC': 1 - (ocv_points['Discharge_Capacity(Ah)'] / 2.5)}).dropna().drop_duplicates('OCV').sort_values('OCV')

f_ocv_to_soc = interp1d(df_lookup['OCV'], df_lookup['SOC'], bounds_error=False, fill_value=(0.0, 1.0))
f_soc_to_ocv = interp1d(df_lookup['SOC'], df_lookup['OCV'], bounds_error=False, fill_value=(df_lookup['OCV'].min(), df_lookup['OCV'].max()))

df_25['V_diff'] = df_25['OCV_GITT_true'] - df_25['V_mea']
df_25['SOC_diff'] = df_25['SOC_true'] - df_25['SOC_mea']
feature_cols = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

# آموزش روی سلول‌های YX01 تا YX06
train_df = df_25[df_25['Cell_ID'].isin(['YX01', 'YX02', 'YX03', 'YX04', 'YX05', 'YX06'])].copy()

# Sub-model 1
rf1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf1.fit(train_df[feature_cols], train_df['V_diff'])
ocv1_tr = train_df['V_mea'].values + rf1.predict(train_df[feature_cols])
soc1_tr = f_ocv_to_soc(ocv1_tr)

# Sub-model 2
X_tr2 = train_df[feature_cols].copy()
X_tr2['OCV_1'], X_tr2['SOC_1'] = ocv1_tr, soc1_tr
rf2 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf2.fit(X_tr2, train_df['SOC_diff'])
soc2_tr = train_df['SOC_mea'].values + rf2.predict(X_tr2)
ocv2_tr = f_soc_to_ocv(soc2_tr)

# Sub-model 3
X_tr3 = X_tr2.copy()
X_tr3['SOC_2'], X_tr3['OCV_2'] = soc2_tr, ocv2_tr

rf3_soc = RandomForestRegressor(n_estimators=120, max_depth=15, random_state=42, n_jobs=-1)
rf3_soc.fit(X_tr3, train_df['SOC_true'])

rf3_ocv = RandomForestRegressor(n_estimators=120, max_depth=15, random_state=42, n_jobs=-1)
rf3_ocv.fit(X_tr3, train_df['OCV_GITT_true'])

print("==========================================================")
print("🎯 نتایج نهایی و رسمی بازتولید روی سلول‌های نادیده (Unseen Cells 7 & 8):")
print("==========================================================")

for cell in ['YX07', 'YX08']:
    test_df = df_25[df_25['Cell_ID'] == cell].copy()
    if test_df.empty:
        continue
        
    X_te1 = test_df[feature_cols]
    ocv1_te = test_df['V_mea'].values + rf1.predict(X_te1)
    soc1_te = f_ocv_to_soc(ocv1_te)
    
    X_te2 = X_te1.copy()
    X_te2['OCV_1'], X_te2['SOC_1'] = ocv1_te, soc1_te
    soc2_te = test_df['SOC_mea'].values + rf2.predict(X_te2)
    ocv2_te = f_soc_to_ocv(soc2_te)
    
    X_te3 = X_te2.copy()
    X_te3['SOC_2'], X_te3['OCV_2'] = soc2_te, ocv2_te
    
    soc_pred = rf3_soc.predict(X_te3)
    ocv_pred = rf3_ocv.predict(X_te3)
    
    mae_soc = mean_absolute_error(test_df['SOC_true'] * 100, soc_pred * 100)
    rmse_soc = np.sqrt(mean_squared_error(test_df['SOC_true'] * 100, soc_pred * 100))
    mae_ocv = mean_absolute_error(test_df['OCV_GITT_true'], ocv_pred) * 1000
    rmse_ocv = np.sqrt(mean_squared_error(test_df['OCV_GITT_true'], ocv_pred)) * 1000
    
    print(f"📌 نتایج سلول {cell}:")
    print(f"   - SOC MAE : {mae_soc:.2f} %   | (تارگت مقاله: < 4.30 %)")
    print(f"   - SOC RMSE: {rmse_soc:.2f} %")
    print(f"   - OCV MAE : {mae_ocv:.2f} mV  | (تارگت مقاله: < 3.00 mV)")
    print(f"   - OCV RMSE: {rmse_ocv:.2f} mV")
    print("-" * 58)
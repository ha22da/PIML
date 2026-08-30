import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import glob

# مسیرهای اصلی داده‌ها
base_raw_dir = r"H:\battery\ML_SOC_Estimation_ACS_Energy_Letters\Raw_data"
output_csv = r"H:\battery\features_all_temperatures.csv"

# ۱. نگاشت پوشه‌ها و فایل‌های GITT مربوط به هر دما
temp_folders = {
    '10C': os.path.join(base_raw_dir, "3_10Deg_experiment"),
    '25C': os.path.join(base_raw_dir, "4_25Deg_experiment"),
    '35C': os.path.join(base_raw_dir, "5_35Deg_experiment"),
    '45C': os.path.join(base_raw_dir, "6_45Deg_experiment")
}

gitt_files = {
    '10C': os.path.join(base_raw_dir, "2_GITT_test", "20231012_YX06_10Deg_Channel_6.xlsx"),
    '25C': os.path.join(base_raw_dir, "2_GITT_test", "20231012_YX06_25Deg_Channel_6.xlsx"),
    '35C': os.path.join(base_raw_dir, "2_GITT_test", "20231012_YX06_35Deg_Channel_6.xlsx"),
    '45C': os.path.join(base_raw_dir, "2_GITT_test", "20231012_YX06_45Deg_Channel_6.xlsx")
}

def build_gitt_lookup(gitt_file_path):
    if not os.path.exists(gitt_file_path):
        return None, None
    xl = pd.ExcelFile(gitt_file_path)
    sheet = [s for s in xl.sheet_names if 'Channel' in s][0]
    df_gitt = pd.read_excel(gitt_file_path, sheet_name=sheet)
    rest_steps = df_gitt[df_gitt['Current(A)'].abs() < 1e-3]
    ocv_points = rest_steps.groupby('Step_Index').last()
    total_cap = df_gitt['Discharge_Capacity(Ah)'].max()
    if total_cap == 0 or pd.isna(total_cap):
        total_cap = 2.5
    soc_list = 1 - (ocv_points['Discharge_Capacity(Ah)'] / total_cap)
    ocv_list = ocv_points['Voltage(V)']
    df_lookup = pd.DataFrame({'OCV': ocv_list, 'SOC': soc_list}).dropna().drop_duplicates(subset=['OCV']).sort_values('OCV')
    f_ocv_to_soc = interp1d(df_lookup['OCV'], df_lookup['SOC'], bounds_error=False, fill_value=(0.0, 1.0))
    f_soc_to_ocv = interp1d(df_lookup['SOC'], df_lookup['OCV'], bounds_error=False, fill_value=(df_lookup['OCV'].min(), df_lookup['OCV'].max()))
    return f_ocv_to_soc, f_soc_to_ocv

def process_file(file_path, f_ocv_to_soc, f_soc_to_ocv, temp_label):
    fname = os.path.basename(file_path)
    cell_id = fname.split('_')[1] if '_' in fname else "YX01"
    
    xl = pd.ExcelFile(file_path)
    sheet = [s for s in xl.sheet_names if 'Channel' in s][0]
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    df['is_rest'] = (df['Current(A)'].abs() < 1e-3).astype(int)
    df['rest_group'] = (df['is_rest'] != df['is_rest'].shift()).cumsum()
    
    total_cap = df['Discharge_Capacity(Ah)'].max()
    if total_cap == 0 or pd.isna(total_cap):
        total_cap = 2.5
        
    records = []
    for group_id, group in df.groupby('rest_group'):
        rest_duration = group['Test_Time(s)'].iloc[-1] - group['Test_Time(s)'].iloc[0]
        if group['is_rest'].iloc[0] == 1 and rest_duration >= 300:
            prev_group_id = group_id - 1
            if prev_group_id in df['rest_group'].values:
                I_m = df[df['rest_group'] == prev_group_id]['Current(A)'].mean()
            else:
                I_m = 0.0
            if abs(I_m) < 1e-3:
                continue
                
            t_start = group['Test_Time(s)'].iloc[0]
            group = group.copy()
            group['t_rel'] = group['Test_Time(s)'] - t_start
            
            v10_sub = group[group['t_rel'] >= 10]
            v30_sub = group[group['t_rel'] >= 30]
            if v10_sub.empty or v30_sub.empty:
                continue
            V_10, V_30 = v10_sub.iloc[0]['Voltage(V)'], v30_sub.iloc[0]['Voltage(V)']
            R = (V_30 - V_10) / I_m
            
            temp_col = [c for c in group.columns if 'Aux_Temperature' in c][0]
            T_env = group[temp_col].iloc[0]
            SOC_true = max(0.0, min(1.0, 1.0 - (group['Discharge_Capacity(Ah)'].iloc[0] / total_cap)))
            OCV_GITT_true = float(f_soc_to_ocv(SOC_true))
            
            for t_target in range(30, 601, 30):
                sub_t = group[group['t_rel'] >= t_target]
                if sub_t.empty:
                    continue
                row_t = sub_t.iloc[0]
                records.append({
                    'Cell_ID': cell_id,
                    'Temp_Group': temp_label,
                    't': t_target,
                    'V_mea': row_t['Voltage(V)'],
                    'SOC_mea': float(f_ocv_to_soc(row_t['Voltage(V)'])),
                    'V_10': V_10,
                    'I_m': I_m,
                    'R': R,
                    'I_flag': 1,
                    'T_env': T_env,
                    'dT': row_t[temp_col] - T_env,
                    'SOC_true': SOC_true,
                    'OCV_GITT_true': OCV_GITT_true
                })
    return pd.DataFrame(records)

# استخراج کل داده‌ها
all_data = []
for temp_label, folder_path in temp_folders.items():
    if not os.path.exists(folder_path):
        continue
    print(f"\n--- در حال استخراج ویژگی‌های دمای {temp_label} ---")
    f_ocv_to_soc, f_soc_to_ocv = build_gitt_lookup(gitt_files[temp_label])
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    for i, fpath in enumerate(excel_files, 1):
        print(f"[{i}/{len(excel_files)}] {os.path.basename(fpath)}")
        try:
            df_rec = process_file(fpath, f_ocv_to_soc, f_soc_to_ocv, temp_label)
            if not df_rec.empty:
                all_data.append(df_rec)
        except Exception as e:
            print(f"   --> خطا: {e}")

final_all = pd.concat(all_data, ignore_index=True)
final_all.to_csv(output_csv, index=False)
print(f"\n✅ دیتابیس جامع تمام دماها ذخیره شد با {len(final_all)} سطر در مسیر {output_csv}")
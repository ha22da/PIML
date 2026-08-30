import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import interp1d

# ==========================================
# ۱. بارگذاری دیتابیس استخراج‌شده
# ==========================================
csv_path = r"H:\battery\features_25C.csv"
gitt_path = r"H:\battery\ML_SOC_Estimation_ACS_Energy_Letters\Raw_data\2_GITT_test\20231012_YX06_25Deg_Channel_6.xlsx"

print("۱. بارگذاری دیتابیس ویژگی‌ها...")
df = pd.read_csv(csv_path)

# ساخت توابع درایابی GITT برای تبدیل OCV به SOC و برعکس
xl = pd.ExcelFile(gitt_path)
data_sheet = [s for s in xl.sheet_names if 'Channel' in s][0]
df_gitt = pd.read_excel(gitt_path, sheet_name=data_sheet)
rest_steps = df_gitt[df_gitt['Current(A)'].abs() < 1e-3]
ocv_points = rest_steps.groupby('Step_Index').last()
total_cap = df_gitt['Discharge_Capacity(Ah)'].max() if df_gitt['Discharge_Capacity(Ah)'].max() > 0 else 2.5

soc_list = 1 - (ocv_points['Discharge_Capacity(Ah)'] / total_cap)
ocv_list = ocv_points['Voltage(V)']
df_lookup = pd.DataFrame({'OCV': ocv_list, 'SOC': soc_list}).dropna().drop_duplicates(subset=['OCV']).sort_values('OCV')

f_ocv_to_soc = interp1d(df_lookup['OCV'], df_lookup['SOC'], bounds_error=False, fill_value=(0.0, 1.0))
f_soc_to_ocv = interp1d(df_lookup['SOC'], df_lookup['OCV'], bounds_error=False, fill_value=(df_lookup['OCV'].min(), df_lookup['OCV'].max()))

# ==========================================
# ۲. تعریف ستون‌های ویژگی (Feature Vector F)
# ==========================================
feature_cols = ['t', 'V_mea', 'SOC_mea', 'V_10', 'I_m', 'R', 'I_flag', 'T_env', 'dT']

# محاسبه برچسب‌های میانی
df['V_diff'] = df['OCV_GITT_true'] - df['V_mea']
df['SOC_diff'] = df['SOC_true'] - df['SOC_mea']

# ==========================================
# ۳. تقسیم داده‌ها به Train و Test (استراتژی Case Split)
# ==========================================
# استفاده از 70% داده‌ها برای آموزش و 30% برای تست
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"تعداد داده‌های آموزش (Train): {len(train_df)}")
print(f"تعداد داده‌های ارزیابی (Test): {len(test_df)}")

# ==========================================
# ۴. آموزش Sub-model 1
# ==========================================
print("\nدر حال آموزش Sub-model 1 (پیش‌بینی V_diff)...")
rf1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf1.fit(X_train, train_df['V_diff'])

# خروجی‌های میانی Sub-model 1 برای Train
v_diff_pred_train = rf1.predict(X_train)
ocv1_train = train_df['V_mea'] + v_diff_pred_train
soc1_train = f_ocv_to_soc(ocv1_train)

# خروجی‌های میانی Sub-model 1 برای Test
v_diff_pred_test = rf1.predict(X_test)
ocv1_test = test_df['V_mea'] + v_diff_pred_test
soc1_test = f_ocv_to_soc(ocv1_test)

# ==========================================
# ۵. آموزش Sub-model 2
# ==========================================
print("در حال آموزش Sub-model 2 (پیش‌بینی SOC_diff)...")
X_train_sub2 = X_train.copy()
X_train_sub2['OCV_1'] = ocv1_train
X_train_sub2['SOC_1'] = soc1_train

X_test_sub2 = X_test.copy()
X_test_sub2['OCV_1'] = ocv1_test
X_test_sub2['SOC_1'] = soc1_test

rf2 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf2.fit(X_train_sub2, train_df['SOC_diff'])

# خروجی‌های میانی Sub-model 2 برای Train
soc_diff_pred_train = rf2.predict(X_train_sub2)
soc2_train = train_df['SOC_mea'] + soc_diff_pred_train
ocv2_train = f_soc_to_ocv(soc2_train)

# خروجی‌های میانی Sub-model 2 برای Test
soc_diff_pred_test = rf2.predict(X_test_sub2)
soc2_test = test_df['SOC_mea'] + soc_diff_pred_test
ocv2_test = f_soc_to_ocv(soc2_test)

# ==========================================
# ۶. آموزش Sub-model 3 (Model Fusion)
# ==========================================
print("در حال آموزش Sub-model 3 (تلفیق نهایی)...")
X_train_sub3 = X_train_sub2.copy()
X_train_sub3['SOC_2'] = soc2_train
X_train_sub3['OCV_2'] = ocv2_train

X_test_sub3 = X_test_sub2.copy()
X_test_sub3['SOC_2'] = soc2_test
X_test_sub3['OCV_2'] = ocv2_test

# برچسب‌های هدف نهایی
y_train_final = train_df[['SOC_true', 'OCV_GITT_true']]
y_test_final = test_df[['SOC_true', 'OCV_GITT_true']]

rf3 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf3.fit(X_train_sub3, y_train_final)

# ==========================================
# ۷. ارزیابی نتایج روی داده‌های تست
# ==========================================
predictions = rf3.predict(X_test_sub3)
soc_pred = predictions[:, 0]
ocv_pred = predictions[:, 1]

soc_true = y_test_final['SOC_true'].values
ocv_true = y_test_final['OCV_GITT_true'].values

# محاسبه معیارهای خطا
mae_soc = mean_absolute_error(soc_true, soc_pred) * 100  # به درصد
rmse_soc = np.sqrt(mean_squared_error(soc_true, soc_pred)) * 100

mae_ocv = mean_absolute_error(ocv_true, ocv_pred) * 1000  # به میلی‌ولت
rmse_ocv = np.sqrt(mean_squared_error(ocv_true, ocv_pred)) * 1000

print("\n==================================================")
print("🎯 نتایج بازتولید (Replication Results) روی داده‌های تست ۲۵ درجه:")
print("==================================================")
print(f"خطای مطلق میانگین SOC (MAE):  {mae_soc:.2f} %")
print(f"جذر میانگین مربعات خطای SOC (RMSE): {rmse_soc:.2f} %")
print("--------------------------------------------------")
print(f"خطای مطلق میانگین OCV (MAE):  {mae_ocv:.2f} mV")
print(f"جذر میانگین مربعات خطای OCV (RMSE): {rmse_ocv:.2f} mV")
print("==================================================")
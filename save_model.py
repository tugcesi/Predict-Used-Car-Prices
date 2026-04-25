"""
save_model.py – Used Car Price Predictor
Notebook pipeline ile birebir uyumlu.
Çalıştır: python save_model.py
"""
import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 1. Veri Yükleme ───────────────────────────────────────────────────────────
print("📥 Veri yükleniyor...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(f"   Train: {train.shape} | Test: {test.shape}")

df = pd.concat([train, test], ignore_index=False)

# ── 2. fuel_type (notebook ile birebir) ───────────────────────────────────────
def extract_fuel_from_engine(engine_str):
    s = str(engine_str).lower()
    if 'electric' in s:
        return 'Electric'
    elif 'diesel' in s:
        return 'Diesel'
    elif 'hybrid' in s or 'mild' in s:
        return 'Hybrid'
    elif 'flex' in s or 'e85' in s:
        return 'E85 Flex Fuel'
    elif 'gasoline' in s:
        return 'Gasoline'
    return np.nan

df['fuel_type'] = df.apply(
    lambda row: extract_fuel_from_engine(row['engine'])
    if pd.isnull(row['fuel_type']) else row['fuel_type'],
    axis=1
)

def fill_mode(x):
    mode = x.mode()
    return x.fillna(mode[0]) if len(mode) else x

df['fuel_type'] = df.groupby(['brand', 'model'])['fuel_type'].transform(fill_mode)
df['fuel_type'].fillna(df['fuel_type'].mode()[0], inplace=True)

# ── 3. accident & clean_title ─────────────────────────────────────────────────
df['accident'].fillna('Unknown', inplace=True)
df['clean_title'].fillna('Unknown', inplace=True)

# ── 4. Engine feature extraction ─────────────────────────────────────────────
df['horsepower']  = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d*)L').astype(float)
df['cylinders']   = df['engine'].str.extract(r'(\d+)\s*Cylinder').astype(float)

# Eksikler: brand+model → brand → global median
for col in ['horsepower', 'engine_size', 'cylinders']:
    df[col] = df.groupby(['brand', 'model'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df.groupby('brand')[col].transform(lambda x: x.fillna(x.median()))
    df[col].fillna(df[col].median(), inplace=True)

# ── 5. Feature Engineering (notebook ile birebir) ────────────────────────────
df['car_age']        = 2026 - df['model_year']
df['milage_per_year']= df['milage'] / df['car_age'].replace(0, 1)
df['has_accident']   = (df['accident'] == 'At least 1 accident or damage reported').astype(int)
df['is_clean_title'] = (df['clean_title'] == 'Yes').astype(int)
df['is_automatic']   = df['transmission'].str.contains(
    'A/T|Automatic|CVT', case=False, na=False
).astype(int)

LUXURY_BRANDS = [
    'BMW', 'Mercedes-Benz', 'Audi', 'Porsche', 'Lamborghini',
    'Ferrari', 'Bentley', 'Rolls-Royce', 'Maserati', 'Genesis',
    'Lexus', 'Cadillac', 'Lincoln', 'Land', 'Volvo', 'Jaguar'
]
df['is_luxury'] = df['brand'].isin(LUXURY_BRANDS).astype(int)

df['hp_per_cylinder'] = df['horsepower'] / df['cylinders'].replace(0, np.nan)
df['hp_per_cylinder'].fillna(df['hp_per_cylinder'].median(), inplace=True)

# ── 6. Train / Test Ayır ─────────────────────────────────────────────────────
train_df = df[df['price'].notna()]
test_df  = df[df['price'].isna()]

x = train_df.drop(['id', 'engine', 'price'], axis=1)
y = train_df['price']
x_final_test = test_df.drop(['id', 'engine', 'price'], axis=1)

# ── 7. Get Dummies (combined) ─────────────────────────────────────────────────
x_all = pd.concat([x, x_final_test])
x_all = pd.get_dummies(x_all, drop_first=True)
x           = x_all.iloc[:len(x)]
x_final_test= x_all.iloc[len(x):]

FEATURE_COLUMNS = x.columns.tolist()
print(f"   Toplam feature sayısı: {len(FEATURE_COLUMNS)}")

# ── 8. Validation ──────────────────────────────────────────────────────────────
print("\n📊 Validation (80/20)...")
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

eval_model = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)
eval_model.fit(x_tr, y_tr)

p = eval_model.predict(x_val)
print(f"   R²  : {r2_score(y_val, p):.4f}")
print(f"   RMSE: ${mean_squared_error(y_val, p)**0.5:,.0f}")
print(f"   MAE : ${mean_absolute_error(y_val, p):,.0f}")

# ── 9. Full Train ─────────────────────────────────────────────────────────────
print("\n🚀 Final model (full train)...")
best_model = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)
best_model.fit(x, y)

# ── 10. Imputation değerleri ve UI için kategoriler ──────────────────────────
# Bunlar inference sırasında aynı pipeline'ı uygulamak için gerekli
raw_train = pd.read_csv("train.csv")
raw_all   = pd.concat([raw_train, pd.read_csv("test.csv")], ignore_index=True)

encoders = {
    # UI dropdown listeleri
    'brands'       : sorted(raw_all['brand'].dropna().unique().tolist()),
    'models'       : sorted(raw_all['model'].dropna().unique().tolist()),
    'fuel_types'   : sorted(raw_all['fuel_type'].dropna().unique().tolist()),
    'ext_cols'     : sorted(raw_all['ext_col'].dropna().unique().tolist()),
    'int_cols'     : sorted(raw_all['int_col'].dropna().unique().tolist()),
    'transmissions': sorted(raw_all['transmission'].dropna().unique().tolist()),
    'luxury_brands': LUXURY_BRANDS,
    # Imputation değerleri
    'fuel_type_global_mode': df['fuel_type'].mode()[0],
    'brand_model_fuel'     : df.groupby(['brand', 'model'])['fuel_type']\
                               .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)\
                               .to_dict(),
    'brand_fuel'           : df.groupby('brand')['fuel_type']\
                               .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)\
                               .to_dict(),
    'hp_median'            : df['horsepower'].median(),
    'engine_size_median'   : df['engine_size'].median(),
    'cylinders_median'     : df['cylinders'].median(),
    'hp_per_cyl_median'    : df['hp_per_cylinder'].median(),
}

# ── 11. Kaydet ────────────────────────────────────────────────────────────────
joblib.dump(best_model,      'model.joblib')
joblib.dump(FEATURE_COLUMNS, 'feature_columns.joblib')
joblib.dump(encoders,        'encoders.joblib')

print("\n✅ Kaydedilen dosyalar:")
print("   model.joblib | feature_columns.joblib | encoders.joblib")
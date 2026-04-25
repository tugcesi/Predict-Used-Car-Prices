# 🚗 Used Car Price Predictor

> **Kaggle Playground Series S4E9** yarışması için geliştirilen makine öğrenmesi projesi.  
> İkinci el araçların özelliklerine göre satış fiyatını **LightGBM** regresyon modeliyle tahmin eder.

---

## 📌 Proje Hakkında

| | |
|---|---|
| **Görev** | Regression |
| **Hedef Değişken** | `price` (USD) |
| **Model** | LGBMRegressor |
| **Encoding** | `pd.get_dummies(drop_first=True)` |
| **Train Satır Sayısı** | 188,533 |
| **Toplam Satır (train+test)** | 314,223 |

---

## 📁 Dosya Yapısı

```
├── train.csv                    # Eğitim verisi (Kaggle'dan indirilmeli)
├── test.csv                     # Test verisi (Kaggle'dan indirilmeli)
├── save_model.py                # Model eğitimi ve artifact kaydetme
├── app.py                       # Streamlit uygulaması
├── requirements.txt             # Gerekli kütüphaneler
├── model.joblib                 # Eğitilmiş model       (save_model.py sonrası oluşur)
├── feature_columns.joblib       # Feature listesi       (save_model.py sonrası oluşur)
└── encoders.joblib              # Imputation & UI meta  (save_model.py sonrası oluşur)
```

---

## 🔧 Kurulum

```bash
# 1. Repoyu klonlayın
git clone https://github.com/tugcesi/Predict-Used-Car-Prices.git
cd Predict-Used-Car-Prices

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

Veri setini [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e9) üzerinden indirip proje kök dizinine yerleştirin:

```
train.csv
test.csv
```

> ⚠️ Veri setleri `.zip` olarak repoya eklenmiştir. Kullanmadan önce açın:
> ```bash
> unzip train.csv.zip
> unzip test.csv.zip
> ```

---

## 🚀 Kullanım

```bash
# Adım 1 — Modeli eğit (model.joblib, feature_columns.joblib, encoders.joblib oluşur)
python save_model.py

# Adım 2 — Uygulamayı başlat
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` açılır.

---

## ⚙️ Pipeline Detayları

### Eksik Değer Doldurma

| Sütun | Strateji |
|---|---|
| `fuel_type` | Engine string regex → brand+model modu → brand modu → global mod |
| `accident` | `fillna('Unknown')` |
| `clean_title` | `fillna('Unknown')` |
| `horsepower` | brand+model medyanı → brand medyanı → global medyan |
| `engine_size` | brand+model medyanı → brand medyanı → global medyan |
| `cylinders` | brand+model medyanı → brand medyanı → global medyan |

### Feature Engineering

| Özellik | Formül |
|---|---|
| `horsepower` | Engine regex → `(\d+\.?\d*)HP` |
| `engine_size` | Engine regex → `(\d+\.?\d*)L` |
| `cylinders` | Engine regex → `(\d+)\s*Cylinder` |
| `car_age` | `2026 - model_year` |
| `milage_per_year` | `milage / car_age` |
| `has_accident` | `"At least 1 accident..."` → 1, diğer → 0 |
| `is_clean_title` | `"Yes"` → 1, diğer → 0 |
| `is_automatic` | `A/T \| Automatic \| CVT` içeriyorsa → 1 |
| `is_luxury` | BMW, Mercedes-Benz, Audi, Porsche... listesi → 1 |
| `hp_per_cylinder` | `horsepower / cylinders` |

### Encoding

| Sütun Tipi | Yöntem |
|---|---|
| Kategorik (`brand`, `model`, `fuel_type`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`) | `pd.get_dummies(drop_first=True)` |
| Numerik | Doğrudan kullanım |

### Model

- **LGBMRegressor** — `n_jobs=-1, random_state=42`
- **Hedef:** Direkt `price` (log transform yok)
- **Train/Val split:** 80/20, `random_state=42`

---

## 📊 Model Karşılaştırması (Notebook)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **LightGBM** | **0.1644** | **68,166** | **19,719** |
| ElasticNet | 0.1471 | 68,872 | 20,816 |
| Lasso | 0.1397 | 69,167 | 21,482 |
| Ridge | 0.1396 | 69,172 | 21,492 |
| LinearRegression | 0.1396 | 69,172 | 21,492 |
| XGBoost | -0.0024 | 74,664 | 20,140 |
| ExtraTrees | -0.1569 | 80,211 | 21,715 |
| DecisionTree | -1.0565 | 106,942 | 26,731 |

> LightGBM tüm modeller arasında en yüksek R² ve en düşük RMSE/MAE değerlerine sahiptir.

---

## 📚 Veri Seti

- **Kaynak:** [Kaggle — Playground Series S4E9](https://www.kaggle.com/competitions/playground-series-s4e9)
- **Train:** 188,533 satır | 13 sütun
- **Test:** 125,690 satır | 12 sütun

**Ham Sütunlar:**

| Sütun | Tür | Açıklama |
|---|---|---|
| `brand` | object | Araç markası |
| `model` | object | Araç modeli |
| `model_year` | int | Model yılı |
| `milage` | int | Kilometre (mil) |
| `fuel_type` | object | Yakıt tipi |
| `engine` | object | Motor açıklaması (HP, L, Cylinder bilgisi içerir) |
| `transmission` | object | Şanzıman tipi |
| `ext_col` | object | Dış renk |
| `int_col` | object | İç renk |
| `accident` | object | Kaza geçmişi |
| `clean_title` | object | Temiz tapu |
| `price` | float | 🎯 Hedef değişken |

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly)

---

## 📄 Lisans

[MIT Lisansı](LICENSE)

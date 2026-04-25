"""
app.py – Used Car Price Predictor (Streamlit)
Notebook pipeline ile birebir uyumlu.
Çalıştır: streamlit run app.py
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ── Sayfa Ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ── Artifact Yükleme ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ['model.joblib', 'feature_columns.joblib', 'encoders.joblib']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, None, f"Eksik dosyalar: {missing}"
    model    = joblib.load('model.joblib')
    features = joblib.load('feature_columns.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, features, encoders, None

model, feature_columns, encoders, err = load_artifacts()

if err:
    st.error(f"⚠️ {err}")
    st.info("Önce şunu çalıştırın: `python save_model.py`")
    st.stop()

# ── Pipeline Fonksiyonları (save_model.py ile birebir aynı) ──────────────────
def extract_fuel_from_engine(engine_str: str):
    s = str(engine_str).lower()
    if 'electric' in s: return 'Electric'
    if 'diesel'   in s: return 'Diesel'
    if 'hybrid'   in s or 'mild' in s: return 'Hybrid'
    if 'flex'     in s or 'e85' in s:  return 'E85 Flex Fuel'
    if 'gasoline' in s: return 'Gasoline'
    return None


def build_input_df(inputs: dict, encoders: dict) -> pd.DataFrame:
    """
    Kullanıcı girdilerinden notebook pipeline'ındaki ile aynı feature satırını oluşturur.
    """
    brand        = inputs['brand']
    model_name   = inputs['model']
    model_year   = inputs['model_year']
    milage       = float(inputs['milage'])
    fuel_type    = inputs['fuel_type']
    engine_str   = inputs['engine']       # raw engine string
    transmission = inputs['transmission']
    ext_col      = inputs['ext_col']
    int_col      = inputs['int_col']
    accident     = inputs['accident']
    clean_title  = inputs['clean_title']

    # ── fuel_type (engine'den de doldur, sonra moda) ─────────────────────────
    if pd.isnull(fuel_type) or fuel_type == '':
        fuel_type = extract_fuel_from_engine(engine_str) or \
                    encoders.get('brand_model_fuel', {}).get((brand, model_name)) or \
                    encoders.get('brand_fuel', {}).get(brand) or \
                    encoders['fuel_type_global_mode']

    # ── Engine feature extraction ─────────────────────────────────────────────
    import re
    hp_match  = re.search(r'(\d+\.?\d*)HP',          engine_str, re.IGNORECASE)
    es_match  = re.search(r'(\d+\.?\d*)L',            engine_str, re.IGNORECASE)
    cyl_match = re.search(r'(\d+)\s*Cylinder',        engine_str, re.IGNORECASE)

    horsepower  = float(hp_match.group(1))  if hp_match  else encoders['hp_median']
    engine_size = float(es_match.group(1))  if es_match  else encoders['engine_size_median']
    cylinders   = float(cyl_match.group(1)) if cyl_match else encoders['cylinders_median']

    # ── Derived features ──────────────────────────────────────────────────────
    car_age         = 2026 - model_year
    milage_per_year = milage / max(car_age, 1)
    has_accident    = 1 if accident == 'At least 1 accident or damage reported' else 0
    is_clean_title  = 1 if clean_title == 'Yes' else 0
    is_automatic    = 1 if any(k in transmission for k in ['A/T', 'Automatic', 'CVT']) else 0
    is_luxury       = 1 if brand in encoders['luxury_brands'] else 0
    hp_per_cylinder = (horsepower / cylinders) if cylinders > 0 else encoders['hp_per_cyl_median']

    row = {
        'brand'          : brand,
        'model'          : model_name,
        'model_year'     : model_year,
        'milage'         : milage,
        'fuel_type'      : fuel_type,
        'transmission'   : transmission,
        'ext_col'        : ext_col,
        'int_col'        : int_col,
        'accident'       : accident,
        'clean_title'    : clean_title,
        'horsepower'     : horsepower,
        'engine_size'    : engine_size,
        'cylinders'      : cylinders,
        'car_age'        : car_age,
        'milage_per_year': milage_per_year,
        'has_accident'   : has_accident,
        'is_clean_title' : is_clean_title,
        'is_automatic'   : is_automatic,
        'is_luxury'      : is_luxury,
        'hp_per_cylinder': hp_per_cylinder,
    }

    df_row = pd.DataFrame([row])

    # ── get_dummies (training ile aynı kolonları oluştur) ─────────────────────
    df_row = pd.get_dummies(df_row, drop_first=True)

    # Eksik kolonları 0 ile doldur, fazla kolonları at
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)

    return df_row


# ── Gauge Chart ───────────────────────────────────────────────────────────────
def make_gauge(price: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        title={'text': "Tahmini Satış Fiyatı (USD)"},
        number={'prefix': "$", 'valueformat': ',.0f'},
        gauge={
            'axis': {'range': [0, 150_000], 'tickformat': '$,.0f'},
            'bar':  {'color': '#1976D2'},
            'steps': [
                {'range': [0,       30_000], 'color': '#C8E6C9'},
                {'range': [30_000,  70_000], 'color': '#FFF9C4'},
                {'range': [70_000, 150_000], 'color': '#FFCCBC'},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🚗 Used Car Price Predictor")
st.caption("Kaggle Playground Series S4E9 | LightGBM | ~0.16 R²")
st.divider()

# ── Sidebar: Kullanıcı Girdileri ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Araç Bilgileri")

    st.subheader("🏷️ Marka & Model")
    brand = st.selectbox("Marka", encoders['brands'])
    model_opts = encoders['models']
    model_name = st.selectbox("Model", model_opts)
    model_year = st.slider("Model Yılı", 1990, 2024, 2018)

    st.subheader("📏 Kilometre")
    milage = st.number_input(
        "Kilometre (mi)", min_value=0, max_value=500_000, value=50_000, step=1_000
    )

    st.subheader("⚙️ Motor (Engine String)")
    engine_str = st.text_input(
        "Engine",
        value="228.0HP 2.0L 4 Cylinder Engine Gasoline Fuel",
        help="Örn: '228.0HP 2.0L 4 Cylinder Engine Gasoline Fuel'"
    )
    st.caption("HP, motor hacmi ve silindir bu alandan otomatik çıkarılır.")

    st.subheader("⛽ Yakıt & Şanzıman")
    fuel_type    = st.selectbox(
        "Yakıt Tipi", ['(engine'den otomatik)'] + encoders['fuel_types']
    )
    if fuel_type == '(engine\'den otomatik)':
        fuel_type = ''

    transmission = st.selectbox("Şanzıman", encoders['transmissions'])

    st.subheader("🎨 Renk")
    ext_col = st.selectbox("Dış Renk", encoders['ext_cols'])
    int_col = st.selectbox("İç Renk",  encoders['int_cols'])

    st.subheader("📋 Durum")
    accident    = st.selectbox(
        "Kaza Geçmişi",
        ["None reported", "At least 1 accident or damage reported", "Unknown"]
    )
    clean_title = st.selectbox(
        "Temiz Tapu (Clean Title)", ["Yes", "No", "Unknown"]
    )

    predict_btn = st.button(
        "💰 Fiyat Tahmin Et", use_container_width=True, type="primary"
    )

# ── Tahmin ────────────────────────────────────────────────────────────────────
if predict_btn:
    inputs = dict(
        brand=brand, model=model_name, model_year=model_year,
        milage=milage, fuel_type=fuel_type, engine=engine_str,
        transmission=transmission, ext_col=ext_col, int_col=int_col,
        accident=accident, clean_title=clean_title
    )

    try:
        X_input    = build_input_df(inputs, encoders)
        pred_price = float(model.predict(X_input)[0])
        pred_price = max(0.0, pred_price)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.success("### ✅ Tahmin Tamamlandı!")
            st.metric("Tahmini Satış Fiyatı", f"${pred_price:,.0f}")
            low, high = pred_price * 0.90, pred_price * 1.10
            st.caption(f"📉 Tahmini aralık: **${low:,.0f}** – **${high:,.0f}** (±10%)")
            st.plotly_chart(make_gauge(pred_price), use_container_width=True)

        with col2:
            st.info("### 📋 Girilen Araç Özellikleri")
            import re
            hp_m  = re.search(r'(\d+\.?\d*)HP', engine_str, re.I)
            es_m  = re.search(r'(\d+\.?\d*)L',  engine_str, re.I)
            cy_m  = re.search(r'(\d+)\s*Cylinder', engine_str, re.I)
            car_age = 2026 - model_year

            display = pd.DataFrame({
                'Özellik': [
                    'Marka', 'Model', 'Model Yılı', 'Araç Yaşı',
                    'Kilometre', 'Yıllık Km', 'Yakıt Tipi', 'Şanzıman',
                    'Beygir Gücü', 'Motor Hacmi', 'Silindir',
                    'Dış Renk', 'İç Renk', 'Kaza', 'Temiz Tapu',
                    'Lüks Marka', 'Otomatik Vites'
                ],
                'Değer': [
                    brand, model_name, model_year, f"{car_age} yıl",
                    f"{milage:,} mi",
                    f"{milage / max(car_age, 1):,.0f} mi/yıl",
                    fuel_type or "(engine'den)",
                    transmission,
                    f"{float(hp_m.group(1)):.0f} HP" if hp_m else encoders['hp_median'],
                    f"{float(es_m.group(1)):.1f} L"  if es_m else encoders['engine_size_median'],
                    int(float(cy_m.group(1)))         if cy_m else encoders['cylinders_median'],
                    ext_col, int_col, accident, clean_title,
                    "✅" if brand in encoders['luxury_brands'] else "❌",
                    "✅" if any(k in transmission for k in ['A/T','Automatic','CVT']) else "❌"
                ]
            })
            st.dataframe(display, hide_index=True, use_container_width=True)

        # ── Feature Importance ─────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Top 15 Feature Importance")

        fi = pd.Series(
            model.feature_importances_, index=feature_columns
        ).sort_values(ascending=False).head(15).sort_values(ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index,
            orientation='h',
            marker_color='#1976D2'
        ))
        fig_fi.update_layout(
            xaxis_title="Importance", height=420,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Hata: {e}")

else:
    # ── Karşılama ekranı ──────────────────────────────────────────────────────
    st.info("👈 Sol panelden araç bilgilerini doldurun ve **Fiyat Tahmin Et** butonuna tıklayın.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     "LightGBM")
    c2.metric("Yarışma",   "Kaggle S4E9")
    c3.metric("Train Rows","188,533")
    c4.metric("Hedef",     "price (USD)")

    st.subheader("🔬 Feature Engineering Özeti")
    fe_df = pd.DataFrame({
        'Özellik'  : [
            'horsepower', 'engine_size', 'cylinders',
            'car_age', 'milage_per_year', 'has_accident',
            'is_clean_title', 'is_automatic', 'is_luxury', 'hp_per_cylinder'
        ],
        'Kaynak / Açıklama': [
            'engine regex → HP değeri',
            'engine regex → "xL" motor hacmi',
            'engine regex → silindir sayısı',
            '2026 - model_year',
            'milage / car_age',
            '"At least 1 accident..." → 1',
            '"Yes" → 1',
            'A/T | Automatic | CVT → 1',
            'Lüks marka listesi → 1',
            'horsepower / cylinders'
        ],
        'Eksik Strateji': [
            'brand+model → brand → global median',
            'brand+model → brand → global median',
            'brand+model → brand → global median',
            '–', '–', '–', '–', '–', '–',
            'global median'
        ]
    })
    st.dataframe(fe_df, hide_index=True, use_container_width=True)

    st.subheader("📊 Encoding")
    st.markdown("""
    | Sütun Tipi | Yöntem |
    |---|---|
    | Kategorik (`brand`, `model`, `fuel_type`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`) | `pd.get_dummies(drop_first=True)` |
    | Numerik | Doğrudan kullanım |
    """)
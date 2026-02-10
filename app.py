# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# =========================
# WAJIB: CLASS KUSTOM (HARUS ADA SEBELUM joblib.load)
# =========================
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.quantile(X, self.lower_q, axis=0)
        self.up_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.up_)


# =========================
# KONFIGURASI FILE
# =========================
DATA_PATH = "dataprediksi.xlsx"
MODEL_PATH = "model_svr_pipeline_outlier_lag.pkl"
PRED_PATH = "Prediksi_P0_2025_2030_SVR_Outlier_Lag.xlsx"

FEATURES = ["RLS", "TPT", "PPK", "AML", "UHH", "TPAK", "AMH", "P0_lag1"]
BASE_FEATURES = ["RLS", "TPT", "PPK", "AML", "UHH", "TPAK", "AMH"]
FUTURE_YEARS = [2025, 2026, 2027, 2028, 2029, 2030]


def inject_css():
    st.markdown(
        """
        <style>
        /* ===== Hilangkan header putih ===== */
        header[data-testid="stHeader"] { display: none; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stToolbar"] { display: none; }

        /* ===== Background ===== */
        .stApp { background: linear-gradient(90deg, #ffd9a8 0%, #ffb0e6 100%) !important; }
        .block-container { padding-top: 0rem !important; }

        /* ===== Card ===== */
        .card {
            background: rgba(255,255,255,0.55);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.55);
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.55);
            border-right: 1px solid rgba(255,255,255,0.65);
        }

        /* ===== Rata tengah khusus card metric ===== */
        .metric-card { text-align: center; }
        .metric-title { font-size: 12px; color: #444; }
        .metric-value { font-size: 22px; font-weight: 800; }

        /* ===== Judul rata tengah ===== */
        .title-center { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.sort_values(["Kabupaten/Kota", "Tahun"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def add_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Kabupaten/Kota", "Tahun"]).reset_index(drop=True)
    df["P0_lag1"] = df.groupby("Kabupaten/Kota")["P0"].shift(1)
    df = df.dropna(subset=["P0_lag1"]).copy()
    return df


def evaluate_on_2024(df_with_lag: pd.DataFrame, model) -> dict:
    test = df_with_lag[df_with_lag["Tahun"] == 2024].copy()
    if test.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    X_test = test[FEATURES]
    y_test = test["P0"].astype(float)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


@st.cache_data(show_spinner=False)
def compute_permutation_importance_per_kab(df_with_lag: pd.DataFrame, kab: str, _model) -> pd.DataFrame:
    train = df_with_lag[(df_with_lag["Kabupaten/Kota"] == kab) & (df_with_lag["Tahun"] < 2024)].copy()

    # Jika data kab/kota terlalu sedikit, fallback ke global agar tidak error/noisy parah
    if len(train) < 6:
        train = df_with_lag[df_with_lag["Tahun"] < 2024].copy()

    X_train = train[FEATURES]
    y_train = train["P0"].astype(float)

    result = permutation_importance(
        _model,
        X_train,
        y_train,
        n_repeats=15,
        random_state=42,
        scoring="r2",
    )

    imp = (
        pd.DataFrame({"Variabel": FEATURES, "Importance": result.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def build_future_predictions_if_missing(df_raw: pd.DataFrame, model) -> pd.DataFrame:
    pred_list = []
    for daerah in df_raw["Kabupaten/Kota"].unique():
        subset = df_raw[df_raw["Kabupaten/Kota"] == daerah].sort_values("Tahun").copy()
        growth = subset[BASE_FEATURES].diff().mean()

        last_row = subset.iloc[-1]
        last_feat = last_row[BASE_FEATURES].copy()
        last_p0 = float(last_row["P0"])

        for i, year in enumerate(FUTURE_YEARS, start=1):
            proj_feat = last_feat + growth * i

            row = proj_feat.to_dict()
            row["P0_lag1"] = last_p0
            row["Kabupaten/Kota"] = daerah
            row["Tahun"] = year

            X_row = pd.DataFrame([row])[FEATURES]
            p0_pred = float(model.predict(X_row)[0])

            row["Prediksi_P0"] = p0_pred
            pred_list.append(row)
            last_p0 = p0_pred

    return pd.DataFrame(pred_list)


@st.cache_data(show_spinner=False)
def load_or_build_future(df_raw: pd.DataFrame, _model, pred_path: str) -> pd.DataFrame:
    if os.path.exists(pred_path):
        return pd.read_excel(pred_path)
    return build_future_predictions_if_missing(df_raw, _model)


# =========================
# APP
# =========================
st.set_page_config(
    page_title="Dashboard Prediksi Kemiskinan Sumatera Barat",
    layout="wide",
    initial_sidebar_state="expanded",  # penting: sidebar default terbuka di Streamlit Cloud
)
inject_css()

if not os.path.exists(DATA_PATH):
    st.error(f"File data tidak ditemukan: {DATA_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
    st.stop()

df_raw = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

df_lag = add_lag_feature(df_raw)
metrics = evaluate_on_2024(df_lag, model)
future_df = load_or_build_future(df_raw, model, PRED_PATH)

with st.sidebar:
    st.markdown("## ℹ️ Informasi")
    st.write("- Prediksi P0 2025–2030\n- Tren per kabupaten/kota\n- Variabel yang paling mempengaruhi")

    st.markdown("---")
    st.markdown("## 📊 Akurasi Model")
    st.caption("SVR Regression")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='card metric-card'>"
            f"<div class='metric-title'>MAE</div>"
            f"<div class='metric-value'>{metrics['MAE']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card metric-card'>"
            f"<div class='metric-title'>RMSE</div>"
            f"<div class='metric-value'>{metrics['RMSE']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card metric-card'>"
        f"<div class='metric-title'>R²</div>"
        f"<div class='metric-value'>{metrics['R2']:.4f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='card title-center'>"
    "<div style='font-size:26px;font-weight:900;color:#6a00ff;'>"
    "Dashboard Prediksi Kemiskinan Provinsi Sumatera Barat</div></div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

kab_list = sorted(df_raw["Kabupaten/Kota"].unique().tolist())
selected_kab = st.selectbox("Pilih Kabupaten/Kota", kab_list, index=0)

# =========================
# HITUNG IMPORTANCE SESUAI KAB/KOTA TERPILIH
# =========================
importance_df = compute_permutation_importance_per_kab(df_lag, selected_kab, model)
importance_wo_lag = importance_df[importance_df["Variabel"] != "P0_lag1"].copy()

# =========================
# DATA PREDIKSI & TREND
# =========================
sub_future = future_df[future_df["Kabupaten/Kota"] == selected_kab].sort_values("Tahun").copy()
sub_future["Prediksi_P0"] = sub_future["Prediksi_P0"].astype(float)

sub_hist = df_raw[df_raw["Kabupaten/Kota"] == selected_kab].sort_values("Tahun").copy()
sub_hist["Jenis"] = "Aktual"

sub_future_plot = sub_future[["Tahun", "Prediksi_P0"]].rename(columns={"Prediksi_P0": "P0"})
sub_future_plot["Jenis"] = "Prediksi"

trend_df = pd.concat([sub_hist[["Tahun", "P0", "Jenis"]], sub_future_plot], axis=0).sort_values("Tahun")

left, right = st.columns([1.25, 1])

with left:
    st.markdown(
        f"<div class='card' style='text-align:center;'><b>Tren Perubahan P₀ - {selected_kab}</b></div>",
        unsafe_allow_html=True,
    )
    fig_trend = px.line(trend_df, x="Tahun", y="P0", color="Jenis", markers=True)
    fig_trend.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig_trend, use_container_width=True)

with right:
    st.markdown(
        "<div class='card' style='text-align:center;'><b>Variabel yang Berpengaruh (Permutation Importance)</b></div>",
        unsafe_allow_html=True,
    )

    hide_lag = st.checkbox("Sembunyikan P0_lag1 (agar variabel lain terlihat)", value=True)
    plot_df = importance_wo_lag if hide_lag else importance_df

    fig_imp = px.bar(plot_df, x="Variabel", y="Importance")
    fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

    if hide_lag:
        st.caption("Catatan: P0_lag1 biasanya dominan karena kemiskinan berubah perlahan dari tahun ke tahun.")

st.markdown(
    "<div class='card' style='text-align:center;'><b>Prediksi P₀ Tahun 2025 - 2030</b></div>",
    unsafe_allow_html=True,
)

table_df = sub_future[["Tahun", "Prediksi_P0"]].copy()
table_df["Prediksi_P0"] = table_df["Prediksi_P0"].round(4)

st.markdown(
    f"""
    <div class="card">
        <table style="width:100%; border-collapse:collapse; text-align:center;">
            <thead>
                <tr>
                    <th style="padding:8px; border-bottom:1px solid #ddd;">Tahun</th>
                    <th style="padding:8px; border-bottom:1px solid #ddd;">Prediksi P₀</th>
                </tr>
            </thead>
            <tbody>
                {''.join([
                    f"<tr><td style='padding:8px;'>{int(r.Tahun)}</td>"
                    f"<td style='padding:8px;'>{r.Prediksi_P0:.4f}</td></tr>"
                    for r in table_df.itertuples()
                ])}
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

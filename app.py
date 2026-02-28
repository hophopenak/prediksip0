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
class WinsorizerIQR(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Q1 = np.quantile(X, 0.25, axis=0)
        Q3 = np.quantile(X, 0.75, axis=0)
        IQR = Q3 - Q1
        self.low_ = Q1 - self.factor * IQR
        self.up_  = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.up_)


# =========================
# KONFIGURASI FILE
# =========================
DATA_PATH  = "dataprediksi.xlsx"
MODEL_PATH = "model_svr.pkl"
PRED_PATH  = "Prediksi_Tingkat_Kemiskinan.xlsx"

FEATURES = ["RLS", "TPT", "PPK", "AML", "UHH", "TPAK", "AMH", "P0_lag1"]
BASE_FEATURES = ["RLS", "TPT", "PPK", "AML", "UHH", "TPAK", "AMH"]
FUTURE_YEARS = [2025, 2026, 2027, 2028, 2029, 2030]


def inject_css():
    st.markdown(
        """
        <style>
        /* =========================
           HIDE DEFAULT STREAMLIT UI
        ========================= */
        header[data-testid="stHeader"] { display: none; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stToolbar"] { display: none; }

        /* =========================
           PENTING:
           Jangan sembunyikan tombol panah collapse sidebar
           (jadi baris ini DIHAPUS dari versi sebelumnya)
           - button[data-testid="stSidebarCollapseButton"] ...
           - div[data-testid="collapsedControl"] ...
        ========================= */

        /* =========================
           PINK SOFT + HIGH READABILITY
        ========================= */
        :root{
            --bg1:#fff1f7;
            --bg2:#ffe0f0;
            --bg3:#f7efff;

            --text:#111827;
            --text2:#1f2937;
            --muted:#374151;

            --card: rgba(255,255,255,0.96);
            --border: rgba(236, 149, 196, 0.35);
            --shadow: 0 10px 22px rgba(0,0,0,0.08);

            --sidebar: rgba(255, 235, 246, 0.92);
        }

        .stApp{
            background: radial-gradient(circle at 20% 10%, var(--bg1) 0%, var(--bg2) 50%, var(--bg3) 100%) !important;
            color: var(--text) !important;
        }

        .block-container{ padding-top: 0rem !important; }

        /* =========================
           GLOBAL FONT
        ========================= */
        html, body, [class*="css"]{
            color: var(--text) !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }

        h1, h2, h3, h4{
            color: var(--text) !important;
            font-weight: 900 !important;
        }

        .title-center{ text-align: center; }

        /* =========================
           CARD
        ========================= */
        .card{
            background: var(--card);
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            color: var(--text) !important;
        }

        .card b, .card strong{
            color: var(--text) !important;
            font-weight: 900 !important;
            font-size: 16px !important;
        }

        /* =========================
           SIDEBAR
        ========================= */
        section[data-testid="stSidebar"]{
            background: var(--sidebar) !important;
            border-right: 1px solid rgba(236, 149, 196, 0.35);
        }
        section[data-testid="stSidebar"] *{
            color: var(--text) !important;
            font-size: 15px !important;
            font-weight: 700 !important;
        }

        label{
            color: var(--text) !important;
            font-weight: 800 !important;
            font-size: 15px !important;
        }

        div[data-testid="stSelectbox"] > div{
            background: rgba(255,255,255,0.98) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(236, 149, 196, 0.45) !important;
        }

        /* =========================
           METRIC CARD
        ========================= */
        .metric-card{ text-align: center; }
        .metric-title{
            font-size: 13px !important;
            color: var(--text2) !important;
            font-weight: 800 !important;
            letter-spacing: .2px;
        }
        .metric-value{
            font-size: 26px !important;
            font-weight: 900 !important;
            color: var(--text) !important;
        }

        div[data-testid="stCheckbox"]{
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(236, 149, 196, 0.35);
            border-radius: 12px;
            padding: 8px 12px;
        }

        .stCaption, .stMarkdown p{
            color: var(--muted) !important;
            font-size: 14px !important;
            font-weight: 650 !important;
        }

        table{
            font-size: 15px !important;
            color: var(--text) !important;
        }
        th{
            font-size: 15px !important;
            font-weight: 900 !important;
            color: var(--text) !important;
            border-bottom: 2px solid rgba(0,0,0,0.18) !important;
        }
        td{
            font-size: 15px !important;
            font-weight: 700 !important;
            color: var(--text) !important;
        }

        /* PLOTLY FONT */
        .js-plotly-plot text{
            font-size: 14px !important;
            font-weight: 700 !important;
            fill: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
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


def evaluate_model_time_based(df_with_lag: pd.DataFrame, model) -> dict:
    df_with_lag = df_with_lag.sort_values(["Kabupaten/Kota", "Tahun"]).reset_index(drop=True)

    test = df_with_lag[df_with_lag["Tahun"].between(2023, 2024)].copy()
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
    if len(train) < 6:
        train = df_with_lag[df_with_lag["Tahun"] < 2024].copy()

    X_train = train[FEATURES]
    y_train = train["P0"].astype(float)

    result = permutation_importance(
        _model, X_train, y_train,
        n_repeats=15, random_state=42, scoring="r2"
    )

    return (
        pd.DataFrame({"Variabel": FEATURES, "Importance": result.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


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
# PREDIKSI MANUAL
# =========================
def get_last_p0_for_kab(df_raw: pd.DataFrame, kab: str) -> float:
    sub = df_raw[df_raw["Kabupaten/Kota"] == kab].sort_values("Tahun")
    if sub.empty:
        return np.nan
    return float(sub.iloc[-1]["P0"])


def predict_manual_input(model,
                         rls: float, tpt: float, ppk: float, aml: float,
                         uhh: float, tpak: float, amh: float,
                         p0_lag1: float) -> float:
    row = {
        "RLS": rls,
        "TPT": tpt,
        "PPK": ppk,
        "AML": aml,
        "UHH": uhh,
        "TPAK": tpak,
        "AMH": amh,
        "P0_lag1": p0_lag1,
    }
    X_row = pd.DataFrame([row])[FEATURES]
    return float(model.predict(X_row)[0])


# =========================
# APP
# =========================
st.set_page_config(
    page_title="Dashboard Prediksi Tingkat Kemiskinan Sumatera Barat",
    layout="wide",
    initial_sidebar_state="expanded"  # sidebar default terbuka (panah tetap ada)
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
metrics = evaluate_model_time_based(df_lag, model)
future_df = load_or_build_future(df_raw, model, PRED_PATH)

if "manual_pred" not in st.session_state:
    st.session_state.manual_pred = None

with st.sidebar:
    st.markdown("""
### 📌 Informasi

- 📅 **Prediksi Tingkat Kemiskinan Tahun 2025–2030**
- 📈 **Tren per Kabupaten/Kota**
- 🧠 **Variabel yang Paling Mempengaruhi**
""")
    st.markdown("---")
    st.markdown("## 📊 Akurasi Model")
    st.caption("🔎 SVR Regression")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='card metric-card'>"
            f"<div class='metric-title'>MAE</div>"
            f"<div class='metric-value'>{metrics['MAE']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='card metric-card'>"
            f"<div class='metric-title'>RMSE</div>"
            f"<div class='metric-value'>{metrics['RMSE']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card metric-card'>"
        f"<div class='metric-title'>R²</div>"
        f"<div class='metric-value'>{metrics['R2']:.4f}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<div class='card title-center'>"
    "<div style='font-size:26px;font-weight:900;color:#6a00ff;'>"
    "📊 Dashboard Prediksi Tingkat Kemiskinan Provinsi Sumatera Barat</div></div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

kab_list = sorted(df_raw["Kabupaten/Kota"].unique().tolist())
selected_kab = st.selectbox("Pilih Kabupaten/Kota", kab_list, index=0)

# =========================
# PREDIKSI MANUAL
# =========================
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='card' style='text-align:center;'><b>✍️ Prediksi Manual (Input Nilai Sendiri)</b></div>",
    unsafe_allow_html=True
)

kab_hist = df_raw[df_raw["Kabupaten/Kota"] == selected_kab].sort_values("Tahun").copy()
base_df = kab_hist if not kab_hist.empty else df_raw.copy()

def default_val(col):
    return float(base_df[col].median())

colA, colB = st.columns([1, 1])
with colA:
    manual_year = st.number_input("Tahun Prediksi (Manual)", min_value=2025, max_value=2030, value=2025, step=1)
with colB:
    use_auto_lag = st.checkbox("Gunakan P0_lag1 otomatis (P0 terakhir kab/kota)", value=True)

auto_p0_lag1 = get_last_p0_for_kab(df_raw, selected_kab)

c1, c2, c3, c4 = st.columns(4)
with c1:
    rls = st.number_input("RLS (manual)", value=default_val("RLS"), step=0.01, format="%.4f")
    tpt = st.number_input("TPT (manual)", value=default_val("TPT"), step=0.01, format="%.4f")
with c2:
    ppk = st.number_input("PPK (manual)", value=float(default_val("PPK")), step=1.0, format="%.0f")
    aml = st.number_input("AML (manual)", value=default_val("AML"), step=0.01, format="%.4f")
with c3:
    uhh = st.number_input("UHH (manual)", value=default_val("UHH"), step=0.01, format="%.4f")
    tpak = st.number_input("TPAK (manual)", value=default_val("TPAK"), step=0.01, format="%.4f")
with c4:
    amh = st.number_input("AMH (manual)", value=default_val("AMH"), step=0.01, format="%.4f")

if use_auto_lag:
    p0_lag1 = auto_p0_lag1
    st.caption(f"P0_lag1 otomatis (P0 terakhir {selected_kab}) = {p0_lag1:.4f}")
else:
    p0_lag1 = st.number_input("P0_lag1 (manual)", value=float(auto_p0_lag1), step=0.01, format="%.4f")

btn_manual = st.button("🔮 Prediksi P0 Manual", use_container_width=True)

if btn_manual:
    if np.isnan(p0_lag1):
        st.error("P0_lag1 tidak tersedia. Pastikan data historis untuk kab/kota ini ada.")
    else:
        pred_p0_manual = predict_manual_input(
            model=model,
            rls=float(rls), tpt=float(tpt), ppk=float(ppk), aml=float(aml),
            uhh=float(uhh), tpak=float(tpak), amh=float(amh),
            p0_lag1=float(p0_lag1)
        )

        st.session_state.manual_pred = {"Tahun": int(manual_year), "P0": float(pred_p0_manual)}

        st.markdown(
            f"<div class='card metric-card'>"
            f"<div class='metric-title'>Hasil Prediksi P0 (Manual)</div>"
            f"<div class='metric-value'>{pred_p0_manual:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='card'><b>Ringkasan Input Manual</b></div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Kabupaten/Kota": selected_kab,
            "Tahun": int(manual_year),
            "RLS": rls, "TPT": tpt, "PPK": ppk, "AML": aml,
            "UHH": uhh, "TPAK": tpak, "AMH": amh,
            "P0_lag1": p0_lag1,
            "Prediksi_P0": pred_p0_manual
        }]), use_container_width=True, hide_index=True)

# =========================
# IMPORTANCE
# =========================
importance_df = compute_permutation_importance_per_kab(df_lag, selected_kab, model)
importance_wo_lag = importance_df[importance_df["Variabel"] != "P0_lag1"].copy()

# =========================
# TREND DATA
# =========================
sub_future = future_df[future_df["Kabupaten/Kota"] == selected_kab].sort_values("Tahun").copy()
sub_future["Prediksi_P0"] = sub_future["Prediksi_P0"].astype(float)

sub_hist = df_raw[df_raw["Kabupaten/Kota"] == selected_kab].sort_values("Tahun").copy()
sub_hist["Jenis"] = "Aktual"

sub_future_plot = sub_future[["Tahun", "Prediksi_P0"]].rename(columns={"Prediksi_P0": "P0"})
sub_future_plot["Jenis"] = "Prediksi"

trend_df = pd.concat([sub_hist[["Tahun", "P0", "Jenis"]], sub_future_plot], axis=0).sort_values("Tahun")

if st.session_state.manual_pred is not None:
    mp = st.session_state.manual_pred
    trend_df = pd.concat(
        [trend_df, pd.DataFrame([{"Tahun": int(mp["Tahun"]), "P0": float(mp["P0"]), "Jenis": "Manual"}])],
        ignore_index=True
    ).sort_values("Tahun")

left, right = st.columns([1.25, 1])

with left:
    st.markdown(
        f"<div class='card' style='text-align:center;'><b>📈 Tren Perubahan Tingkat Kemiskinan - {selected_kab}</b></div>",
        unsafe_allow_html=True
    )
    fig_trend = px.line(trend_df, x="Tahun", y="P0", color="Jenis", markers=True)
    fig_trend.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig_trend, use_container_width=True)

with right:
    st.markdown(
        "<div class='card' style='text-align:center;'><b>🧠 Variabel yang Berpengaruh (Permutation Importance)</b></div>",
        unsafe_allow_html=True
    )
    hide_lag = st.checkbox("Sembunyikan P0_lag1 (agar variabel lain terlihat)", value=True)
    plot_df = importance_wo_lag if hide_lag else importance_df

    fig_imp = px.bar(plot_df, x="Variabel", y="Importance")
    fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

    if hide_lag:
        st.caption("Catatan: P0_lag1 biasanya dominan karena kemiskinan berubah perlahan dari tahun ke tahun.")

# =========================
# TABEL PREDIKSI KAB
# =========================
st.markdown(
    "<div class='card' style='text-align:center;'><b>📅 Prediksi Tingkat Kemiskinan Tahun 2025 - 2030</b></div>",
    unsafe_allow_html=True
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
    unsafe_allow_html=True
)

# =========================
# PREDIKSI PROVINSI
# =========================
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

st.markdown(
    "<div class='card' style='text-align:center;'><b>🏛️ Prediksi Tingkat Kemiskinan Provinsi Sumatera Barat (2025 - 2030)</b></div>",
    unsafe_allow_html=True
)

prov_df = (
    future_df.groupby("Tahun", as_index=False)["Prediksi_P0"]
            .mean()
            .rename(columns={"Prediksi_P0": "Prediksi_Tingkat_Kemiskinan_Prov"})
)

prov_df["Tahun"] = prov_df["Tahun"].astype(int)
prov_df["Prediksi_Tingkat_Kemiskinan_Prov"] = prov_df["Prediksi_Tingkat_Kemiskinan_Prov"].astype(float)

fig_prov = px.line(prov_df, x="Tahun", y="Prediksi_Tingkat_Kemiskinan_Prov", markers=True)
fig_prov.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=55, b=10),
    xaxis_title="Tahun",
    yaxis_title="Prediksi Tingkat Kemiskinan Provinsi (%)",
    title=dict(
        text="📈 Tren Prediksi Tingkat Kemiskinan Provinsi Sumatera Barat 2025–2030 (Rata-rata Kab/Kota)",
        x=0.5
    )
)
st.plotly_chart(fig_prov, use_container_width=True)

prov_table = prov_df.copy()
prov_table["Prediksi_Tingkat_Kemiskinan_Prov"] = prov_table["Prediksi_Tingkat_Kemiskinan_Prov"].round(4)

st.markdown(
    "<div class='card' style='text-align:center;'><b>📋 Tabel Prediksi Tingkat Kemiskinan Provinsi Sumatera Barat</b></div>",
    unsafe_allow_html=True
)
st.dataframe(prov_table, use_container_width=True, hide_index=True)

st.caption("Catatan: Nilai provinsi dihitung dari rata-rata sederhana prediksi seluruh kabupaten/kota pada setiap tahun.")

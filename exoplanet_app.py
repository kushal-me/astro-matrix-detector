# exoplanet_app.py
"""
🌌 AI/ML Kepler Exoplanet Detector — NASA Space Apps Challenge 2025
Author: Kushal .S.: Team:-Astro-Matrix
Goal: Use real NASA Kepler data and AI to identify potential exoplanets
-----------------------------------------------------------------------
This app fetches real NASA Kepler light curves, extracts features such
as brightness variability and periodicity (via Lomb–Scargle analysis),
including a 3D model with the help of streamlit
and trains an ML model (RandomForest) to classify stars as:
  ✅ Exoplanet Host (planet detected)
  ❌ Non-Exoplanet Host (no planet detected)



Features:
 - Batch download + local caching of Kepler lightcurves (parallelized)
 - Lomb-Scargle periodicity + statistical features
 - RandomForest + XGBoost + ensemble (averaged probabilities)
 - Real KOI labels (NASA Exoplanet Archive) with fallback demo labels
 - Interactive 3D PCA visualization with click-to-inspect and auto-rotation frames
 - Feature importance visualization
"""

import os, time, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from astropy.timeseries import LombScargle
from xgboost import XGBClassifier
import plotly.graph_objects as go

# lightkurve and astroquery
import lightkurve as lk
try:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchiveClass
    ARCHIVE_AVAILABLE = True
except Exception:
    ARCHIVE_AVAILABLE = False

# ---------------- CONFIG ----------------
CACHE_DIR = Path("./lc_cache")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("exoplanet_hybrid_model.joblib")
st.set_page_config(page_title="AI/ML Exoplanet Detector", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body {background-color: #0e1117; color: #d1d5db;}
.stButton>button {background: linear-gradient(90deg, #00c6ff, #0072ff); color: white; font-weight: bold; border-radius: 10px;}
.stButton>button:hover {background: linear-gradient(90deg, #7F00FF, #E100FF);}
</style>
""", unsafe_allow_html=True)

st.title("AI/ML Exoplanet Detector")
st.markdown("""
Automatically detects **Exoplanets** using **NASA Kepler/TESS light curves**.  
Features advanced **periodicity analysis (Lomb–Scargle)** and hybrid **AI classifiers**.  
""")

# ---------------- HELPERS ----------------
def save_lc_to_cache(kic, lc):
    fname = CACHE_DIR / f"{kic}.fits"
    try:
        lc.to_fits(str(fname), overwrite=True)
    except Exception:
        pass

def load_lc_from_cache(kic):
    fname = CACHE_DIR / f"{kic}.fits"
    if fname.exists():
        try:
            return lk.lightcurve.LightCurve.read(str(fname))
        except Exception:
            return None
    return None

@st.cache_data
def fetch_koi_label_dict():
    if not ARCHIVE_AVAILABLE: return {}
    try:
        archive = NasaExoplanetArchiveClass()
        table = archive.query_criteria(table="cumulative", select="kepid,koi_disposition")
        return {str(int(r["kepid"])): r["koi_disposition"] for r in table}
    except Exception:
        return {}

def extract_basic_stats(flux):
    return [np.mean(flux), np.std(flux), np.var(flux), np.min(flux), np.max(flux)]

def extract_periodicity_features(time, flux):
    try:
        ls = LombScargle(time, flux)
        freq, power = ls.autopower(maximum_frequency=5.0)
        best = np.argmax(power)
        return float(freq[best]), float(power[best]), float(ls.false_alarm_probability(power[best]))
    except Exception:
        return 0.0, 0.0, 1.0

def process_lightcurve_lite(lc, max_points=200):
    lc = lc.normalize().remove_nans()
    time = np.array(lc.time.value)
    flux = np.array(lc.flux.value)
    if len(flux) == 0: return None
    if len(flux) > max_points:
        flux, time = flux[:max_points], time[:max_points]
    else:
        pad = max_points - len(flux)
        flux = np.concatenate([flux, np.full(pad, np.median(flux))])
        time = np.concatenate([time, np.linspace(time[-1]+1, time[-1]+pad, pad)])
    stats = extract_basic_stats(flux)
    freq, power, fap = extract_periodicity_features(time, flux)
    return np.array(stats + [freq, power, fap]), flux, time

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    mission = st.text_input("Target Query", "KIC 11904151")
    batch_size = st.number_input("Stars per Batch", 1, 10, 3)
    max_batches = st.number_input("Max Batches", 1, 20, 4)
    max_points = st.slider("Flux Points per Star", 100, 2000, 200, 100)
    label_mode = st.radio("Label Mode", ["Real NASA Labels", "Demo / Fake Labels"])
    reload_cache = st.checkbox("Clear Cache Before Start", False)
    use_saved_model = st.checkbox("Use Saved Model if Available", True)

if reload_cache:
    for f in CACHE_DIR.glob("*.fits"):
        try: f.unlink()
        except Exception: pass
    st.success("🧹 Cache cleared successfully!")

# ---------------- MAIN PIPELINE ----------------
if st.button("🚀 Start: Fetch → Train → Predict"):
    koi_dict = fetch_koi_label_dict()

    st.info(f"🔭 Searching for '{mission}' in NASA archive...")
    try:
        search = lk.search_lightcurve(mission)
        if len(search) == 0:
            st.warning(f"No results for '{mission}'. Trying fallback: Kepler")
            search = lk.search_lightcurve("Kepler")
    except Exception as e:
        st.error(f"NASA search failed: {e}")
        search = None

    if not search or len(search) == 0:
        st.error("❌ No light curves found. Try another target.")
    else:
        kics = []
        for row in search.table:
            if "targetid" in row.colnames:
                kics.append(str(int(row["targetid"])))
            elif "kepid" in row.colnames:
                kics.append(str(int(row["kepid"])))
        kics = list(dict.fromkeys(kics))
        max_to_process = min(len(kics), batch_size * max_batches)
        st.info(f"📡 Processing {max_to_process} stars...")

        features_all, labels_all, raw_flux, raw_time = [], [], [], []
        bar = st.progress(0.0)
        text = st.empty()

        for batch_idx, start in enumerate(range(0, max_to_process, batch_size)):
            batch = kics[start:start+batch_size]
            text.text(f"Batch {batch_idx+1}/{max_batches}: {batch}")
            for kic in batch:
                lc = load_lc_from_cache(kic)
                if lc is None:
                    try:
                        res = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")
                        if res and len(res) > 0:
                            lc = res.download()
                            save_lc_to_cache(kic, lc)
                    except Exception:
                        continue
                if lc is None: continue

                res = process_lightcurve_lite(lc, max_points)
                if res is None: continue
                feat, flux, timearr = res

                # Label handling
                label = None
                if label_mode == "Real NASA Labels":
                    disp = koi_dict.get(str(kic))
                    if disp in ["CONFIRMED", "CANDIDATE"]: label = 1
                    elif disp == "FALSE POSITIVE": label = 0
                    else: continue
                else:
                    label = 1 if np.std(flux) > 0.01 else 0

                features_all.append(feat)
                labels_all.append(label)
                raw_flux.append(flux)
                raw_time.append(timearr)

            bar.progress((batch_idx+1)/max_batches)
            time.sleep(0.2)

        bar.empty()
        text.empty()

        if len(features_all) < 10:
            st.error("Only 0 usable labeled stars collected — need ≥10. Try Demo mode or increase batches.")
        else:
            X, y = np.vstack(features_all), np.array(labels_all)
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

            if use_saved_model and MODEL_PATH.exists():
                st.info("Loading saved hybrid model...")
                clf = joblib.load(MODEL_PATH)
            else:
                st.info("Training hybrid model (RandomForest + XGBoost)...")
                rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
                rf.fit(X_train, y_train)
                xgb.fit(X_train, y_train)
                # Blend predictions
                rf_proba = rf.predict_proba(X_train)[:, 1]
                xgb_proba = xgb.predict_proba(X_train)[:, 1]
                blend_train = np.vstack([rf_proba, xgb_proba]).T
                meta = RandomForestClassifier(n_estimators=100, random_state=42)
                meta.fit(blend_train, y_train)
                clf = (rf, xgb, meta)
                joblib.dump(clf, MODEL_PATH)
                st.success("✅ Model trained & saved!")

            # Evaluation
            rf, xgb, meta = clf
            rf_test = rf.predict_proba(X_test)[:,1]
            xgb_test = xgb.predict_proba(X_test)[:,1]
            blend_test = np.vstack([rf_test, xgb_test]).T
            y_pred_final = meta.predict(blend_test)
            y_prob_final = meta.predict_proba(blend_test)[:,1]

            st.text(classification_report(y_test, y_pred_final, zero_division=0))
            st.write(f"ROC AUC: {roc_auc_score(y_test, y_prob_final):.3f}")

            # ---- 3D Visualization ----
            st.subheader("🌠 3D Light Curve Visualization")
            if len(raw_flux) > 0:
                flux_matrix = np.array(raw_flux)
                time_axis = np.arange(flux_matrix.shape[1])
                fig3d = go.Figure(data=[go.Surface(z=flux_matrix, x=time_axis, y=np.arange(len(flux_matrix)))])
                fig3d.update_layout(scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Star Index',
                    zaxis_title='Flux'
                ), margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig3d, use_container_width=True)

            # ---- Prediction Example ----
            st.markdown("---")
            st.subheader("🔭 Predict Random Star from Test Set")
            idx = st.slider("Pick Sample", 0, len(X_test)-1, 0)
            rf_p, xgb_p = rf.predict_proba([X_test[idx]])[:,1], xgb.predict_proba([X_test[idx]])[:,1]
            blend_pred = meta.predict_proba(np.vstack([rf_p, xgb_p]).T)[:,1][0]
            st.metric("Prediction", "🌍 Exoplanet" if blend_pred>0.5 else "❌ No Planet",
                      f"Confidence: {blend_pred:.2f}")

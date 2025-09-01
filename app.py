import streamlit as st
import numpy as np
import pandas as pd
import mne
import pickle
import tempfile
import os
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# ---------------------- UI ----------------------
st.set_page_config(page_title="EEG Depression Detection", layout="centered")
st.title("🧠 Depression Detection from Raw EEG (EO/EC)")
st.caption("Pipeline: Filter → Notch → ICA → Avg Ref → 5s Epochs → Z-score → FFT band means → Top-20 features → Model")

# ---------------------- Constants (match training) ----------------------
SELECTED_CHANNELS = [
    'EEG Fp1-LE','EEG F3-LE','EEG C3-LE','EEG P3-LE','EEG O1-LE',
    'EEG F7-LE','EEG T3-LE','EEG T5-LE','EEG Fz-LE','EEG Fp2-LE',
    'EEG F4-LE','EEG C4-LE','EEG P4-LE','EEG O2-LE','EEG F8-LE',
    'EEG T4-LE','EEG T6-LE','EEG Cz-LE','EEG Pz-LE'
]

BANDS = {
    'delta': (0.1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta' : (13, 30),
    'gamma': (30, 100),
}

ALL_FEATURE_COLS = []
for ch in SELECTED_CHANNELS:
    ch_name = ch.replace("EEG ", "").replace("-LE", "").replace("-", "_")
    for band in BANDS:
        ALL_FEATURE_COLS.append(f"{ch_name}_{band}")

SFREQ = 256
SEGMENT_DURATION = 5  
AMP_THRESHOLD_UV = 100  

EO_FEATURES = [
    "C4_gamma","O2_delta","T6_beta","T4_alpha",
    "Cz_delta","T4_gamma","P4_gamma","C4_beta",
    "C4_delta","Cz_gamma","F8_theta","F7_delta",
    "P4_alpha","P4_beta","O2_gamma","Pz_theta",
    "P4_delta","T4_beta","Cz_alpha","T4_theta"
]

EC_FEATURES = [
    "P4_alpha","C4_gamma","T6_alpha","O2_beta",
    "Pz_beta","P3_alpha","T5_delta","C3_delta",
    "F8_theta","O2_gamma","O2_delta","Fp2_delta",
    "Fp1_gamma","T5_alpha","T6_beta","F8_gamma",
    "T6_delta","T4_alpha","Cz_theta","C4_beta"
]

# ---------------------- Load models ----------------------
@st.cache_resource
def load_models():
    with open("models/xgboost_model_EO.pkl", "rb") as f:
        eo_package = pickle.load(f)
    with open("models/adaboost_model_EC.pkl", "rb") as f:
        ec_package = pickle.load(f)

    return (
        eo_package['model'], eo_package['features'],
        ec_package['model'], ec_package['features']
    )

eo_model, eo_features, ec_model, ec_features = load_models()
eo_results = pickle.load(open("models/eo_results.pkl", "rb"))
ec_results = pickle.load(open("models/ec_results.pkl", "rb"))

# ---------------------- Preprocessing ----------------------
def preprocess_eeg(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    missing = [ch for ch in SELECTED_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}")

    raw.pick_channels(SELECTED_CHANNELS)
    raw.filter(0.5, 70, fir_design='firwin', verbose=False)
    raw.notch_filter(50, verbose=False)

    ica = mne.preprocessing.ICA(n_components=len(SELECTED_CHANNELS), random_state=42, max_iter='auto')
    ica.fit(raw)
    raw = ica.apply(raw)

    raw.set_eeg_reference('average', projection=False)
    return raw

def extract_fft_features(segment, sfreq):
    n_channels, n_times = segment.shape
    freqs = np.fft.fftfreq(n_times, d=1/sfreq)
    pos = freqs > 0
    fft_vals = np.abs(fft(segment))[:, pos]
    freqs = freqs[pos]

    band_means = []
    for _, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_means.append(fft_vals[:, mask].mean(axis=1))
    return np.concatenate(band_means)

def extract_features_from_raw(raw):
    data = raw.get_data()
    seg_len = int(SEGMENT_DURATION * SFREQ)
    n_ch, n_samp = data.shape
    n_segments = n_samp // seg_len

    feats = []
    for i in range(n_segments):
        seg = data[:, i*seg_len:(i+1)*seg_len]
        if np.any(np.abs(seg) > AMP_THRESHOLD_UV):
            continue
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
        feat_vec = extract_fft_features(seg, SFREQ)
        if feat_vec.shape[0] == len(ALL_FEATURE_COLS):
            feats.append(feat_vec)

    if len(feats) == 0:
        return pd.DataFrame(columns=ALL_FEATURE_COLS)
    return pd.DataFrame(feats, columns=ALL_FEATURE_COLS)

# ---------------------- Streamlit Tabs ----------------------
tab1, tab2 = st.tabs(["EEG Prediction", "Model Performance"])

# ---------------------- EEG Prediction Tab ----------------------
with tab1:
    condition = st.radio("Select EEG condition", ["Eye Open (EO)", "Eye Closed (EC)"])
    uploaded = st.file_uploader("Upload raw EEG file (.edf)", type=["edf"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Preprocessing & extracting features..."):
                raw = preprocess_eeg(tmp_path)
                df_feats = extract_features_from_raw(raw)

            if df_feats.empty:
                st.warning("No valid 5-second segments found after preprocessing.")
            else:
                if condition == "Eye Open (EO)":
                    model, use_cols = eo_model, EO_FEATURES
                else:
                    model, use_cols = ec_model, EC_FEATURES

                X = df_feats.reindex(columns=use_cols, fill_value=0)
                seg_pred = model.predict(X)

                try:
                    seg_proba = model.predict_proba(X)[:, 1]
                    avg_prob = float(np.mean(seg_proba))
                except Exception:
                    seg_proba, avg_prob = None, None

                depressed = int((seg_pred == 1).sum())
                healthy = int((seg_pred == 0).sum())
                final_pred = 1 if depressed > healthy else 0

                st.subheader("Result")
                if final_pred == 1:
                    st.error("Prediction: **MDD (Depressed)**")
                else:
                    st.success("Prediction: **Healthy**")

                st.caption(f"Majority vote → Depressed: {depressed}, Healthy: {healthy}")
                if avg_prob is not None:
                    st.info(f"Average MDD probability: **{avg_prob:.2f}**")

                try:
                    st.markdown("---")
                    st.subheader("Raw EEG (first 10 s, post-processing)")
                    st.line_chart(raw.to_data_frame().iloc[:int(raw.info['sfreq'] * 10)])
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Error during analysis: {e}")

        finally:
            try: os.unlink(tmp_path)
            except Exception: pass

# ---------------------- Model Performance Tab ----------------------
with tab2:
    st.subheader("Model Performance on Test Data")
    choice_perf = st.radio("Select Model", ["Eye Open (EO)", "Eye Closed (EC)"])

    if choice_perf == "Eye Open (EO)":
        results, title = eo_results, "XGBoost (EO)"
    else:
        results, title = ec_results, "AdaBoost (EC)"

    y_test, y_pred, y_score = results["y_test"], results["y_pred"], results["y_score"]

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Classification Report")
    st.dataframe(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {title}")
    ax.legend(loc="lower right")
    st.pyplot(fig)
# ---------------------- Notes ----------------------
with st.expander("Notes (important for reviewers)"):
    st.markdown("""
- Preprocessing = 0.5–70 Hz bandpass, 50 Hz notch, ICA (19 comps), avg ref, 5-s windows, per-channel z-score, FFT band means.
- Top-20 fixed features per condition (EO/EC) → trained models.
- Prediction = majority vote across segments + average probability.

**Dataset Used:**  
[EEG_Data_New (Figshare)](https://figshare.com/articles/dataset/EEG_Data_New/4244171)  
- **H** = Healthy Controls  
- **MDD** = Major Depressive Disorder  
- **EC** = Eyes Closed  
- **EO** = Eyes Open  
- **TASK** = P300 task data  
    """)


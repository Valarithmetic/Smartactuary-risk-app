import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Smart Actuary — Micro-Insurance Risk Calculator",
    page_icon="Σ",
    layout="centered",
)

# ---------- Smart Actuary brand styling ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #F4EFE6; }

    .sa-header {
        background: #10233A;
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 28px;
    }
    .sa-header h1 {
        color: #F4EFE6;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 26px;
        margin: 0 0 6px 0;
    }
    .sa-header .sigma { color: #D9A441; }
    .sa-header p {
        color: #AEB9C6;
        font-size: 14px;
        margin: 0;
    }

    .sa-result {
        background: white;
        border-radius: 12px;
        padding: 24px 28px;
        border-left: 5px solid #D9A441;
        margin-top: 20px;
    }
    .sa-result h3 { color: #10233A; font-family: 'Poppins', sans-serif; margin-top:0; }
    .sa-premium {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        color: #10233A;
        font-weight: 600;
    }
    .sa-band-low { color: #2f7a4f; font-weight: 700; }
    .sa-band-high { color: #b3541e; font-weight: 700; }

    .stButton>button {
        background-color: #D9A441;
        color: #412402;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover { background-color: #c4913a; color: #412402; }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sa-header">
    <h1>Smart<span class="sigma">Σ</span>Actuary — Micro-Insurance Risk Calculator</h1>
    <p>A live demo of the Random Forest risk classification model behind Smart Actuary's micro-insurance pricing case study.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter a policyholder profile below to see the model classify their risk band "
    "and generate a dynamic daily premium — the same approach used in the "
    "[micro-insurance case study](https://smartactuary.co.ke/#case-microinsurance)."
)

# ---------- Train the model on realistic synthetic data ----------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 800
    age = np.random.randint(18, 65, n)
    income = np.random.randint(3000, 60000, n)
    dependents = np.random.randint(0, 6, n)
    chronic_illness = np.random.binomial(1, 0.18, n)
    education_years = np.random.randint(0, 17, n)

    # Realistic underlying risk score -> label
    risk_score = (
        0.03 * age
        - 0.00006 * income
        + 0.25 * dependents
        + 1.8 * chronic_illness
        - 0.05 * education_years
        + np.random.normal(0, 0.6, n)
    )
    risk_band = (risk_score > np.median(risk_score)).astype(int)  # 1 = high risk

    X = pd.DataFrame({
        "age": age,
        "income": income,
        "dependents": dependents,
        "chronic_illness": chronic_illness,
        "education_years": education_years,
    })
    y = risk_band

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ---------- Input form ----------
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65, 30)
    income = st.number_input("Monthly income (KES)", min_value=1000, max_value=100000, value=15000, step=500)
    dependents = st.slider("Number of dependents", 0, 8, 1)
with col2:
    education_years = st.slider("Years of formal education", 0, 17, 8)
    chronic_illness = st.radio("Chronic illness history?", ["No", "Yes"])

if st.button("Calculate risk band & premium"):
    chronic_val = 1 if chronic_illness == "Yes" else 0
    input_df = pd.DataFrame([{
        "age": age,
        "income": income,
        "dependents": dependents,
        "chronic_illness": chronic_val,
        "education_years": education_years,
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]

    band_label = "High risk" if pred == 1 else "Low risk"
    band_class = "sa-band-high" if pred == 1 else "sa-band-low"
    premium = 150 if pred == 1 else 100

    st.markdown(f"""
    <div class="sa-result">
        <h3>Result</h3>
        <p>Risk classification: <span class="{band_class}">{band_label}</span> ({proba*100:.1f}% model confidence)</p>
        <p>Recommended daily premium:</p>
        <div class="sa-premium">KES {premium}</div>
        <p style="margin-top:14px; color:#5C6670; font-size:13px;">
            Premium is derived from the model's risk classification, consistent with the
            dynamic pricing structure validated in the full case study
            (Expected Present Value of Premiums vs. Benefits).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### What drove this classification")
    importances = pd.DataFrame({
        "Factor": ["Age", "Income", "Dependents", "Chronic illness", "Education (years)"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    st.bar_chart(importances.set_index("Factor"))

st.markdown("---")
st.markdown(
    "<p style='color:#5C6670; font-size:13px;'>This calculator runs a simplified, "
    "synthetic-data version of the model for demonstration purposes. "
    "For the full methodology, validation, and actuarial pricing framework, "
    "<a href='https://smartactuary.co.ke/#contact'>get in touch</a>.</p>",
    unsafe_allow_html=True
)

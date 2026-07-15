import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Smart Actuary — Micro-Insurance Risk Calculator",
    page_icon="Σ",
    layout="wide",
)

# ---------- Smart Actuary brand styling (dark navy theme) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #10233A; }
    section[data-testid="stSidebar"] { background-color: #0C1B2E; }

    h1, h2, h3 { font-family: 'Poppins', sans-serif; color: #F4EFE6 !important; }
    p, span, label, li, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        color: #C7D0DC !important;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        color: #F4EFE6 !important;
    }

    .sa-eyebrow {
        font-family: 'JetBrains Mono'; font-size: 12px; letter-spacing: 2px;
        color: #D9A441 !important; text-transform: uppercase;
    }
    .sa-title { font-size: 34px; font-weight: 800; color: #F4EFE6 !important; margin: 6px 0 18px; }

    .sa-card {
        background: #F4EFE6; border-radius: 12px; padding: 22px 26px; height: 100%;
        border-top: 4px solid #D9A441;
    }
    .sa-card h4 { color: #10233A !important; margin: 0 0 6px; font-family:'Poppins'; }
    .sa-card p, .sa-card span { color: #10233A !important; }
    .sa-card .big { font-family: 'JetBrains Mono'; font-size: 30px; font-weight: 600; color: #10233A !important; }
    .sa-card.risk-high { border-top-color: #b3541e; }
    .sa-card.risk-low { border-top-color: #2f7a4f; }

    .stButton>button {
        background-color: #D9A441; color: #412402; font-weight: 600;
        border-radius: 8px; border: none; padding: 10px 24px; width: 100%;
    }
    .stButton>button:hover { background-color: #c4913a; color: #412402; }

    .sa-plot-card { background: #F4EFE6; border-radius: 12px; padding: 18px; }

    hr { border-color: #26405F !important; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.markdown("### Policyholder Profile")
    age = st.slider("Age", 18, 65, 30, help="Age of the policyholder in years.")
    income = st.number_input("Monthly income (KES)", min_value=1000, max_value=100000, value=15000, step=500,
                              help="Gross monthly income, used as a proxy for financial resilience.")
    dependents = st.slider("Number of dependents", 0, 8, 1, help="People financially dependent on the policyholder.")
    education_years = st.slider("Years of formal education", 0, 17, 8, help="Total years of completed formal schooling.")
    chronic_illness = st.radio("Chronic illness history?", ["No", "Yes"], help="Any diagnosed chronic condition.")
    st.markdown("---")
    calc = st.button("Calculate risk band & premium")

# ---------- Header ----------
st.markdown('<span class="sa-eyebrow">Smart Σ Actuary · Live Model Demo</span>', unsafe_allow_html=True)
st.markdown('<div class="sa-title">Micro-Insurance Risk & Premium Calculator</div>', unsafe_allow_html=True)

st.markdown("""
This tool is a live, simplified version of the Random Forest risk classification model behind
Smart Actuary's [micro-insurance pricing case study](https://smartactuary.co.ke/#case-microinsurance).
Informal-sector workers are difficult to underwrite with traditional methods — income is irregular,
formal records are sparse, and one-size-fits-all pricing either overcharges low-risk users or
under-prices high-risk ones. This model classifies each policyholder into a risk band and derives a
dynamic daily premium from that classification.

**Model inputs:** age, monthly income, number of dependents, years of education, and chronic illness history.

**Method:** a Random Forest classifier (200 trees, max depth 6) trained on policyholder data, benchmarked
against Logistic Regression, KNN, SVM, and Decision Trees before selection — chosen for its accuracy and
resistance to overfitting on mixed data types. In the full case study, the model was validated with 5-fold
cross-validation and confirmed financially sustainable via Expected Present Value of Premiums vs. Benefits.
""")

st.markdown("---")

# ---------- Train the model ----------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 800
    age_ = np.random.randint(18, 65, n)
    income_ = np.random.randint(3000, 60000, n)
    dependents_ = np.random.randint(0, 6, n)
    chronic_ = np.random.binomial(1, 0.18, n)
    edu_ = np.random.randint(0, 17, n)
    risk_score = (
        0.03 * age_ - 0.00006 * income_ + 0.25 * dependents_
        + 1.8 * chronic_ - 0.05 * edu_ + np.random.normal(0, 0.6, n)
    )
    risk_band = (risk_score > np.median(risk_score)).astype(int)
    X = pd.DataFrame({"age": age_, "income": income_, "dependents": dependents_,
                       "chronic_illness": chronic_, "education_years": edu_})
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, risk_band)
    return model

model = train_model()

def predict(age_, income_, dependents_, chronic_, edu_):
    df = pd.DataFrame([{"age": age_, "income": income_, "dependents": dependents_,
                         "chronic_illness": chronic_, "education_years": edu_}])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return pred, proba

# ---------- Results ----------
st.markdown("### Model Parameters and Risk Classification")

param_df = pd.DataFrame({
    "Age": [age], "Monthly income (KES)": [income], "Dependents": [dependents],
    "Education (years)": [education_years], "Chronic illness": [chronic_illness]
})
st.dataframe(param_df, hide_index=True, use_container_width=True)

chronic_val = 1 if chronic_illness == "Yes" else 0
pred, proba = predict(age, income, dependents, chronic_val, education_years)
band_label = "High risk" if pred == 1 else "Low risk"
card_class = "risk-high" if pred == 1 else "risk-low"
premium = 150 if pred == 1 else 100

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="sa-card {card_class}">
        <h4>Risk classification</h4>
        <div class="big">{band_label}</div>
        <p>{proba[pred]*100:.1f}% model confidence</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="sa-card">
        <h4>Recommended daily premium</h4>
        <div class="big">KES {premium}</div>
        <p>Derived from risk classification, actuarially validated in the full case study.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Feature importance ----------
st.markdown("### What drives the model's decisions")
st.markdown("Relative importance of each factor across the full training population, not just this one profile.")

fi = pd.DataFrame({
    "Factor": ["Age", "Income", "Dependents", "Chronic illness", "Education (years)"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_fi = go.Figure(go.Bar(
    x=fi["Importance"], y=fi["Factor"], orientation="h",
    marker_color="#4A6FA5"
))
fig_fi.update_layout(
    plot_bgcolor="#F4EFE6", paper_bgcolor="#F4EFE6",
    font_color="#10233A", height=320, margin=dict(l=10, r=10, t=10, b=10)
)
st.markdown('<div class="sa-plot-card">', unsafe_allow_html=True)
st.plotly_chart(fig_fi, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- Sensitivity heatmap ----------
st.markdown("### Risk Probability Sensitivity")
st.markdown(
    "How predicted high-risk probability shifts across **age** and **income**, "
    "holding dependents, education, and chronic illness at your current sidebar values."
)

age_range = np.linspace(18, 65, 40)
income_range = np.linspace(3000, 60000, 40)
grid = np.zeros((len(income_range), len(age_range)))
for i, inc in enumerate(income_range):
    batch = pd.DataFrame({
        "age": age_range,
        "income": [inc] * len(age_range),
        "dependents": [dependents] * len(age_range),
        "chronic_illness": [chronic_val] * len(age_range),
        "education_years": [education_years] * len(age_range),
    })
    grid[i, :] = model.predict_proba(batch)[:, 1]

fig_hm = go.Figure(data=go.Heatmap(
    z=grid, x=age_range, y=income_range,
    colorscale=[[0, "#2f7a4f"], [0.5, "#D9A441"], [1, "#b3541e"]],
    colorbar=dict(title="P(high risk)")
))
fig_hm.update_layout(
    plot_bgcolor="#F4EFE6", paper_bgcolor="#F4EFE6",
    font_color="#10233A", height=420,
    xaxis_title="Age", yaxis_title="Monthly income (KES)",
    margin=dict(l=10, r=10, t=10, b=10)
)
st.markdown('<div class="sa-plot-card">', unsafe_allow_html=True)
st.plotly_chart(fig_hm, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "This calculator runs a simplified, synthetic-data version of the model for demonstration "
    "purposes. For the full methodology, validation, and actuarial pricing framework, "
    "[get in touch](https://smartactuary.co.ke/#contact)."
)

# Insurancecost_streamlit_inr.py
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sklearn

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(
    page_title="Medical Health Insurance Cost Prediction - India (₹)",
    page_icon="💊",
    layout="wide"
)

# --- Now you can use other Streamlit commands ---
st.caption(f"Using scikit-learn {sklearn.__version__}")



# -------------------- Global Styles --------------------
STYLES = """
<style>
:root {
  --brand1:#4e54c8; --brand2:#8f94fb;
}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.header-hero{
  background: linear-gradient(90deg, var(--brand1), var(--brand2));
  border-radius: 18px; padding: 18px 22px; color: #fff; margin-bottom: 20px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
.header-hero h1{ margin:0; font-size: 28px; font-weight: 800;}
.header-sub{ opacity:.95; font-size:14px; margin-top:6px}
.card{
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.12);
  border-radius:16px; padding:18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
  color:#e9eef7;
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stSlider > div > div { background:#111826 !important; color:#e9eef7 !important; border-radius:10px; border:1px solid #283244; }
.stSelectbox > div > div { background:#111826 !important; }
.stButton>button {
  background: linear-gradient(90deg, #06b6d4, #3b82f6);
  color:white; border:none; border-radius:12px; padding:10px 18px; font-weight:700;
}
.stButton>button:hover { transform: translateY(-1px) scale(1.02); }
.predict-box{
  background:#0b2a1a; border:1px solid #1e7f4f; color:#c9ffe1;
  padding:18px; border-radius:14px; text-align:center; font-weight:800; font-size:22px;
}
.tip{font-size:13px; opacity:.85}
hr.soft {border:none; border-top:1px dashed #2b3648; margin:14px 0}
footer { text-align:center; margin-top:2rem; font-size:12px; color:#888; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=88)
st.sidebar.markdown("### ⚙️ Menu")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🧮 Cost Prediction", "🩺 Symptom Checker", "📊 Insights", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Health Tips")
st.sidebar.write("• Drink 3–4L water/day 💧")
st.sidebar.write("• 7–8 hours sleep 😴")
st.sidebar.write("• 30 min brisk walk 🚶‍♂️")
st.sidebar.write("• Limit sugar & smoking 🚭")

# -------------------- Data/Model --------------------
MODEL_PATH = "model_joblib_gb.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Model load error: {e}")

def try_load_df():
    try:
        return pd.read_csv("insurance.csv")
    except Exception:
        return None

# Session history
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------- Header --------------------
st.markdown(
    """
    <div class="header-hero">
      <div class="badge">MEDICAL • AI</div>
      <h1>Medical Health Insurance Cost Prediction (India ₹)</h1>
      <div class="header-sub">Smart estimates for insurance charges + helpful health guidance (educational).</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Helper Functions --------------------

def usd_to_inr(usd):
    # Exchange rate approx (you can update as needed)
    return usd * 82.0

def calculate_emi(principal, annual_rate=10.0, months=12):
    # EMI Formula: E = P * r * (1 + r)^n / ((1 + r)^n - 1)
    r = annual_rate / (12 * 100)  # monthly interest rate in decimal
    emi = principal * r * (1 + r)**months / ((1 + r)**months - 1)
    return emi

# ===================================================
#                      PAGES
# ===================================================

# -------- HOME --------
if page == "🏠 Home":
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.markdown(
            """
            <div class="card">
            <h3>Welcome 👋</h3>
            This app helps you:
            <ul>
              <li><b>Predict insurance cost</b> using Gradient Boosting ML model.</li>
              <li><b>Auto-calculate BMI</b> from height & weight.</li>
              <li><b>Check symptoms</b> and get <i>diet / precautions / OTC</i> suggestions.</li>
              <li><b>View insights</b> from the dataset.</li>
            </ul>
            <div class="tip">Note: Health suggestions are general & educational — always consult a doctor for diagnosis and prescriptions.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with colB:
        st.markdown(
            """
            <div class="card">
              <h3>Get Started</h3>
              <ol>
                <li>Open <b>🧮 Cost Prediction</b> and enter details.</li>
                <li>Or try <b>🩺 Symptom Checker</b> for quick guidance.</li>
                <li>Explore <b>📊 Insights</b> to see patterns.</li>
              </ol>
              <hr class="soft"/>
              <b>Project:</b> Medical Health Insurance Cost Prediction<br/>
              <b>Author:</b> Deepanshu Adhikari (Modified for INR & EMI)
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- COST PREDICTION --------
elif page == "🧮 Cost Prediction":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧮 Insurance Cost Prediction")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Gender", ("Male", "Female"))
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            bmi = round(weight / ((height / 100) ** 2), 2)
            st.info(f"📌 Calculated BMI: **{bmi}**")

        with col2:
            children = st.slider("No. of Children", 0, 5, 0)
            smoker = st.selectbox("Smoker", ("Yes", "No"))
            region = st.selectbox("Region", ("Southwest", "Southeast", "Northwest", "Northeast"))
            st.markdown('<span class="tip">Regions map to dataset codes SW=1, SE=2, NW=3, NE=4</span>', unsafe_allow_html=True)

        sex_val = 1 if sex == "Male" else 0
        smoker_val = 1 if smoker == "Yes" else 0
        region_map = {"Southwest": 1, "Southeast": 2, "Northwest": 3, "Northeast": 4}
        region_val = region_map[region]

        btn_cols = st.columns([1, 3, 1])
        with btn_cols[1]:
            predict_clicked = st.button("🔍 Predict Cost", use_container_width=True)

        if predict_clicked:
            if model is None:
                st.error("Model not found or failed to load. Make sure 'model_joblib_gb.pkl' exists.")
            else:
                pred_usd = model.predict([[age, sex_val, bmi, children, smoker_val, region_val]])
                cost_usd = float(pred_usd[0])
                cost_inr = round(usd_to_inr(cost_usd), 2)
                emi = round(calculate_emi(cost_inr), 2)

                st.markdown(
                    f'<div class="predict-box">💰 Estimated Insurance Cost: <span style="font-size:26px">₹ {cost_inr:,}</span></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="predict-box" style="font-size:18px; padding: 12px;">💳 Estimated Monthly EMI (12 months, 10% p.a.): <b>₹ {emi:,}</b></div>',
                    unsafe_allow_html=True)

                st.session_state["history"].append([age, sex, bmi, children, smoker, region, cost_inr, emi])

        # History
        if st.session_state["history"]:
            st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
            st.markdown("#### 📜 Prediction History (this session)")
            hist_df = pd.DataFrame(
                st.session_state["history"],
                columns=["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Predicted Cost (₹)", "Monthly EMI (₹)"]
            )
            st.dataframe(hist_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# -------- SYMPTOM CHECKER --------
elif page == "🩺 Symptom Checker":
    st.subheader("🩺 Quick Symptom Checker (Educational)")

    CONDITIONS = {
        "Common Cold / Flu": {
            "symptoms": {"fever", "cold", "cough", "sore throat", "runny nose", "body ache", "sneezing", "chills"},
            "diet": [
                "Warm fluids: ginger-tulsi tea, soups, dal broth",
                "Soft foods: khichdi, oats, curd rice (if no sore throat)",
                "Vitamin C sources: oranges, amla, lemon water"
            ],
            "avoid": ["Cold drinks/ice-cream", "Oily/deep-fried food", "Dust/smoke exposure"],
            "otc": ["Paracetamol 500 mg for fever", "Steam inhalation", "Saline gargle", "Antihistamine (e.g., cetirizine) for runny nose"]
        },
        "Viral Fever": {
            "symptoms": {"fever", "headache", "body ache", "weakness", "chills"},
            "diet": ["Plenty of water, ORS", "Light meals: khichdi, bananas, soup", "Coconut water"],
            "avoid": ["Strenuous exercise", "Street food", "Caffeine in excess"],
            "otc": ["Paracetamol 500 mg (as per label)", "Sponge bath if fever high", "Rest"]
        },
        "Migraine / Headache": {
            "symptoms": {"headache", "nausea", "light sensitivity", "sound sensitivity"},
            "diet": ["Magnesium-rich: nuts, spinach", "Small frequent meals", "Hydration"],
            "avoid": ["Skipping meals", "Excess caffeine", "Bright screens for long time"],
            "otc": ["Paracetamol/Ibuprofen (as per label)", "Cold/warm compress", "Dark, quiet room rest"]
        },
        "Indigestion / Acidity": {
            "symptoms": {"acidity", "burning chest", "bloating", "belching", "nausea"},
            "diet": ["Plain curd/buttermilk", "Banana, rice, apple", "Small, frequent meals"],
            "avoid": ["Spicy/oily food", "Late-night heavy meals", "Caffeine, alcohol"],
            "otc": ["Antacid syrup/tablet (as per label)"]
        }
    }

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### Select your symptoms")
        options = sorted({s for v in CONDITIONS.values() for s in v["symptoms"]})
        chosen = st.multiselect("Common symptoms", options, help="Choose all that apply")
        other = st.text_input("Anything else (comma separated)?", placeholder="e.g., mild cough, nasal congestion")

        user_syms = set([s.strip().lower() for s in chosen])
        if other:
            user_syms.update([s.strip().lower() for s in other.split(",") if s.strip()])

        check = st.button("🧠 Get Suggestions", use_container_width=True)

        if check:
            if not user_syms:
                st.warning("Please enter/select at least one symptom.")
            else:
                scores = []
                for cond, info in CONDITIONS.items():
                    overlap = len(user_syms.intersection(info["symptoms"]))
                    scores.append((overlap, cond))
                scores.sort(reverse=True)
                best_overlap, best_cond = scores[0]

                if best_overlap == 0:
                    st.info("🤔 We couldn't match your symptoms well. Please consult a doctor if symptoms persist.")
                else:
                    info = CONDITIONS[best_cond]
                    st.success(f"🩺 You may be experiencing: **{best_cond}** (approximation)")
                    with st.expander("🥗 What to eat"):
                        st.write("\n".join([f"• {d}" for d in info["diet"]]))
                    with st.expander("🚫 What to avoid"):
                        st.write("\n".join([f"• {a}" for a in info["avoid"]]))
                    with st.expander("💊 Possible OTC support (read label, allergies, age limits)"):
                        st.write("\n".join([f"• {m}" for m in info["otc"]]))
                    st.markdown(
                        "<div class='tip'>⚠️ This is not medical advice. For severe, persistent, or unusual symptoms "
                        "seek professional medical care immediately.</div>",
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

# -------- INSIGHTS --------
elif page == "📊 Insights":
    st.subheader("📊 Dataset Insights")
    df = try_load_df()
    if df is None:
        st.warning("`insurance.csv` not found. Keep it in the project root to view insights.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Average Charges by Smoker")
            fig, ax = plt.subplots()
            df.groupby("smoker")["charges"].mean().plot(kind="bar", ax=ax)
            ax.set_xticklabels(["No", "Yes"], rotation=0)
            ax.set_ylabel("Avg Charges")
            st.pyplot(fig)

        with c2:
            st.markdown("##### Age vs Charges (sample)")
            fig2, ax2 = plt.subplots()
            sample = df.sample(min(300, len(df)), random_state=42)
            ax2.scatter(sample["age"], sample["charges"], alpha=0.5)
            ax2.set_xlabel("Age")
            ax2.set_ylabel("Charges")
            st.pyplot(fig2)

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

# -------- ABOUT --------
elif page == "ℹ️ About":
    st.subheader("ℹ️ About this Project")
    st.markdown(
        """
        - **Project:** Medical Health Insurance Cost Prediction  
        - **Model:** Gradient Boosting Regression  
        - **Currency:** Converted USD to Indian Rupees (₹)  
        - **Additional:** Monthly EMI calculation (12 months, 10% annual interest)  
        - **Tech:** Python, Streamlit, scikit-learn, joblib, matplotlib  
        - **Author:** Deepanshu Adhikari (Modified for INR & EMI)  
        - **Disclaimer:** This is an educational tool, not a substitute for professional medical or financial advice.
        """
    )
    st.markdown("---")
    st.markdown("© 2025 Medical AI Solutions")

# -------- FOOTER --------
st.markdown(
    """
    <footer>
      <small>Built with ❤️ using Streamlit. Model and dataset are for educational use only.</small>
    </footer>
    """,
    unsafe_allow_html=True
)

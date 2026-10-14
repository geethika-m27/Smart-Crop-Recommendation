import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 2. Custom styling
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f2faf4 0%, #ffffff 50%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(120deg, #064e3b, #15803d);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(6, 78, 59, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0f2e9;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-title {
        color: #14532d;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 10px 0 14px 0;
    }

    .info-card {
        background: white;
        border: 1px solid #dceee1;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px rgba(20, 83, 45, 0.05);
    }

    .crop-name {
        color: #14532d;
        font-size: 1.1rem;
        font-weight: 700;
    }

    .crop-score {
        color: #15803d;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #15803d, #047857);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #166534;
        color: white;
        border: none;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dceee1;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #dceee1;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 22px 5px 8px 5px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌱 Smart Crop Recommendation</h1>
    <p>
        Make informed crop choices using soil nutrients
        and environmental conditions.
    </p>
    <p style="margin-top:14px;">
        🤖 Machine Learning &nbsp; • &nbsp;
        🌾 22 Crop Classes &nbsp; • &nbsp;
        📊 Prediction Scores
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 4. Load the trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("crop_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.caption(str(error))
    st.stop()

# --------------------------------------------------
# 5. Input form
# --------------------------------------------------
st.markdown(
    '<div class="section-title">🧪 Soil and Weather Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the available field measurements, then request recommendations.")

with st.form("crop_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)", min_value=0.0,
            max_value=200.0, value=90.0, step=1.0
        )
        P = st.number_input(
            "Phosphorus (P)", min_value=0.0,
            max_value=200.0, value=42.0, step=1.0
        )
        K = st.number_input(
            "Potassium (K)", min_value=0.0,
            max_value=250.0, value=43.0, step=1.0
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0,
            max_value=60.0, value=25.0, step=0.5
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0,
            max_value=100.0, value=80.0, step=1.0
        )
        ph = st.number_input(
            "Soil pH", min_value=0.0,
            max_value=14.0, value=6.5, step=0.1
        )

    with col3:
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0,
            max_value=5000.0, value=200.0, step=10.0
        )

        st.markdown("""
        <div class="info-card">
            <b>💡 Tip</b><br>
            Enter measurements from your soil test
            and local weather data when available.
        </div>
        """, unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🌿 Recommend Crops",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 6. Prediction results
# --------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    try:
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[list(model.feature_names_in_)]

        probabilities = model.predict_proba(input_data)[0]
        crop_names = model.classes_

        results = pd.DataFrame({
            "Crop": crop_names,
            "Prediction Score (%)": probabilities * 100
        })

        results = results.sort_values(
            "Prediction Score (%)",
            ascending=False
        ).reset_index(drop=True)

        st.markdown("---")
        st.markdown(
            '<div class="section-title">🏆 Top 3 Recommendations</div>',
            unsafe_allow_html=True
        )

        top_three = results.head(3)
        recommendation_columns = st.columns(3)

        for index, (_, row) in enumerate(top_three.iterrows()):
            with recommendation_columns[index]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="crop-name">
                            {'🥇' if index == 0 else '🥈' if index == 1 else '🥉'}
                            {str(row['Crop']).title()}
                        </div>
                        <div class="crop-score">
                            {row['Prediction Score (%)']:.2f}%
                        </div>
                        <div class="small-note">
                            Model-estimated probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(float(row["Prediction Score (%)"]) / 100, 0), 1)
                )

        st.success(
            f"Highest-scoring crop: "
            f"{str(results.iloc[0]['Crop']).title()}"
        )

        # All crop scores
        st.markdown(
            '<div class="section-title">📊 Scores for All Crop Classes</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Ranked by model-estimated probability. "
            "These scores are not guarantees of cultivation success."
        )

        display_results = results.copy()
        display_results.insert(
            0, "Rank", range(1, len(display_results) + 1)
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank", format="%d"
                ),
                "Crop": st.column_config.TextColumn("Crop"),
                "Prediction Score (%)": st.column_config.ProgressColumn(
                    "Prediction Score (%)",
                    min_value=0,
                    max_value=100,
                    format="%.2f%%"
                )
            }
        )

        st.caption(
            f"Model classes available: {len(model.classes_)}"
        )

    except Exception as error:
        st.error("Unable to generate crop recommendations.")
        st.caption(str(error))

# --------------------------------------------------
# 7. Project information
# --------------------------------------------------
with st.expander("ℹ️ About this project"):
    info1, info2 = st.columns(2)

    with info1:
        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Dataset:** 2,200 records")
        st.write("**Crop classes:** 22")

    with info2:
        st.write("**Input features:** 7")
        st.write("**Reported test accuracy:** 99.32%")
        st.write("**Framework:** Streamlit")

    st.info(
        "This is an educational decision-support tool. "
        "Verify recommendations with local agricultural expertise "
        "before making farming decisions."
    )

# --------------------------------------------------
# 8. Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    🌱 Smart Crop Recommendation System<br>
    Powered by Python, Streamlit and Random Forest Machine Learning
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import pickle

# Step 1: Configure the page
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Crop Recommendation System")
st.write(
    "An ML-based decision support tool that recommends crops "
    "using soil nutrients and weather conditions."
)

# Step 2: Load the trained model
@st.cache_resource
def load_model():
    with open("crop_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Step 3: Display input fields
st.header("Soil and Weather Information")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
    P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
    K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
    temperature = st.number_input(
        "Temperature (°C)", -10.0, 60.0, 25.0
    )

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 200.0)

# Step 4: Recommend crops
if st.button("Recommend Crops", type="primary"):

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Ensure the input columns match the model's training features
    input_data = input_data[model.feature_names_in_]

    # Predict probabilities for all learned crop classes
    probabilities = model.predict_proba(input_data)[0]
    crop_names = model.classes_

    results = pd.DataFrame({
        "Crop": crop_names,
        "Prediction Score (%)": probabilities * 100
    })

    results = results.sort_values(
        by="Prediction Score (%)",
        ascending=False
    ).reset_index(drop=True)

    st.subheader("🌾 Top 3 Crop Recommendations")
    top_three = results.head(3)

    for _, row in top_three.iterrows():
        st.metric(
            label=str(row["Crop"]).title(),
            value=f'{row["Prediction Score (%)"]:.2f}%'
        )

    st.subheader("📊 Prediction Scores for All 22 Crops")
    st.caption(
        "Scores are model-estimated probabilities, not guarantees "
        "of successful cultivation. Some crops may have a score of 0%."
    )

    st.dataframe(
        results.style.format({
            "Prediction Score (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Highest-scoring candidate: "
        f"{str(results.iloc[0]['Crop']).title()}"
    )

# Step 5: Show project information
with st.expander("About this ML project"):
    st.write("Algorithm: Random Forest Classifier")
    st.write(
        "Input features: N, P, K, temperature, humidity, pH and rainfall."
    )
    st.write("Target: Crop label from the training dataset.")
    st.write(f"Crop classes learned: {len(model.classes_)}")

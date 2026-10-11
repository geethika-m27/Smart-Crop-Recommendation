
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

# Step 2: Load the existing trained model
with open("crop_model.pkl", "rb") as file:
    model = pickle.load(file)

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

# Step 4: Predict the top three candidate crops
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

    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    top_indices = probabilities.argsort()[::-1][:3]

    st.header("🌾 Top 3 Crop Recommendations")

    results = pd.DataFrame({
        "Crop": classes[top_indices],
        "Model Probability (%)": (
            probabilities[top_indices] * 100
        ).round(2)
    })

    results = results.reset_index(drop=True)
    results.index = results.index + 1

    st.dataframe(results, use_container_width=True)

    best_crop = classes[top_indices[0]]
    st.success(f"Top candidate crop: {str(best_crop).title()}")

    st.caption(
        "Probabilities are model estimates, not guaranteed chances "
        "of successful cultivation. The top three are candidates, "
        "not a verified ranking of agricultural suitability."
    )

# Step 5: Show project information
with st.expander("About this ML project"):
    st.write(
        "Algorithm: Random Forest Classifier"
    )
    st.write(
        "Input features: N, P, K, temperature, humidity, pH and rainfall."
    )
    st.write(
        "Target: Crop label from the training dataset."
    )

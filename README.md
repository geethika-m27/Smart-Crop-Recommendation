# 🌱 Smart Crop Recommendation System

## Project Overview

The Smart Crop Recommendation System is a machine learning web application that recommends suitable crops based on soil nutrients and environmental conditions.

## Features

- Interactive Streamlit web interface
- Random Forest classification model
- Supports 22 crop classes
- Accepts seven soil and weather inputs
- Displays the top 3 crop recommendations
- Shows prediction scores for all crop classes
- Evaluates model performance using a test dataset

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

## Dataset

The project uses a crop recommendation dataset containing 2,200 records and 22 crop classes.

Input features:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Target: Crop label

## Model Performance

Algorithm: Random Forest Classifier

Test accuracy: 99.32%

The accuracy was measured on a held-out test set. Actual agricultural outcomes may differ from model predictions.

## Project Files

- `app.py` — Streamlit application
- `train_model.py` — Model training and evaluation
- `crop_model.pkl` — Trained machine learning model
- `Crop_recommendation.csv` — Dataset
- `requirements.txt` — Python dependencies

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

## Live Application
**[Click here to open the live website] 
(https://smart-crop-recommendation-gorfmyhwiskydhaxwytibk.streamlit.app/)







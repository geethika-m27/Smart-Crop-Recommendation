
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the complete dataset
data = pd.read_csv("Crop_recommendation.csv")

# Prepare the input features and target
features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

X = data[features]
y = data["label"]

# Check the crop classes available in the dataset
print("Total dataset rows:", len(data))
print("Total crop classes:", y.nunique())
print("Available crops:", sorted(y.unique()))

# Split data to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train using all classes represented in the training split
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions))

# Save the trained model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully.")
print("Classes learned:", len(model.classes_))
print("Crop classes:", sorted(model.classes_))

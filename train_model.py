
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

print("Dataset Preview:")
print(data.head())

# Step 2: Separate input features and target
X = data.drop("label", axis=1)
y = data["label"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict crops
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Save the trained model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as crop_model.pkl")

import pandas as pd
import joblib
import os
import sys

# ✅ 1️⃣ Config paths
MODEL_PATH = r'D:\projects\breast cnacer project\cancer_model.pkl'
SCALER_PATH = r'D:\projects\breast cnacer project\scaler.pkl'
OUTPUT_PATH = r'D:\projects\breast cnacer project\new_patient.csv'

# ✅ 2️⃣ Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model and scaler loaded successfully.\n")
except Exception as e:
    print(f"❌ ERROR loading model/scaler: {e}")
    sys.exit(1)

# ✅ 3️⃣ Define feature columns
columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# ✅ 4️⃣ Take user input with validation
print("🔢 Please enter the following 30 feature values for the tumor sample:")
input_values = []

for feature in columns:
    while True:
        try:
            value = float(input(f"➡️  {feature}: "))
            input_values.append(value)
            break
        except ValueError:
            print("⚠️ Invalid input! Please enter a numeric value.")

# ✅ 5️⃣ Convert to DataFrame
patient_df = pd.DataFrame([input_values], columns=columns)
print("\n✅ Input received successfully.")

# ✅ 6️⃣ Scale input
scaled_input = scaler.transform(patient_df)
print("✅ Input scaled.")

# ✅ 7️⃣ Make prediction
prediction = model.predict(scaled_input)[0]
result_label = 'Benign' if prediction == 1 else 'Malignant'

# ✅ 8️⃣ Print and save result
print(f"\n🎯 Prediction Result: The tumor is **{result_label.upper()}**")

# Add prediction column to saved file
patient_df['prediction'] = result_label
try:
    patient_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Prediction and input saved to: {OUTPUT_PATH}")
except Exception as e:
    print(f"⚠️ Could not save CSV: {e}")

print("\n✅ Prediction complete.")

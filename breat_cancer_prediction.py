import pandas as pd
import joblib
import os

# 1️⃣ Load model and scaler
model = joblib.load(r'D:\projects\breast cnacer project\cancer_model.pkl')
scaler = joblib.load(r'D:\projects\breast cnacer project\scaler.pkl')
print("✅ Model and scaler loaded successfully.\n")

# 2️⃣ Define columns (same as your training data, without diagnosis)
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

# 3️⃣ Take user input for all columns
input_values = []
for col in columns:
    val = float(input(f"Enter {col}: "))
    input_values.append(val)

# 4️⃣ Convert to DataFrame
new_patient_df = pd.DataFrame([input_values], columns=columns)

# 5️⃣ Scale the input
new_patient_scaled = scaler.transform(new_patient_df)

# 6️⃣ Make prediction
prediction = model.predict(new_patient_scaled)[0]
result = 'Malignant' if prediction == 0 else 'Benign'

print(f"\n✅ Prediction: {result}")

# 7️⃣ Add prediction column
new_patient_df['prediction'] = result

# 8️⃣ Save to new_patient.csv (overwrite)
output_path = r'D:\projects\breast cnacer project\new_patient.csv'
new_patient_df.to_csv(output_path, index=False)
print(f"✅ Input and prediction saved to {output_path}")

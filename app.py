from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ✅ Load trained model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# ✅ Define the input features
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

@app.route('/')
def home():
    return render_template('home.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Collect input values from the form
        input_data = []
        for feature in columns:
            val = float(request.form[feature])
            input_data.append(val)

        # ✅ Convert to DataFrame
        df = pd.DataFrame([input_data], columns=columns)

        # ✅ Scale the data
        scaled_input = scaler.transform(df)

        # ✅ Make prediction
        prediction = model.predict(scaled_input)[0]
        result = 'Benign' if prediction == 1 else 'Malignant'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)

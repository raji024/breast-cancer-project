from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ========== Load model, scaler, data ==========
MODEL_PATH = r'D:\projects\breast cnacer project\cancer_model.pkl'
SCALER_PATH = r'D:\projects\breast cnacer project\scaler.pkl'
DATA_PATH = r'D:\projects\breast cnacer project\data.csv'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define features
FEATURES = df.columns.drop('diagnosis').tolist()

# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('home.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_values = [float(request.form.get(feature)) for feature in FEATURES]
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        result = 'Benign' if prediction == 1 else 'Malignant'
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"

@app.route('/eda')
def eda():
    plot_folder = os.path.join('static', 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    # 1. Histogram
    plt.figure(figsize=(20, 15))
    df.hist(bins=30, color='steelblue')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'histogram.png'))
    plt.close()

    # 2. Scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='radius_mean',
        y='texture_mean',
        hue='diagnosis',
        palette={0: 'green', 1: 'red'}
    )
    plt.title('Scatterplot of Radius Mean vs Texture Mean')
    plt.savefig(os.path.join(plot_folder, 'scatterplot.png'))
    plt.close()

    # 3. Heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(os.path.join(plot_folder, 'heatmap.png'))
    plt.close()

    # 4. Pairplot
    subset = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']]
    sns.pairplot(subset, hue='diagnosis', palette={0: 'green', 1: 'red'})
    plt.savefig(os.path.join(plot_folder, 'pairplot.png'))
    plt.close()

    # Render EDA page with images
    plots = {
        'Histogram': os.path.join('static', 'plots', 'histogram.png'),
        'Scatterplot': os.path.join('static', 'plots', 'scatterplot.png'),
        'Heatmap': os.path.join('static', 'plots', 'heatmap.png'),
        'Pairplot': os.path.join('static', 'plots', 'pairplot.png')
    }

    return render_template('eda.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)

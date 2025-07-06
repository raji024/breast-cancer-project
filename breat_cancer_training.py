# breast_cancer_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset
df = pd.read_csv(r'D:\projects\breast cnacer project\data.csv')
print("✅ Loaded dataset:")
print(df.head())

# 2️⃣ Drop useless columns
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])
df = df.dropna()  # Drop rows with any NaNs
print("\n✅ After cleaning:", df.shape)

# 3️⃣ Encode target: M = 0 (Malignant), B = 1 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

# 4️⃣ Split features and labels
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 5️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# 8️⃣ Evaluate
y_pred = model.predict(X_test_scaled)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9️⃣ Save model and scaler
joblib.dump(model, r'D:\projects\breast cnacer project\cancer_model.pkl')
joblib.dump(scaler, r'D:\projects\breast cnacer project\scaler.pkl')

print("\n✅ Model and scaler saved successfully!")

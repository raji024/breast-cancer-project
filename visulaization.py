import pandas as pd
import matplotlib.pyplot as plt

# Load your training data
df = pd.read_csv(r"D:\projects\breast cnacer project\data.csv")

# Optionally drop unnecessary columns (like ID, unnamed)
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# Convert diagnosis to 0/1 if not done yet
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Histograms
df.hist(bins=30, figsize=(20, 15), color='steelblue')
plt.tight_layout()
plt.show()
import seaborn as sns

# Example: radius_mean vs texture_mean, colored by diagnosis
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='radius_mean',
    y='texture_mean',
    hue='diagnosis',
    palette={0:'green', 1:'red'}
)
plt.title('Scatterplot of Radius Mean vs Texture Mean')
plt.show()
plt.figure(figsize=(16, 14))
corr_matrix = df.corr()

sns.heatmap(
    corr_matrix,
    annot=False,
    cmap='coolwarm',
    linewidths=0.5
)
plt.title('Correlation Matrix Heatmap')
plt.show()
subset = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']]
sns.pairplot(subset, hue='diagnosis', palette={0:'green', 1:'red'})
plt.show()

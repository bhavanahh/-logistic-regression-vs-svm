import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/fertilizer.csv")

# Remove ID column
df = df.drop(columns=["ID"])

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values if present
for column in df.columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].median())

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("\nCrop distribution:")
print(df["Crop"].value_counts().sort_index())

# Plot 1: Crop distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Crop")
plt.title("Crop Class Distribution")
plt.tight_layout()
plt.savefig("plots/1_crop_distribution.png")
plt.close()

# Plot 2: Carbon histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Carbon", kde=True)
plt.title("Carbon Distribution")
plt.tight_layout()
plt.savefig("plots/2_carbon_distribution.png")
plt.close()

# Plot 3: Rainfall vs Temperature
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Rainfall", y="Temperature", hue="Crop")
plt.title("Rainfall vs Temperature by Crop")
plt.tight_layout()
plt.savefig("plots/3_rainfall_vs_temperature.png")
plt.close()

# Plot 4: Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/4_correlation_heatmap.png")
plt.close()

print("\nCleaned dataset saved successfully!")
print("Four plots saved in the plots folder.")
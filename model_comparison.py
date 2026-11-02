import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Separate features and target
X = df.drop(columns=["Crop"])
y = df["Crop"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Crop classes:", sorted(y.unique()))

# Define models and parameter grids
models = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
    )
}

results = []
trained_models = {}

# Train and tune each model
for name, (pipeline, parameters) in models.items():
    print(f"\n{'=' * 45}")
    print("Training:", name)

    search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    trained_models[name] = best_model

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Best Parameters": str(search.best_params_)
    })

    print("Best parameters:", search.best_params_)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
        colorbar=False
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"plots/{filename}_confusion_matrix.png")
    plt.close()

    # Save individual predictions
    prediction_df = pd.DataFrame({
        "Actual_Crop": y_test.to_numpy(),
        "Predicted_Crop": predictions
    })
    prediction_df.to_csv(
        f"results/{filename}_predictions.csv",
        index=False
    )

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n" + "=" * 45)
print("FINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

# Plot metric comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

plot_data = results_df.set_index("Model")[metrics]

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0
)

ax.set_title("Logistic Regression vs SVM")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/model_performance_comparison.png")
plt.close()

print("\nComparison saved to results/model_comparison.csv")
print("Performance graph saved to plots/model_performance_comparison.png")
print("Confusion matrices and prediction CSVs saved successfully.")
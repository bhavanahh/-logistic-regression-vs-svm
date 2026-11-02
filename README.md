# Logistic Regression vs SVM: Multi-Class Crop Classification

## Project Overview

This project compares Logistic Regression and Support Vector Machine (SVM) algorithms for multi-class crop classification.

The dataset contains soil and environmental features, including Carbon, Moisture, Phosphorous, Potassium, Rainfall, and Temperature. The models use these features to predict one of five crop classes.

## Dataset

- Original records: 500
- Features used: 6
- Target variable: Crop
- Number of classes: 5
- Missing values: 0
- Duplicate rows in original dataset: 0

The ID column was removed because it is an identifier rather than a useful predictive feature.

The cleaned dataset is saved as `data/cleaned_dataset.csv`.

Each crop class contains 100 records, so the original dataset is balanced across the five classes.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Preprocessing

The preprocessing script performs the following operations:

1. Loads the original dataset.
2. Removes the ID column.
3. Removes duplicate rows.
4. Handles missing numerical values if present.
5. Saves the cleaned dataset.
6. Generates four exploratory visualizations.

## Exploratory Data Analysis

Four plots are generated:

1. **Crop Class Distribution:** Displays the number of records in each crop class.
2. **Carbon Distribution:** Shows the distribution of Carbon values.
3. **Rainfall vs Temperature:** Visualizes the relationship between rainfall and temperature for each crop class.
4. **Correlation Heatmap:** Displays correlations between numerical columns.

The plots are stored in the `plots/` directory.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates class probabilities. It uses a logistic function and can be extended to multiple classes.

The regularization parameter `C` controls the strength of regularization inversely: smaller values apply stronger regularization, while larger values apply weaker regularization.

### 2. Support Vector Machine (SVM)

SVM is a supervised learning algorithm that separates classes by finding decision boundaries with large margins.

The `C` parameter controls the trade-off between a wider margin and classification errors. SVM can use different kernels to model decision boundaries.

### Feature Scaling

StandardScaler is applied through a scikit-learn pipeline before model training. This is important because the features have different numerical scales.

### Training and Testing

- Training data: 80%
- Testing data: 20%
- Train-test split random state: 42
- Stratified splitting preserves class proportions.
- GridSearchCV with 5-fold cross-validation is used to tune model parameters.
- Macro F1-score is used to select the best parameter combinations.

## Evaluation Metrics

Four metrics are used to evaluate the models.

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** How reliable positive predictions are, averaged equally across classes.
- **Recall:** How well the model identifies actual examples of each class, averaged equally across classes.
- **F1-score:** The harmonic mean of precision and recall, averaged equally across classes.

## Model Comparison Results

| Metric | Logistic Regression | SVM |
|---|---:|---:|
| Accuracy | 82.00% | 86.00% |
| Precision (macro) | 83.44% | 86.94% |
| Recall (macro) | 82.00% | 86.00% |
| F1-score (macro) | 81.67% | 85.86% |
| Best C | 10 | 10 |
| Best SVM kernel | Not applicable | Linear |

### Interpretation

SVM achieved 86% test accuracy, compared with 82% for Logistic Regression.

SVM also achieved higher macro precision, recall, and F1-score. Therefore, SVM performed better on this particular train-test split.

The test set contained 100 samples. Logistic Regression correctly classified 82 samples, while SVM correctly classified 86 samples.

These results describe performance on this dataset and test split; they do not guarantee the same performance on new datasets.

## Performance Visualizations

The project generates:

- `plots/model_performance_comparison.png`
- `plots/logistic_regression_confusion_matrix.png`
- `plots/svm_confusion_matrix.png`

These plots help compare the evaluation metrics and examine classification errors.

## Prediction Results

The `results/` directory contains:

- `logistic_regression_predictions.csv`
- `svm_predictions.csv`
- `model_comparison.csv`

The prediction CSV files contain actual and predicted crop classes for the test samples.

## Project Structure

```text
logistic-regression-vs-svm/
├── data/
│   ├── fertilizer.csv
│   └── cleaned_dataset.csv
├── plots/
├── results/
├── preprocessing.py
├── model_comparison.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the original dataset at:

```text
data/fertilizer.csv
```

### 3. Run preprocessing

```bash
python preprocessing.py
```

### 4. Train and evaluate the models

```bash
python model_comparison.py
```

The scripts generate the cleaned dataset, exploratory plots, model comparison results, confusion matrices, and prediction CSV files.

## Conclusion

Both Logistic Regression and SVM successfully classified the five crop classes.

SVM achieved the better result, with 86% accuracy and an 85.86% macro F1-score. Logistic Regression achieved 82% accuracy and an 81.67% macro F1-score.

Based on the evaluation results, SVM is the better-performing model for this experiment.
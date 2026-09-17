# Credit Card Fraud Detection

## 📌 Project Overview

Credit card fraud is a major problem in digital transactions. The objective of this project is to build a machine learning model that can identify whether a credit card transaction is **legitimate or fraudulent**.

In this project, **XGBoost Classifier** is used to classify transactions. Since fraudulent transactions are much fewer than legitimate transactions, the dataset has a **highly imbalanced class distribution**.

---

## 🎯 Objective

The main objectives of this project are:

* Detect fraudulent credit card transactions.
* Understand the class imbalance in the dataset.
* Perform basic data preprocessing.
* Analyze transaction patterns using data visualization.
* Train an XGBoost classification model.
* Evaluate the model using multiple classification metrics.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset** stored as:

```text
creditcard.csv
```

The dataset contains transaction information along with a target column called `Class`.

### Target Variable

```text
Class = 0 → Legitimate transaction
Class = 1 → Fraudulent transaction
```

The dataset is highly imbalanced, with legitimate transactions being much more common than fraudulent transactions.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Load Data
   ↓
Check Missing Values
   ↓
Handle Missing Values
   ↓
Analyze Class Distribution
   ↓
Exploratory Data Analysis
   ↓
Feature Analysis
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
XGBoost Model
   ↓
Predictions
   ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Check

The dataset is checked for missing values.

```python
df.isnull().sum().sum()
```

If missing values are present, rows containing missing values are removed using:

```python
df.dropna(inplace=True)
```

### 2. Feature Engineering

A new feature called `hour_of_day` is created from the `Time` column:

```python
df['hour_of_day'] = (df['Time'] // 3600) % 24
```

This helps analyze transactions according to the hour of the day.

### 3. Removing Time

The original `Time` column is removed before model training:

```python
df_cleaned = df.drop(columns=['Time'])
```

### 4. Feature Scaling

The `Amount` feature is standardized using `StandardScaler`:

```python
scaler = StandardScaler()
df_cleaned['Amount'] = scaler.fit_transform(df_cleaned[['Amount']])
```

---

## 🔍 Exploratory Data Analysis

The notebook performs several visualizations to understand the dataset.

### Class Distribution

The number of legitimate and fraudulent transactions is visualized using a bar chart with a logarithmic y-axis.

### Transaction Amount

The distribution of transaction amounts for legitimate and fraudulent transactions is compared using histograms.

### Transactions by Hour

The `hour_of_day` feature is used to visualize the number of legitimate and fraudulent transactions at different hours.

### Feature Correlation

The correlation of different features with the `Class` variable is calculated and visualized.

### Important Features

The four features with the highest absolute correlation with the target are selected and displayed using boxplots.

---

## 🤖 Model Used

### XGBoost Classifier

The project uses **XGBoost (Extreme Gradient Boosting)** for fraud classification.

The model parameters used in the notebook are:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)
```

### Parameter Explanation

* `n_estimators=100` → Number of boosting trees.
* `max_depth=5` → Maximum depth of each tree.
* `learning_rate=0.1` → Controls the contribution of each tree.
* `scale_pos_weight` → Used to give additional importance to the minority fraud class.
* `random_state=42` → Makes the results reproducible.
* `eval_metric="logloss"` → Evaluation metric used during training.

---

## ⚖️ Class Imbalance

Fraud detection is a highly imbalanced classification problem because fraudulent transactions are much fewer than legitimate transactions.

The notebook checks the class distribution using:

```python
print(df["Class"].value_counts())
```

and also checks the normalized class distribution:

```python
print(y.value_counts(normalize=True))
```

The XGBoost model uses `scale_pos_weight` to account for the imbalance.

---

## ✂️ Train-Test Split

The data is divided into training and testing sets using an **80:20 ratio**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` maintains a similar proportion of legitimate and fraudulent transactions in both sets.

---

## 📈 Model Evaluation

The model is evaluated using several metrics.

### Accuracy

Measures the overall percentage of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were correctly detected.

### F1 Score

Provides a balance between Precision and Recall.

### Confusion Matrix

The confusion matrix shows:

```text
                 Predicted
                 Legit   Fraud

Actual Legit       TN      FP
Actual Fraud       FN      TP
```

Where:

* **TN** → Legitimate transaction correctly classified.
* **TP** → Fraudulent transaction correctly classified.
* **FP** → Legitimate transaction incorrectly classified as fraud.
* **FN** → Fraudulent transaction incorrectly classified as legitimate.

### ROC-AUC

ROC-AUC measures how well the model distinguishes between legitimate and fraudulent transactions.

The notebook also plots the ROC curve using:

```python
RocCurveDisplay.from_predictions(y_test, y_probability)
```

---

## 📊 Model Performance

The model produced the following results in the notebook run:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 99.83% |
| Precision | 50.00% |
| Recall    | 84.69% |
| F1 Score  | 62.88% |
| ROC-AUC   | 97.34% |

Because the dataset is highly imbalanced, accuracy should not be considered alone. Precision, Recall, F1-score and ROC-AUC provide additional information about the model's fraud-detection performance.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
│
├── CreditCard_fraud_detection.ipynb
│
└── README.md
```

---

## ▶️ How to Run

### 1. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 2. Open the notebook

```text
CreditCard_fraud_detection.ipynb
```

### 3. Make sure `creditcard.csv` is available at the path used in the notebook.

### 4. Run the notebook cells from top to bottom.

---

## 🔑 Key Learning Outcomes

Through this project, we learn:

* How credit card fraud detection is treated as a binary classification problem.
* How to check and handle missing values.
* How to analyze an imbalanced dataset.
* How to perform basic exploratory data analysis.
* How to create a time-based feature such as `hour_of_day`.
* How to scale numerical features.
* How XGBoost can be used for classification.
* How class imbalance can be handled using `scale_pos_weight`.
* How to interpret a confusion matrix.
* Why Accuracy alone is not sufficient for fraud detection.
* How Precision, Recall, F1-score and ROC-AUC are used for evaluation.

---

## 📌 Conclusion

This project demonstrates the use of **XGBoost for credit card fraud detection**. The dataset is highly imbalanced, making proper preprocessing and evaluation important.

The model is evaluated using Accuracy, Precision, Recall, F1-score, Confusion Matrix and ROC-AUC. The project shows why multiple evaluation metrics are useful when detecting fraudulent transactions in an imbalanced dataset.

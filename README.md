# Hospital Readmission Prediction

## 📌 Project Overview

This project predicts whether a diabetic patient will be **readmitted to the hospital within 30 days** using **Logistic Regression with L2 Regularization**.

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset. The dataset contains patient records collected from multiple hospitals.

The main objective is to use patient information and previous hospital-related information to predict the risk of readmission within 30 days.

---

## 🎯 Problem Statement

Hospital readmissions can increase healthcare costs and may indicate that a patient requires additional care or follow-up.

In this project, a machine learning model is developed to predict:

* `1` → Patient is readmitted within 30 days
* `0` → Patient is not readmitted within 30 days

The `readmitted` column from the original dataset is converted into a binary target variable called `readmitted_30`.

---

## 📊 Dataset

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008

The dataset contains information related to diabetic patients, including:

* Patient demographics
* Hospital visits
* Number of diagnoses
* Laboratory procedures
* Procedures performed
* Medications
* Emergency visits
* Outpatient visits
* Inpatient visits
* Hospital stay information

### Target Variable

The original `readmitted` column contains three values:

| Value | Meaning                   |
| ----- | ------------------------- |
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days  |
| `NO`  | Not readmitted            |

For this project:

```text
<30  → 1
>30  → 0
NO   → 0
```

Therefore, the model specifically predicts **30-day readmission**.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Load Data
   ↓
Check Missing Values
   ↓
Replace '?' with Missing Values
   ↓
Create 30-Day Readmission Target
   ↓
Remove Unnecessary Columns
   ↓
Handle Missing Values
   ↓
Exploratory Data Analysis
   ↓
Encode Categorical Features
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression with L2 Regularization
   ↓
Predictions
   ↓
Model Evaluation
   ↓
ROC-AUC & Confusion Matrix
```

---

## 🧹 Data Preprocessing

### 1. Missing Values

The dataset uses `?` to represent missing values.

These values are replaced with missing-value markers.

Categorical missing values are filled using the **mode**, while numerical missing values are filled using the **median**.

### 2. Removing Unnecessary Columns

The following columns are removed:

```text
encounter_id
patient_nbr
weight
payer_code
medical_specialty
```

These columns were not used for the model.

### 3. Categorical Encoding

Categorical features are converted into numerical values using **Label Encoding**.

### 4. Feature Scaling

`StandardScaler` is used to standardize the features before training the Logistic Regression model.

---

## 📈 Exploratory Data Analysis

The notebook performs several visualizations to understand the dataset:

* Readmission distribution
* Box plots of numerical features
* Patient distribution by age group
* Correlation matrix
* Readmission rate by gender

These visualizations help understand the data before building the machine learning model.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is used because the target variable is binary.

```python
lr_model = LogisticRegression(
    penalty='l2',
    max_iter=1000
)
```

### L2 Regularization

The model uses **L2 regularization**.

L2 regularization adds a penalty for large coefficients and helps reduce overfitting.

The important parameter is:

```python
penalty='l2'
```

---

## 📊 Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The model is trained using the training data and evaluated using the test data.

---

## 📏 Model Evaluation

The following evaluation methods are used:

### 1. Accuracy

Measures the overall percentage of correct predictions.

```python
accuracy_score(y_test, y_pred)
```

### 2. Precision

Precision tells us how many of the patients predicted as readmitted were actually readmitted.

```text
Precision = TP / (TP + FP)
```

### 3. Recall

Recall tells us how many of the patients who were actually readmitted were correctly identified by the model.

```text
Recall = TP / (TP + FN)
```

### 4. F1 Score

F1-score provides a balance between precision and recall.

### 5. Confusion Matrix

The confusion matrix shows:

* True Positive
* True Negative
* False Positive
* False Negative

### 6. ROC-AUC

ROC-AUC is used to measure how well the model distinguishes between patients who are and are not readmitted within 30 days.

The ROC curve is generated using the predicted probabilities.

---

## 🏥 Clinical Importance of False Positives and False Negatives

### False Negative

A **False Negative** occurs when:

> The model predicts that a patient will not be readmitted within 30 days, but the patient actually is readmitted.

This can be important clinically because a patient who may need additional monitoring or follow-up could be missed.

### False Positive

A **False Positive** occurs when:

> The model predicts that a patient will be readmitted within 30 days, but the patient is not actually readmitted.

This may lead to unnecessary monitoring or use of healthcare resources.

Therefore, both types of errors should be considered when evaluating a hospital readmission model.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Hospital-Readmission-Prediction/
│
├── diabetic_data.csv
├── Hospital_readmission_prediction.ipynb
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Open the notebook

Open:

```text
Hospital_readmission_prediction.ipynb
```

using Jupyter Notebook or JupyterLab.

### 3. Keep the dataset in the same location

Make sure:

```text
diabetic_data.csv
```

is available to the notebook.

### 4. Run the notebook

Run the cells from top to bottom to perform:

```text
Data Loading
→ Preprocessing
→ EDA
→ Encoding
→ Scaling
→ Model Training
→ Prediction
→ Evaluation
```

---

## 📌 Conclusion

This project demonstrates how **Logistic Regression with L2 regularization** can be used to predict 30-day hospital readmission among diabetic patients.

The project covers the complete basic machine learning workflow, including data preprocessing, exploratory data analysis, feature encoding, feature scaling, model training, and evaluation using classification metrics, a confusion matrix, and ROC-AUC.

The results can help demonstrate how machine learning can be applied to healthcare prediction problems while considering the different consequences of false positive and false negative predictions.

---

## 👩‍💻 Project Type

**Machine Learning | Healthcare | Binary Classification | Logistic Regression**

# Grad_Emp_Dissertation-

# Author Name : Praneeth Pulluru 
# Student Id : 34044907

# An Explainable Machine Learning and Deep Learning-Based Graduate Employability Prediction Prototype with Hyperparameter Optimisation for International Graduates

## Project Overview

This project develops an explainable machine learning and deep learning-based prototype for predicting the employment status of international graduates.

The target variable is **Employment Status**, with three possible outcomes:

- Employed
- Unemployed
- Continuing Education

The project compares multiple modelling approaches, investigates the effect of class balancing using SMOTE, applies hyperparameter optimisation using Grid Search and Random Search, and uses SHAP and LIME to improve model interpretability.

An interactive prototype was developed. The prototype allows users to enter graduate information and obtain an employment prediction together with prediction probabilities, confidence, and explainability information.

## Dataset

The project uses the publicly available **International Graduates Employment Dataset** from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/quackquackrp/international-graduates-employment-dataset

The dataset contains more than 300,000 anonymised graduate records and includes information related to:

- Education Level
- Field of Study
- Language Proficiency
- Visa Type
- University Ranking
- Internship Experience
- Work Experience
- Salary
- Employment Status

The target variable used for prediction is **Employment Status**.

> The dataset itself is not included in this repository. Please download it from the original Kaggle source before running the notebook.

## Objectives

The main objectives of this project are to:

1. Develop classification models for graduate employability prediction.
2. Compare model performance using the original dataset.
3. Investigate whether SMOTE improves classification performance.
4. Optimise model hyperparameters using Grid Search and Random Search.
5. Compare the performance of all modelling approaches.
6. Select the final model primarily using F1-Score.
7. Apply explainable AI techniques to understand model predictions.
8. Develop an interactive employability prediction prototype.

## Data Preprocessing

The preprocessing workflow included:

- Checking the dataset structure and data types.
- Handling missing values.
- Checking and removing duplicate records.
- Encoding categorical variables.
- Identifying and handling outliers.
- Applying feature engineering where required.
- Applying logarithmic transformation to the Salary variable using `log1p`.
- Creating derived features used by the modelling pipeline.
- Splitting the dataset into training and testing sets.
- Standardising feature values using `StandardScaler`.

The same preprocessing workflow is used during prediction in the prototype.

## Class Balancing with SMOTE

The class distribution of the Employment Status target was analysed before model training.

**SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data to address class imbalance.

The project compares:

- Models trained using the original training data.
- Models trained using the SMOTE-balanced training data.

This comparison was used to investigate whether balancing the training classes improved predictive performance.

## Machine Learning and Deep Learning Models

Four models were used for the final comparison:

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

### 2. XGBoost

XGBoost was used to capture nonlinear relationships and interactions between graduate-related features.

### 3. LightGBM

LightGBM was used as an efficient gradient boosting model for the multi-class classification task.

### 4. Artificial Neural Network

An Artificial Neural Network was used to investigate whether a neural-network-based model could learn more complex relationships in the graduate employability data.

Random Forest was considered during the earlier development of the project but was not included in the final four-model comparison.

## Model Development and Comparison

The models were evaluated through four stages.

### Original Dataset

I trained and evaluated Logistic Regression, XGBoost, LightGBM, and Artificial Neural Network models using the original training data.

### SMOTE Dataset

I trained and evaluated the same four models using the SMOTE-balanced training data.

### Grid Search Hyperparameter Optimisation

I applied **GridSearchCV** to the four models to identify suitable hyperparameter combinations and evaluated the resulting models.

### Random Search Hyperparameter Optimisation

I applied **RandomizedSearchCV** to the four models and evaluated the resulting models.

### Final Model Comparison

I combined the results from:

- Original Dataset
- SMOTE
- GridSearchCV
- RandomizedSearchCV

The final model was selected primarily using **F1-Score**, while the other evaluation metrics were also considered.

## Results

### Original Dataset

The four models were first trained and evaluated using the original dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8044 | 0.8040 | 0.8044 | 0.8040 | 0.9449 |
| XGBoost | 0.8198 | 0.8198 | 0.8198 | 0.8198 | 0.9528 |
| LightGBM | **0.8224** | **0.8223** | **0.8224** | **0.8215** | **0.9533** |
| Artificial Neural Network | 0.8196 | 0.8195 | 0.8196 | 0.8196 | 0.9527 |

LightGBM achieved the highest F1-Score of **0.8215** on the original dataset.

### SMOTE

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8044 | 0.8052 | 0.8044 | 0.8044 | 0.9449 |
| XGBoost | 0.8171 | 0.8190 | 0.8171 | 0.8169 | 0.9526 |
| LightGBM | 0.8172 | 0.8205 | 0.8172 | 0.8164 | 0.9531 |
| Artificial Neural Network | 0.8167 | 0.8212 | 0.8167 | 0.8152 | 0.9520 |

### GridSearchCV

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8044 | 0.8051 | 0.8044 | 0.8044 | 0.9449 |
| XGBoost | 0.8192 | 0.8198 | 0.8192 | 0.8192 | 0.9531 |
| LightGBM | 0.8204 | 0.8211 | 0.8204 | 0.8204 | 0.9532 |
| Artificial Neural Network | 0.8193 | 0.8208 | 0.8193 | 0.8191 | 0.9523 |

### RandomizedSearchCV

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8044 | 0.8051 | 0.8044 | 0.8044 | 0.9449 |
| XGBoost | 0.8198 | 0.8199 | 0.8198 | **0.8199** | 0.9530 |
| LightGBM | 0.8201 | 0.8206 | 0.8201 | 0.8201 | 0.9530 |
| Artificial Neural Network | 0.8152 | 0.8188 | 0.8152 | 0.8142 | 0.9501 |

### Final Comparison

The final comparison across all modelling approaches showed that **LightGBM using the Original Dataset** achieved the highest overall F1-Score.

| Method | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Original Data | **LightGBM** | **0.8224** | **0.8223** | **0.8224** | **0.8215** | **0.9533** |
| GridSearchCV | LightGBM | 0.8204 | 0.8211 | 0.8204 | 0.8204 | 0.9532 |
| RandomizedSearchCV | LightGBM | 0.8201 | 0.8206 | 0.8201 | 0.8201 | 0.9530 |
| RandomizedSearchCV | XGBoost | 0.8198 | 0.8199 | 0.8198 | 0.8199 | 0.9530 |
| Original Data | XGBoost | 0.8198 | 0.8198 | 0.8198 | 0.8198 | 0.9528 |
| Original Data | Artificial Neural Network | 0.8196 | 0.8195 | 0.8196 | 0.8196 | 0.9527 |
| GridSearchCV | XGBoost | 0.8192 | 0.8198 | 0.8192 | 0.8192 | 0.9531 |
| GridSearchCV | Artificial Neural Network | 0.8193 | 0.8208 | 0.8193 | 0.8191 | 0.9523 |
| SMOTE | XGBoost | 0.8171 | 0.8190 | 0.8171 | 0.8169 | 0.9526 |
| SMOTE | LightGBM | 0.8172 | 0.8205 | 0.8172 | 0.8164 | 0.9531 |
| SMOTE | Artificial Neural Network | 0.8167 | 0.8212 | 0.8167 | 0.8152 | 0.9520 |
| RandomizedSearchCV | Artificial Neural Network | 0.8152 | 0.8188 | 0.8152 | 0.8142 | 0.9501 |
| GridSearchCV | Logistic Regression | 0.8044 | 0.8051 | 0.8044 | 0.8044 | 0.9449 |
| SMOTE | Logistic Regression | 0.8044 | 0.8052 | 0.8044 | 0.8044 | 0.9449 |
| RandomizedSearchCV | Logistic Regression | 0.8044 | 0.8051 | 0.8044 | 0.8044 | 0.9449 |
| Original Data | Logistic Regression | 0.8044 | 0.8040 | 0.8044 | 0.8040 | 0.9449 |

### Best Model

**Selected Model:** LightGBM  
**Training Approach:** Original Dataset  
**Accuracy:** 0.8224  
**Precision:** 0.8223  
**Recall:** 0.8224  
**F1-Score:** 0.8215  
**ROC-AUC:** 0.9533

F1-Score was used as the primary criterion for final model selection.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

**F1-Score was used as the primary model-selection metric** because it provides a balance between precision and recall and is suitable for the multi-class employment-status prediction task.

Comparative tables and graphs were created for the different modelling stages.

## Explainable Artificial Intelligence

### SHAP

**SHAP (SHapley Additive Explanations)** was used to analyse feature contributions to model predictions.

SHAP was used to:

- Identify important features influencing employability.
- Understand the contribution of individual features.
- Explain individual model predictions.
- Improve model interpretability.



### LIME

LIME was also used to provide a local explanation for individual predictions, complementing the SHAP analysis.


## Interactive Prototype

An interactive prototype was developed using **Python** and designed to run in **Jupyter Notebook**.

The prototype includes:

### Graduate Information

Users can enter graduate-related information through interactive controls corresponding to the features used by the trained model.

### Employability Prediction

After entering the required information, the user can select **Predict Employability**.

The prototype displays:

- Predicted Employment Status
- Prediction Confidence
- Probability for each employment-status class

The possible outcomes are:

- Employed
- Unemployed
- Continuing Education

### Explainable AI Insights

The prototype provides a local explanation of the prediction using the explainability workflow implemented in the notebook.

### Prediction History

The prototype maintains a simple history of previous predictions, including prediction status and confidence.

### Interface Controls

The prototype includes:

- Predict Employability
- Clear
- Exit



### Results Folder

The `Results` folder contains outputs from the different stages of model development:

- `Original/` contains original dataset model outputs and comparison results.
- `SMOTE/` contains SMOTE-balanced model outputs and comparison results.
- `GridSearchCV/` contains GridSearchCV tuning outputs and results.
- `RandomizedSearchCV/` contains RandomizedSearchCV tuning outputs and results.
- `SHAP/` contains SHAP analysis and visualisations.
- `LIME/` contains LIME analysis and visualisations.

### Prototype Results

The `Prototype_Results` folder contains outputs and screenshots for the three employment-status scenarios:

- `Employed/`
- `Unemployed/`
- `Continuing_Education/`

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- SHAP
- LIME
- ipywidgets
- Joblib

## How to Run

1. Download the International Graduates Employment Dataset from Kaggle.
2. Open Jupyter environment.
3. Install the required Python libraries.
4. Run the notebook cells in order.
5. Complete the preprocessing, model training, tuning, explainability, and prototype stages.

The overall workflow is:

```text
Data Loading
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Original Dataset Models
      ↓
SMOTE Models
      ↓
GridSearchCV
      ↓
RandomizedSearchCV
      ↓
Final Model Comparison
      ↓
Best Model Selection
      ↓
SHAP Analysis
      ↓
LIME Analysis
      ↓
Model Saving
      ↓
Interactive Prototype
```

## Beneficiaries

The project is intended to support:

- International graduates seeking employability guidance.
- Universities working to improve graduate employability support.
- Career advisors and employability teams.
- Educational policymakers.
- Researchers working in educational data analytics, machine learning, deep learning, and explainable AI.

## Project Outcome

The project provides a complete workflow for graduate employability prediction, beginning with data preprocessing and exploratory analysis and progressing through model development, class balancing, hyperparameter optimisation, model comparison, explainability, and interactive prediction.

The final system demonstrates how explainable machine learning and deep learning can be applied to graduate employment data while providing understandable prediction results rather than relying only on a black-box classification output.

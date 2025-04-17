# Irrigation Prediction Model

## Overview

This project focuses on predicting the need for irrigation based on soil moisture sensor data and time-series analysis. The model is designed to predict whether irrigation is required for crops based on real-time sensor readings. The dataset contains time-series soil moisture data along with timestamps, and the target variable is a binary flag indicating whether irrigation is needed or not.

The goal of the model is to help automate irrigation decisions, conserving water by triggering irrigation only when necessary. The model is trained using a **Random Forest Classifier** and evaluated using various performance metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

---

## Dataset

### FAO Crop Kc Extended Data

This dataset contains information about crop water requirements at different stages of growth. However, for this specific model, the dataset is not directly used. The model relies on the **Plant Vase 1 Data** for predicting irrigation needs.

### Plant Vase 1 Data

- **Rows**: 4,117 records
- **Columns**: 12 columns with time-series data.
  - **Sensor Features**: `moisture0` to `moisture4` (soil moisture readings from different sensors), `irrigation` (target variable: binary flag for irrigation need).
  - **Time Features**: `year`, `month`, `day`, `hour`, `minute`, `second` (timestamp of each reading).
  
#### Key Features:
- **moisture0**: Most variable moisture sensor (critical in detecting irrigation needs).
- **moisture4**: Almost constant, likely indicates malfunction or a less influential sensor.
- **irrigation**: The target variable (1 = irrigation needed, 0 = no irrigation).

---

## Model Overview

### Model Type: **Random Forest Classifier**

A **Random Forest Classifier** is used to predict whether irrigation is required based on sensor readings and the changes in moisture levels over time. The Random Forest algorithm is suitable for this task because it handles non-linear relationships well and can handle a mix of numerical and categorical features.

### Features Used:
1. **Soil Moisture Sensors**: `moisture0`, `moisture1`, `moisture2`, `moisture3`, `moisture4`.
2. **Rate of Change in Moisture**: Calculated by taking the difference between consecutive readings (i.e., `moisture0_change`, `moisture1_change`, etc.).
3. **Time Features**: Derived from timestamps to capture time-based patterns like daily or hourly moisture fluctuations.

### Target Variable: 
- **irrigation**: Binary output indicating whether irrigation is needed (1) or not (0).

---

## Data Preprocessing

### 1. **Handling Missing Data**:
Missing values are imputed using the **mean** of the respective feature or removed if too many values are missing.

### 2. **Feature Engineering**:
- **Rate of Change in Moisture**: The difference between consecutive moisture readings is computed to capture the dynamics of soil moisture changes.

### 3. **Train-Test Split**:
The dataset is split into a **training set** (80%) and a **test set** (20%) to evaluate the model’s generalization ability. The split is done using the `train_test_split()` function.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Model Training and Testing

### Model Training:
- The **Random Forest Classifier** is trained on the **training data** (`X_train`, `y_train`).

```python
model.fit(X_train, y_train)
```

### Model Testing:
- The trained model is evaluated on the **test set** (`X_test`), and predictions (`y_pred`) are made.

```python
y_pred = model.predict(X_test)
```

### Model Evaluation:
- The model’s performance is evaluated using several metrics:
  - **Accuracy**: The overall percentage of correct predictions.
  - **Precision**: The proportion of true positives among all positive predictions (how many predicted irrigation events were actually needed).
  - **Recall**: The proportion of true positives among all actual irrigation events (how well the model detected irrigation needs).
  - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
  - **Confusion Matrix**: A table that summarizes the performance of the model, showing the true positives, true negatives, false positives, and false negatives.
  - **ROC Curve and AUC**: The ROC curve plots the true positive rate against the false positive rate at various thresholds. The AUC (Area Under the Curve) summarizes the model’s ability to distinguish between irrigation and no-irrigation events.

---

## Model Evaluation Results

### **Confusion Matrix**:
The confusion matrix provides insight into how well the model distinguishes between the two classes (irrigation vs. no irrigation):
- **True Positives (TP)**: Correctly predicted irrigation events.
- **True Negatives (TN)**: Correctly predicted no-irrigation events.
- **False Positives (FP)**: Incorrectly predicted irrigation when not needed.
- **False Negatives (FN)**: Incorrectly predicted no irrigation when it was needed.

### **Precision, Recall, and F1-Score**:
- **High Precision**: Indicates that when the model predicts irrigation, it is mostly correct.
- **High Recall**: Indicates that the model correctly identifies most irrigation needs.
- **F1-Score**: Balances precision and recall, providing a single metric for model performance.

### **ROC Curve and AUC**:
- **AUC** close to 1 indicates a strong ability of the model to differentiate between the two classes (irrigation needed vs. no irrigation).

---

## Visualizations

### **Confusion Matrix**:
A heatmap of the confusion matrix is used to visualize the performance of the model in terms of true positives, true negatives, false positives, and false negatives.

### **ROC Curve**:
The ROC curve plots the trade-off between the true positive rate (recall) and the false positive rate, showing the model’s discrimination ability at various thresholds.

### **Feature Importance**:
A bar plot showing the importance of each feature (e.g., moisture sensors and their rate of change) in predicting irrigation needs.

---

## Conclusion

The model demonstrates a solid understanding of when irrigation is needed based on soil moisture readings and their changes over time. It can be used to predict irrigation events in a way that conserves water and reduces the need for manual monitoring. The evaluation metrics show that the model is performing well, though further improvements can be made by fine-tuning hyperparameters, using different models, or exploring more advanced feature engineering techniques.

---

## Future Improvements

1. **Hyperparameter Tuning**: Tune the Random Forest model's hyperparameters (e.g., number of trees, max depth) to improve performance.
2. **Additional Features**: Incorporate other weather-related features like temperature, humidity, and rainfall for better predictions.
3. **Data Imbalance Handling**: Use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) if the dataset is imbalanced (e.g., more no-irrigation instances).

---

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Run the model script:
   ```bash
   python irrigation_model.py
   ```

3. The results, including metrics and visualizations, will be displayed.

---

Feel free to copy this into your README file! Let me know if you need any further adjustments or explanations.

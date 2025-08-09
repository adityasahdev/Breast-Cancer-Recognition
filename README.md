# 🩺 Breast Cancer using Logistic Regression

This project uses **Logistic Regression** to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** based on features from the **Breast Cancer Dataset** from Kaggle.

---

## 📌 Overview
- **Goal:** Predict whether a tumor is malignant or benign.
- **Algorithm:** Logistic Regression
- **Dataset:** Breast Cancer Dataset - https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset/data
- **Input Features:** 30 numerical features describing tumor characteristics (e.g., radius, texture, smoothness)
- **Output Labels:**  
  - **M** → Malignant (Cancerous)  
  - **B** → Benign (Non-cancerous)  

---

## ⚙️ How It Works
1. **Load Dataset** – Read the CSV file using `pandas`.
2. **Preprocessing**:
   - Extract relevant columns for features (columns 2–32) and labels (column 1).
   - Encode labels (`M` → 1, `B` → 0) using `LabelEncoder`.
   - Scale features using `StandardScaler` for better model performance.
3. **Train-Test Split** – Split data into 80% training and 20% testing sets.
4. **Train Model** – Fit a Logistic Regression model on the training data.
5. **Evaluate** – Calculate accuracy, classification report, and confusion matrix.

---

## 📂 Project Structure
Breast-Cancer-Recognition/
│
├── Breast_cancer_dataset.csv # Dataset file
├── ML.py # Main script
└── README.md # Project description

## 📊 Evaluation

The Logistic Regression model was evaluated on the test dataset (20% split) and achieved the following results:

- **Accuracy:** 97.37%
- **Precision:**  
  - Benign (B): 0.99  
  - Malignant (M): 0.95  
- **Recall:**  
  - Benign (B): 0.97  
  - Malignant (M): 0.98  
- **F1-Score:** Balanced performance for both classes.

**Confusion Matrix:**
|               | Predicted B | Predicted M |
|---------------|-------------|-------------|
| **Actual B**  | 70          | 2           |
| **Actual M**  | 1           | 41          |

**Insights:**
- The model performs extremely well on both classes with very few misclassifications.
- A slightly lower precision for the Malignant class indicates a small number of false positives (benign cases predicted as malignant).
- High recall for Malignant is important in medical diagnosis as it minimizes the chance of missing cancer cases.

---

## 🚀 Future Work

  
- **Feature Engineering:**
  - Analyze feature importance to identify the most influential predictors.
  - Apply dimensionality reduction techniques like PCA to improve training efficiency.

- **Performance Metrics:**
  - Visualize confusion matrix, ROC curve, and Precision-Recall curve for better interpretability.
  - Evaluate with cross-validation for more robust performance measurement.

- **Deployment:**
  - Create a user-friendly web interface using Streamlit or Flask so medical professionals can upload data and receive predictions.
  - Deploy as a cloud-based service for remote access.

- **Ethical Considerations:**
  - Ensure fairness and bias mitigation in model predictions.
  - Incorporate explainability tools (e.g., SHAP, LIME) for transparency in decision-making.

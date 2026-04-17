#  Heart Disease Prediction — Clinical Safety Optimization

##  Project Overview

This project builds a robust Machine Learning pipeline for heart disease screening. The goal was to move beyond "vanity metrics" (like high accuracy on dirty data) and create a model that prioritizes **Recall** to ensure high-risk patients are never missed.

---

##  Engineering Strategy: Why "Consensus" Over "EDA"?

Most entry-level projects rely solely on Correlation Matrices. In this project, I implemented a **Multi-Model Consensus Scoring** system for feature selection:

* **The Limitation of EDA:** EDA and Correlation Matrices primarily capture **linear** relationships. Biological data is often non-linear; a feature might look "weak" in a scatter plot but be highly predictive in a complex model.
* **The Triple-Check Approach:** I combined three different mathematical perspectives to find the **Elite 9** features:

  1. **Mutual Information:** To capture non-linear statistical dependencies.
  2. **XGBoost Importance:** To identify the best non-linear decision splits.
  3. **Logistic Regression:** To confirm linear signal strength.
* **The Result:** Only features that showed high importance across **all three** methods were selected, ensuring the model is grounded in robust biological signals rather than random noise.

---

##  Model Choice: Why Random Forest?

I evaluated several algorithms (Logistic Regression, SVM, and KNN), but **Random Forest** was selected for three core reasons:

1. **Ensemble Stability:** By averaging multiple decision trees, the model reduces overfitting on small datasets (~300 rows).
2. **Outlier Robustness:** Medical data often contains extreme values. Random Forest is less sensitive to outliers compared to distance-based models.
3. **Feature Interaction:** It captures non-linear feature interactions automatically, which is critical in medical diagnosis.

---

## Threshold Optimization: 0.40 (Final Decision)

The most critical engineering decision was adjusting the **Decision Threshold**.

* **The Default (0.50):** Treats False Negatives and False Positives equally — not suitable for healthcare.
* **Optimization Approach:** Used **Precision-Recall Curve** to analyze trade-offs.

### Threshold Comparison:

* **0.50 → Balanced model**

  * Recall: ~90%
  * Accuracy: ~91%

* **0.35 → Maximum safety**

  * Recall: **100%**
  * Precision drops significantly → too many false alarms

* **0.40 → Final Selection (Optimal Trade-off)**

  * **Recall: 98%**
  * **Precision: 89%**
  * **Accuracy: 93%**

###  Final Decision:

Threshold **0.40** was selected because it:

* Minimizes **false negatives (critical)**
* Maintains **high precision**
* Avoids excessive false alarms

 This creates a **clinically practical model**, not just a mathematically optimal one.

---

## 📊 Final Performance Metrics

| Metric                        | Score |
| :---------------------------- | :---- |
| **Accuracy**                  | 93%   |
| **Recall (Disease Class)**    | 98%   |
| **Precision (Disease Class)** | 89%   |
| **F1-Score**                  | 93%   |
| **ROC-AUC**                   | 0.98  |

---

## 🛠️ Tech Stack & Methodology

* **Python / Scikit-Learn:** Core ML pipeline
* **XGBoost:** Feature importance validation
* **Joblib:** Model serialization
* **Feature Engineering:** One-Hot Encoding
* **Cross-Validation:** Stratified K-Fold
* **Threshold Tuning:** Precision-Recall optimization

---

## Next Steps

* Deploy using FastAPI
* Build Streamlit dashboard
* Add SHAP for explainability

🩺 Heart Disease Prediction Project – Full Machine Learning Pipeline

 📌 Project Description

This project builds a **comprehensive machine learning pipeline** to predict heart disease using patient health records.
It includes **data preprocessing, visualization, feature selection, supervised and unsupervised learning, hyperparameter tuning, and evaluation**.
The aim is to demonstrate how ML can help in **early detection of heart disease**.

---

 📊 Dataset Information

* **Source**: UCI Machine Learning Repository – Heart Disease Dataset
* **Rows**: 303
* **Columns**: 14 (13 features + 1 target)
* **Target Variable**: `num` (0 = No Heart Disease, 1 = Heart Disease)
* **Missing Values**: Present in some columns (`ca`, `thal`)
* **Included Columns**: 13 main features

---

 🛠️ Tools & Technologies

* **Programming Language**: Python
* **Libraries**:

  * Data Handling → Pandas, NumPy
  * Visualization → Matplotlib, Seaborn, Plotly
  * Machine Learning → Scikit-learn



 🔄 Workflow Steps

1. **Step 1 – Data Preprocessing & Cleaning**

   * Handled missing values
   * Encoded categorical variables
   * Scaled numerical features
   * Performed exploratory data analysis (EDA)

2. **Step 2 – PCA (Visualization)**

   * Applied PCA for variance analysis
   * Plotted explained variance & scatter plots

3. **Step 3 – Feature Selection**

   * Used RFE, Random Forest importance, Chi-Square test
   * Selected best 10 predictors

4. **Step 4 – Supervised Learning Models**

   * Trained: Logistic Regression, Decision Tree, Random Forest, SVM, KNN

5. **Step 5 – Unsupervised Learning**

   * Applied K-Means & Hierarchical Clustering
   * Evaluated with Silhouette Score

6. **Step 6 – Hyperparameter Tuning**

   * GridSearchCV applied on Random Forest, KNN, SVM
   * Improved accuracy significantly


 📈 Models and Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.83     | 0.81      | 0.85   | 0.83     |
| Decision Tree       | 0.86     | 0.85      | 0.82   | 0.84     |
| Random Forest       | 0.90     | 0.89      | 0.92   | 0.90     |
| SVM                 | 0.89     | 0.87      | 0.89   | 0.88     |
| KNN                 | 0.91     | 0.90      | 0.91   | 0.90     |



 🏆 Final Model – Random Forest Classifier

* **Chosen Model**: Random Forest
* **Best Parameters (after tuning)**:

  * `n_estimators = 100`
  * `max_depth = 3`
  * `min_samples_split = 2`
  * `max_features = "sqrt"`
* **Performance (Test Set)**:

  * Accuracy: **0.90+**
  * Precision: 0.89
  * Recall: 0.92
  * F1-score: 0.90

---

 🚀 How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run training and evaluation**

* Open the Jupyter/Colab Notebooks in the `notebooks/` folder
* Run all cells step by step

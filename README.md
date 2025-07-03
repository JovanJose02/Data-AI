# Student Performance Analysis and Prediction

This project explores and models academic performance data of students to uncover insights, trends, and key predictors of success using data visualization, feature engineering, and machine learning.

---

## Project Goals

- Clean and explore the dataset
- Analyze student scores by gender, parental background, test preparation, and lunch type
- Perform regression and correlation analysis
- Build classification models to predict above-average performance
- Tune models and compare performance
- Interpret results using SHAP and feature importance

---

## Dataset

**Name:** Students Performance in Exams  
**Source:** Royce Kimmons - Synthetic Data
**Size:** ~1000 rows, 8 features (categorical + numerical)  
**Target Variable:** Above-average performance (binary)

---

## Data Cleaning & Preprocessing

- Removed **duplicates** and **missing values**
- Standardized column names (replaced spaces with underscores)
- Created new features:
  - `total_score` = math + reading + writing
  - `average_score` = total_score / 3
  - `above_avg` = binary label (1 if average_score > mean, else 0)
- Used `LabelEncoder` for categorical encoding

---

## Exploratory Data Analysis (EDA)

### 1. **Gender Distribution**
- Pie chart showed balanced gender representation

### 2. **Score Distributions**
- **Boxplots** and **Violin plots** (by gender) showed:
  - Males excel in **math**
  - Females excel in **reading** and **writing**

### 3. **Correlation Heatmap**
- Strong correlation: **Reading vs Writing**
- Moderate correlation: **Math vs others**

### 4. **Scatter Plot + Linear Regression**
- Reading vs Writing: Strong linear trend
- Equation and R² displayed with summary stats box

### 5. **Score Distributions**
- Histogram of `total_score` showed near-normal distribution with minor skew

### 6. **Parental Education and Test Prep**
- Barplots showed higher scores linked to higher **parental education**
- **Test preparation** correlated with improved scores

### 7. **Lunch Type (Socioeconomic Proxy)**
- Standard lunch associated with higher scores
- Suggests socioeconomic impact on performance

---

## Machine Learning Pipeline

### Feature Matrix `X`
Dropped: `total_score`, `average_score`, `above_avg`  
Used: Numerical + encoded categorical features

### Target Variable `y`
- Binary label: `above_avg`

### Train/Test Split
- 80% training, 20% testing

---

## Model Training & Evaluation

### Models Tested:

- **Random Forest (Tuned)**  
- **XGBoost Classifier**  
- **Logistic Regression**  
- **Support Vector Machine (SVM)**

### Hyperparameter Tuning

- **GridSearchCV** used to optimize Random Forest parameters:
  - `n_estimators`, `max_depth`, `min_samples_split`

### Cross-Validation

- 5-fold CV used for robust evaluation
- Mean accuracy: Reported for all models

### Results Summary:

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~0.79    |
| SVM                  | ~0.81    |
| XGBoost              | ~0.83    |
| Random Forest (Tuned) | **~0.86** |

---

## Model Explainability

### 🔍 SHAP (SHapley Additive exPlanations)
- Provided local + global interpretability
- Identified feature impact on predictions

### Feature Importance Plot

Top features:
- Math, Reading, and Writing Scores
- Test Preparation Course
- Parental Level of Education

---

## Final Reflections & Takeaways

> _“I noticed that there was a lot of missing data in my dataset, so I used **row deletion imputation** BECAUSE the remaining data was sufficient and complete-case analysis avoided bias from poor estimation.”_

- Used **EDA tools** (boxplot, violin, scatter, heatmap) to uncover gender and subject-based trends
- Identified **socioeconomic factors** like lunch type and parental education as strong influences
- Regression analysis (reading vs writing) revealed a **strong positive linear relationship**
- Built multiple **classification models**, with **Random Forest** performing the best
- Applied **SHAP** for transparent model interpretation

---

## Libraries Used

```python
pandas, numpy, matplotlib, seaborn, sklearn, shap, xgboost
```

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn shap scikit-learn xgboost
python student_performance_analysis.py
```

---

## Future Improvements

- Use **imbalanced learning** methods (if data is unbalanced)
- Integrate **deep learning** for more complex modeling
- Expand with **time-based** or **longitudinal student data**
- Deploy using **Dash** or **Streamlit** for real-time interactivity

---

## Author

**Jovan Jose Asker Fredy**  
*Aspiring Data Scientist & ML Engineer*

---

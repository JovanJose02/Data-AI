# Student Performance Analysis and Prediction

This project explores and models academic performance data of students to uncover insights, trends, and key predictors of success using **data visualization**, **feature engineering**, and **machine learning**.

---

## Project Goals
- Clean and explore the dataset.
- Analyze student scores by gender, parental background, test preparation, and lunch type.
- Perform regression and correlation analysis.
- Build classification models to predict above-average performance.
- Tune models and compare performance.
- Interpret results using SHAP and feature importance.

---

## Dataset
- **Name:** Students Performance in Exams
- **Source:** Royce Kimmons (Synthetic Data)
- **Size:** 1000 rows, 8 features (categorical + numerical)

---

## Data Cleaning & Preprocessing
- Removed duplicates and missing values.  
  *I used row deletion BECAUSE it preserved clean, unbiased data for modeling.*
- Standardized column names.
- Created new features:  
  - `total_score = math score + reading score + writing score`  
  - `average_score = total_score / 3`  
  - `above_avg` as binary target variable.
- Applied `LabelEncoder`.  
  *I used LabelEncoder BECAUSE machine learning models require numerical input.*

---

## Exploratory Data Analysis (EDA)

### 1. Gender Distribution  
*I used a pie chart BECAUSE it effectively visualizes proportions.*
![Gender Distribution](https://github.com/user-attachments/assets/4bd0eb2c-8a37-498f-a437-e1bee0d7f7f9)


---

### 2. Score Distributions (Boxplot & Violin)  
*Males excel in math; females in reading/writing.*  
*I used boxplot & violinplot BECAUSE they show median, spread, and distribution shape.*  
![Score Boxplot](https://github.com/user-attachments/assets/05a3fc0f-4917-4795-b708-1081e8789f0c)
![Score Violinplot](https://github.com/user-attachments/assets/d74f2a61-e57a-4113-831a-57fdb5bd88c1)


---

### 3. Correlation Heatmap  
*Strong reading-writing correlation.*  
*I used a heatmap BECAUSE it reveals feature relationships visually.*  
![Correlation Heatmap](https://github.com/user-attachments/assets/d553c411-11c4-4945-9489-385849effe6f)


---

### 4. Reading vs. Writing Scatter Plot + Linear Regression  
*Strong linear relationship (equation, R² shown).*  
*I used a scatter plot with best-fit line BECAUSE it visually demonstrates correlations with statistical backing.*  
![Reading vs Writing Scatter](https://github.com/user-attachments/assets/de2bb3ad-c8f5-466a-a70c-61d9e4c67282)


---

### 5. Total Score Distribution Histogram  
*Near-normal total score distribution.*  
*I used a histogram BECAUSE it highlights skewness and outliers.*  
![Total Score Histogram](https://github.com/user-attachments/assets/f234e211-54f9-4fb5-b19d-b78fccc7db2c)


---

### 6. Parental Education vs. Average Score (Barplot)  
*Higher parental education linked to better student scores.*  
*I used barplots BECAUSE they clarify group comparisons.*  
![Parental Education Barplot](https://github.com/user-attachments/assets/e4ca49d0-96a2-4822-b14e-d40d70567921)


---

### 7. Test Preparation vs. Average Score (Boxplot)  
*Completed test preparation associated with higher scores.*  
*I used barplots BECAUSE they visually compare group outcomes.*  
![Test Preparation Boxplot](https://github.com/user-attachments/assets/91d81f78-8a8d-46e8-8e76-2db173c95a0a)

---

### 8. Lunch Type & Gender vs. Average Score (Barplot w/ Hue)  
*Standard lunch linked to better performance (proxy for socioeconomic status).*  
*I used barplots BECAUSE they display socioeconomic disparities effectively.*  
![Lunch Type Barplot](https://github.com/user-attachments/assets/f77607e5-a8a3-4d60-b205-3b6b4c775324)

---

## Machine Learning Pipeline

- **Features (X):** Numerical + encoded categorical features (excluding `total_score`, `average_score`, `above_avg`).
- **Target (y):** Binary `above_avg`.
  
*I used binary classification BECAUSE it simplifies outcome prediction.*

- **Train/Test Split:** 80/20  
  *I used this split BECAUSE it ensures unbiased model evaluation.*

### Models Tested:
| Model                   | Accuracy |
|-------------------------|----------|
| Logistic Regression      | 1.0000  |
| Support Vector Machine   | 0.9900  |
| XGBoost Classifier       | 0.9750  |
| Random Forest (Tuned)    | 0.9750  |

*I used multiple models BECAUSE comparison reveals the best-performing approach.*

---

### Hyperparameter Tuning:
- `GridSearchCV` used for Random Forest optimization.  
  *I used GridSearchCV BECAUSE it systematically improves model accuracy.*

#### Tuned Random Forest - Classification Report:
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.96      | 0.99   | 0.98     | 99      |
| 1                | 0.99      | 0.96   | 0.97     | 101     |
| **Accuracy**     |           |        | **0.97** | **200** |
| **Macro Avg**    | 0.98      | 0.98   | 0.97     | 200     |
| **Weighted Avg** | 0.98      | 0.97   | 0.97     | 200     |

---

### 9. Confusion Matrix Plot:
*Highlights correct classifications (true positives/negatives) and errors (false positives/negatives).*

*I visualized the confusion matrix BECAUSE it provides a clear snapshot of how well the model distinguishes between above-average & below-average students, making it easier to spot misclassification patterns.*
![Confusion Matrix](https://github.com/user-attachments/assets/d92cf447-d8da-42fa-ac70-c803111171d3)


---


### Cross-Validation:
- 5-fold cross-validation for robust evaluation.
  *I used cross-validation BECAUSE it reduces overfitting and ensures consistency.*

#### Cross-Validation Accuracy Scores:
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|--------|--------|--------|--------|--------|
| 0.985  | 0.97   | 0.98   | 0.99   | 0.98   |


---

## Model Explainability

### 10. Feature Importance Plot  
*Top predictors: Math, Reading, Writing Scores, Test Preparation, Parental Education.*  
*I used feature importance BECAUSE it highlights the most influential features.*  
![Feature Importance](https://github.com/user-attachments/assets/561fb656-079e-4f0a-8c36-e8ffb340d4ad)


---

### 11. SHAP Summary Plot  
*Visualizes how features such as gender and race/ethnicity interact to influence model predictions using SHAP interaction values.*  
*I used SHAP BECAUSE it provides interpretable AI insights.*  
![SHAP Summary](https://github.com/user-attachments/assets/78d21468-7225-41b2-b3c2-5f4ab099bf6c)



## Takeaways

- Used multiple visualizations to uncover gender, subject, and socioeconomic patterns.
- Applied regression to confirm reading-writing score correlations.
- Built and compared multiple classification models (Logistic Regression, SVM, Random Forest, XGBoost).
- Tuned models systematically with GridSearchCV for better performance.
- Applied cross-validation for reliable evaluation.
- Used SHAP for transparent and explainable AI decisions.

This project demonstrates how **EDA, feature engineering, model tuning, validation, and explainability** work together to solve real-world data problems.

---

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `shap`

---

## Future Improvements
- Apply techniques for imbalanced datasets.
- Explore deep learning models.
- Deploy using **Dash** or **Streamlit** for interactive visualization.

---

## Author

**Jovan Jose Asker Fredy**  
*Aspiring Data Scientist & ML Engineer*



# Student Performance Analysis and Prediction

This project explores and models academic performance data of students to uncover insights, trends, and key predictors of success using data visualization, feature engineering, and machine learning.

## Project Goals

- Clean and explore the dataset  
- Analyze student scores by gender, parental background, test preparation, and lunch type  
- Perform regression and correlation analysis  
- Build classification models to predict above-average performance  
- Tune models and compare performance  
- Interpret results using SHAP and feature importance  

## Dataset

- **Name:** Students Performance in Exams  
- **Source:** Royce Kimmons (Synthetic Data)  
- **Size:** ~1000 rows, 8 features (categorical + numerical)  
- **Target Variable:** Above-average performance (binary)  

## Data Cleaning & Preprocessing

- Removed duplicates and missing values  
  I used row deletion BECAUSE it preserved clean, unbiased data for modeling.  
- Standardized column names  
- Created new features:  
  - `total_score = math + reading + writing`  
  - `average_score = total_score / 3`  
  - `above_avg` = binary target  
- Applied LabelEncoder  
  I used LabelEncoder BECAUSE machine learning models require numerical input.  

## Exploratory Data Analysis (EDA)

- **Gender Distribution**  
  Pie chart: Balanced gender representation  
  I used a pie chart BECAUSE it effectively visualizes proportions.  

- **Score Distributions (Boxplot & Violin)**  
  Males excel in math; females in reading/writing  
  I used boxplots/violins BECAUSE they show median, spread, and distribution shape.  

- **Correlation Heatmap**  
  Strong reading-writing correlation  
  I used a heatmap BECAUSE it reveals feature relationships visually.  

- **Scatter Plot + Linear Regression**  
  Reading vs. Writing: Strong linear relationship (equation, R² shown)  
  I used a scatter plot with best-fit line BECAUSE it visually demonstrates correlations with statistical backing.  

- **Score Distribution Histogram**  
  Near-normal total score distribution  
  I used a histogram BECAUSE it highlights skewness and outliers.  

- **Parental Education & Test Preparation**  
  Barplots reveal higher education & prep linked to better scores  
  I used barplots BECAUSE they clarify group comparisons.  

- **Lunch Type (Socioeconomic Factor)**  
  Standard lunch linked to better performance  
  I used lunch type BECAUSE it serves as a socioeconomic proxy.  

## Machine Learning Pipeline

- **Features (X):** Numerical + encoded categorical (excluding total_score, average_score, above_avg)  
- **Target (y):** Binary above_avg  
  I used binary classification BECAUSE it simplifies outcome prediction.  
- **Train/Test Split:** 80/20  
  I used this split BECAUSE it ensures unbiased model evaluation.  
- **Models Tested:**  
  - Random Forest (Tuned)  
  - XGBoost Classifier  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  I used multiple models BECAUSE comparison reveals the best-performing approach.  
- **Hyperparameter Tuning:**  
  GridSearchCV for Random Forest  
  I used GridSearchCV BECAUSE it systematically improves model accuracy.  
- **Cross-Validation:**  
  5-fold CV for robust evaluation  
  I used cross-validation BECAUSE it reduces overfitting and ensures consistency.  

## Results Summary

| Model                 | Accuracy  |
|-----------------------|-----------|
| Logistic Regression    | ~0.79     |
| SVM                   | ~0.81     |
| XGBoost               | ~0.83     |
| Random Forest (Tuned) | ~0.86     |

## Model Explainability

- **SHAP:** Explained both global and local predictions  
  I used SHAP BECAUSE it provides interpretable AI insights.  

- **Feature Importance Plot:**  
  Top predictors: Math, Reading, Writing Scores, Test Prep, Parental Education  
  I used feature importance BECAUSE it highlights the most influential features.  

## Reflections Incorporated

Each section contains:  

- "I used ___ BECAUSE ___" statements for:  
  - Visualization choices  
  - Feature selection decisions  
  - Modeling strategies  
  - Evaluation metrics  

These statements demonstrate data literacy, modeling rationale, and explainability, essential for professional data science and machine learning portfolios.

## Final Reflections & Takeaways

- “I noticed that there was a lot of missing data, so I used row deletion imputation BECAUSE the remaining data was sufficient and it avoided bias.”  
- Used multiple visualizations to uncover gender, subject, and socioeconomic patterns  
- Applied regression to confirm reading-writing score correlations  
- Built and compared multiple classification models  
- Tuned models systematically with GridSearchCV  
- Applied cross-validation for reliable evaluation  
- Used SHAP for transparent AI decisions  

This project demonstrates how EDA, feature engineering, model tuning, validation, and explainability work together to solve real-world data problems.

## Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- shap  

## Future Improvements

- Apply techniques for imbalanced datasets  
- Explore deep learning models  
- Add longitudinal or time-series data  
- Deploy using Dash or Streamlit for interactive visualization  

## Author

Jovan Jose Asker Fredy  
Aspiring Data Scientist & ML Engineer

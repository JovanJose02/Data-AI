import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")

# Load Dataset
df = pd.read_csv('Data Science Projects/StudentsPerformance.csv')

# Clean the Data
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.dropna(inplace=True)           # Remove rows with missing values

# Standardize column names for easier access
new_columns = []
for col in df.columns:
    new_col = col.replace(" ", "_")
    new_columns.append(new_col)

df.columns = new_columns

# View dataset structure
df.info()

# Summary statistics
print(df.describe(include='all'))  # Gives numeric and categorical summaries

# Analyze categorical variable distributions
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n{col} value counts:\n{df[col].value_counts()}")

# ----------------------------------------------------------------------------------------

# Pie Chart: Gender Distribution
gender_counts = df['gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'])
plt.title("Gender Distribution in Dataset")
plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle.
plt.show()

# Pie Chart Explanation:
# - Shows proportion of male vs female students in dataset
# - Helps assess balance in gender representation

# Reflection:
# I used a pie chart BECAUSE it effectively visualizes proportional distributions of categorical variables.

# ----------------------------------------------------------------------------------------

# Boxplots for score distribution by gender
plt.figure(figsize=(15, 5))
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='gender', y=subject, data=df, palette='Set2')
    plt.title(f'{subject.replace("_", " ").title()} by Gender')

# Boxplot Explanation:
# - Males tend to have slightly higher math scores
# - Females generally perform better in reading and writing
# - Shows median and variability by gender
plt.tight_layout()
plt.show()

# Reflection:
# I used gender as a key feature BECAUSE I wanted to analyze performance differences across groups.

# ----------------------------------------------------------------------------------------

# Violin plots showing score distributions by gender
plt.figure(figsize=(15, 5))
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score']):
    plt.subplot(1, 3, i+1)
    sns.violinplot(x='gender', y=subject, data=df, palette='coolwarm')
    plt.title(f'{subject.replace("_", " ").title()} Distribution by Gender')

# Violin Plot Explanation:
# - Females have denser distributions in reading and writing
# - Males have wider distributions in math
# - Shows full distribution with KDE shape
plt.tight_layout()
plt.show()

# Reflection:
# I used violin plots BECAUSE they show the full distribution shape, not just quartiles like boxplots.

# ----------------------------------------------------------------------------------------

# Correlation heatmap for subject scores
plt.figure(figsize=(8, 6))
sns.heatmap(df[['math_score', 'reading_score', 'writing_score']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Exam Scores")
plt.show()

# Correlation Heatmap Explanation:
# - Reading and writing scores are highly correlated
# - Math is moderately correlated with both
# - Indicates performance in one subject is linked to others

# Reflection:
# I used correlation analysis BECAUSE I wanted to identify relationships between the key numeric features.

# ----------------------------------------------------------------------------------------

# Enhanced Scatter Plot: Reading Score vs Writing Score with Best Fit Line and Stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data for regression
X = df[['reading_score']].values
y = df['writing_score'].values

# Fit linear regression model
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

# Calculate R²
r2 = r2_score(y, y_pred)

# Line equation
slope = reg.coef_[0]
intercept = reg.intercept_
equation = f"y = {slope:.2f}x + {intercept:.2f}"

# Plot scatter and regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='reading_score', y='writing_score', hue='gender', data=df, palette='Set1', alpha=0.6)
plt.plot(df['reading_score'], y_pred, color='black', linewidth=2, label='Best Fit Line')

# Summary statistics for box
mean_reading = df['reading_score'].mean()
mean_writing = df['writing_score'].mean()
std_reading = df['reading_score'].std()
std_writing = df['writing_score'].std()

summary_text = (
    f"Line: {equation}\n"
    f"R² = {r2:.3f}\n\n"
    f"Reading (mean±std): {mean_reading:.1f} ± {std_reading:.1f}\n"
    f"Writing (mean±std): {mean_writing:.1f} ± {std_writing:.1f}"
)

# Display stats box
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
plt.text(55, 60, summary_text, fontsize=10, bbox=props)

plt.title("Scatter Plot of Reading vs Writing Scores with Line of Best Fit")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Scatter Plot Explanation:
# - Clear positive linear relationship between reading and writing scores
# - Equation of line and R² value shown for model interpretability
# - Summary box shows descriptive stats for both variables

# Reflection:
# I used a scatter plot with a regression line BECAUSE it visually represents the strength of the relationship
# between two continuous features and provides statistical context via R² and summary metrics.

# ----------------------------------------------------------------------------------------

# Add total and average score columns for overall performance analysis
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average_score'] = df['total_score'] / 3

# Histogram of total score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['total_score'], kde=True, color='skyblue')
plt.title("Distribution of Total Scores")
plt.xlabel("Total Score")
plt.ylabel("Frequency")
plt.show()

# Histogram Explanation:
# - Most students score between 180–240
# - Slight left skew (some low outliers)
# - Distribution is roughly normal

# Reflection:
# I used histograms BECAUSE they effectively visualize data distribution and detect skewness or outliers.


# ----------------------------------------------------------------------------------------

# Barplot: Average score by parental education level
sorted_order = df.groupby('parental_level_of_education')['average_score'].mean().sort_values(ascending=False).index

plt.figure(figsize=(10, 5))
sns.barplot(
    x='parental_level_of_education',
    y='average_score',
    data=df,
    order=sorted_order,
    palette='viridis'
)
plt.xticks(rotation=45)
plt.title("Average Score by Parental Education Level (Sorted)")
plt.ylabel("Average Score")
plt.tight_layout()
plt.show()


# Barplot Explanation:
# - Students with parents holding master’s/bachelor’s degrees score higher
# - Indicates a positive influence of parental education on performance

# Reflection:
# I used parental education as a categorical feature BECAUSE I wanted to explore socio-economic influences.

# ----------------------------------------------------------------------------------------

# Boxplot: Test prep vs average score
plt.figure(figsize=(6, 4))
sns.boxplot(x='test_preparation_course', y='average_score', data=df, palette='pastel')
plt.title("Test Preparation vs Average Score")
plt.show()

# Test Prep Boxplot Explanation:
# - Students who completed test prep scored significantly higher
# - Test prep programs seem effective in boosting scores

# Reflection:
# I used test preparation course as a feature BECAUSE it is a direct intervention likely to impact scores.

# ----------------------------------------------------------------------------------------

# Barplot: Lunch type and gender vs average score
plt.figure(figsize=(10, 6))
sns.barplot(x='lunch', y='average_score', hue='gender', data=df)
plt.title("Lunch Type & Gender vs Average Score")
plt.ylabel("Average Score")
plt.show()

# Lunch Type Explanation:
# - Standard lunch students consistently outperform those with free/reduced lunch
# - Suggests lunch type (a socioeconomic proxy) affects academic success

# Reflection:
# I used lunch type BECAUSE it serves as a proxy for socio-economic status, important for performance analysis.

# ----------------------------------------------------------------------------------------

# Encode categorical columns to prepare for ML
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])


# Create binary classification target: 1 = above average, 0 = below
df_encoded['above_avg'] = (df_encoded['average_score'] > df_encoded['average_score'].mean()).astype(int)

# Train/Test split for model
X = df_encoded.drop(['total_score', 'average_score', 'above_avg'], axis=1)
y = df_encoded['above_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reflection:
# I used train/test split BECAUSE it allows unbiased evaluation of model performance on unseen data.

# ----------------------------------------------------------------------------------------

# Hyperparameter Tuning: Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Tuned Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
fig, ax = plt.subplots()
disp.plot(cmap='Blues', ax=ax, colorbar=False)
ax.grid(False)
plt.title("Confusion Matrix: Above-Average Performance Classification")
plt.show()

# Explanation:
# - GridSearchCV finds best hyperparameters across folds
# - Improves model accuracy and reduces overfitting
# - The confusion matrix visually summarizes model predictions vs actual outcomes,
#   highlighting correct classifications (true positives/negatives) and errors (false positives/negatives).

# Reflection:
# I used GridSearchCV BECAUSE it allows automated tuning of multiple parameters for better performance.
# I visualized the confusion matrix BECAUSE it provides a clear, intuitive snapshot of how well the model
# distinguishes between above-average and below-average students, making it easier to spot misclassification patterns.


# ----------------------------------------------------------------------------------------


# Cross-Validation Scores
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Reflection:
# I used cross-validation BECAUSE it provides a more reliable estimate of model generalization performance.

# ----------------------------------------------------------------------------------------


# Model Comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Random Forest (Tuned)": best_rf
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.4f}")

# Reflection:
# I used multiple models BECAUSE comparing them helps validate which algorithm performs best for this task.

# -----------------------------
# SHAP: Model Interpretability
explainer = shap.Explainer(best_rf, X_test)
shap_values = explainer(X_test)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Explanation:
# - SHAP values show how each feature influences model prediction
# - Improves interpretability and trust in ML models

# Reflection:
# I used SHAP BECAUSE it explains feature contributions and helps interpret black-box models like Random Forest.


# ----------------------------------------------------------------------------------------

# Feature importance visualization
importances = best_rf.feature_importances_  # changed from model to best_rf
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette='coolwarm')
plt.title("Feature Importances in Predicting Above-Average Performance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Feature Importance Explanation:
# - Math, reading, and writing scores are most important (as expected)
# - Test preparation and parental education also contribute

# Reflection:
# I used feature importance BECAUSE I wanted to identify which features contributed most to predictions.

# ----------------------------------------------------------------------------------------

# Final Reflection & Key Takeaways:
"""
I noticed that there was a lot of missing data in my dataset, so I used row deletion imputation
BECAUSE the remaining data was sufficient and complete-case analysis avoided bias from poor estimation.

I began by cleaning and standardizing the data, then explored it through boxplots and violin plots,
which revealed gender-based performance trends — males slightly excel in math, while females
outperform in reading and writing.

A correlation heatmap showed strong links between reading and writing scores, implying that
language-related skills are interconnected. This was visually confirmed using a scatter plot
with a regression line of best fit, which showed a strong positive linear trend. The regression
line's equation and R² value quantified this relationship, and a summary box on the plot displayed
the means and standard deviations for both reading and writing scores. Females generally occupied
the higher end of both axes, reinforcing earlier observations.

When I visualized total and average scores, I observed a near-normal distribution with a few low outliers.

Parental education and test preparation were key factors — students with higher-educated parents
or who completed test prep scored noticeably better, suggesting socioeconomic and preparatory
influences on academic success.

A pie chart showed a fairly balanced gender distribution, ensuring fair model evaluation.

I used Label Encoding BECAUSE I needed to convert categorical variables into numerical form,
and created a binary target classifying students as above or below average BECAUSE it simplifies
the classification problem.

I used train/test splitting BECAUSE it provides an unbiased performance estimate on unseen data,
and trained a Random Forest Classifier BECAUSE of its robustness and ability to handle both numeric
and categorical features.

To improve performance, I used GridSearchCV BECAUSE it systematically tuned model hyperparameters
for optimal accuracy. I also performed cross-validation BECAUSE it provides a reliable, generalized
performance estimate across folds. 

I visualized the confusion matrix BECAUSE it provides a clear and intuitive snapshot of how well the model 
distinguishes between above-average and below-average students. The confusion matrix highlighted not only 
correct classifications (true positives and true negatives) but also the types of errors (false positives 
and false negatives)—making it easier to spot potential misclassification patterns and areas for improvement.

I tested multiple models including Logistic Regression, SVM, and XGBoost BECAUSE comparing performance
ensures I select the most effective algorithm for the task.

Finally, I applied SHAP values to interpret my Random Forest model BECAUSE it explains individual feature
influences and enhances model transparency.

The feature importance plot confirmed that subject scores were the most predictive, but test prep
and parental education also contributed meaningfully.

Overall, this project demonstrated the critical role of exploratory data analysis, visualization,
model selection, hyperparameter tuning, and interpretability in understanding educational outcomes
and building trustworthy predictive tools.
"""



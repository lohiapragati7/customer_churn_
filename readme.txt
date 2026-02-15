Customer Churn Prediction — Comparative Machine Learning Study

A practical machine learning project focused on model behaviour, not just accuracy.
This study evaluates how different algorithms react to imbalanced real-world data and determines which model truly detects customers at risk of churn.

Problem Statement :

Businesses lose revenue when customers leave (churn).
The goal is not only to predict churn, but to correctly identify as many churn customers as possible while keeping false alarms reasonable.

This makes the task an imbalanced classification problem, where accuracy alone is misleading.

Models Compared :

Logistic Regression (regularized)
Decision Tree (parameter brute force)
Random Forest (GridSearchCV tuned)

Each model was trained, tuned, cross-validated, and finally tested on unseen data.

Project Workflow : 

1. Data Cleaning & preprocessing
2. Exploratory Data Analysis (pattern understanding)
3. Encoding categorical features
4. Feature Engineering and Scaling data for distance-based model
5. Stratified train-test split
6. Feature scaling (for linear model)
7. Hyperparameter tuning
8. Cross-validation selection
9. Final model evaluation
10. Behaviour interpretation using plots

This project demonstrates why a simpler model can outperform complex ensembles in practice.

Final Results --

Model	                Accuracy	Precision	Recall   	F1 Score
Logistic Regression	  0.698	           0.461	 0.805	           0.586
Decision Tree	          0.729	           0.492	 0.701	           0.578
Random Forest	          0.773	           0.590	 0.484	           0.532

Key Observations:

Random Forest maximized accuracy but missed many churn customers
Logistic Regression detected the most churn customers
Tree models biased toward majority class
Dataset shows overlapping patterns → limits complex model advantage

Conclusion:

Higher accuracy ≠ better business model.
Logistic Regression provided the most reliable predictions.

Model Behaviour Insights : 

ROC AUC ≈ 0.79 → good class separation

Precision-Recall trade-off confirms imbalance impact

Multiple moderate features influence churn instead of one dominant factor

Complex models overfit majority behaviour

--------------------------------------------------------------------------

Tech Stack : Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

--------------------------------------------------------------------------

Run notebooks in order:

eda.ipynb

2. model_evaluation.ipynb


What This Project Demonstrates?

Proper evaluation for imbalanced datasets

Model selection beyond accuracy

Bias behaviour of tree vs linear models

Practical ML thinking rather than library usage

Takeaways : 

In structured tabular data with overlapping classes,
a well-regularized linear model can generalize better than complex ensembles.
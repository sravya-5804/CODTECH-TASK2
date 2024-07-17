**Name:** Bayana Sravya Sindhu

**Company:** CODTECH IT SOLUTIONS

**ID:** CT08DS4330

**Domain:** Machine Learning

**Duration:** July to August 2024

**Mentor:** Sri Lakshmi
## Overview of the Project
![image](https://github.com/user-attachments/assets/73230591-7fd3-4575-8605-40b2ece0f905)


## Project: Developing a fraud detection model to identify fraudulent credit card transactions.
## Objective:
The Objective of this project is to Develop a fraud detection model to identify fraudulent credit card
transactions. Use techniques like anomaly detection or supervised
learning with imbalanced data.
## Key Activities:
**Loading and Preparing Data:** Loading a dataset (credit_card_transactions.csv) using pandas (pd.read_csv()) and splitting it into features (X) and target variable (y). Here, the target variable is 'Class', which typically denotes whether a transaction is legitimate or fraudulent.

**Data Splitting:** Splitting the data into training and testing sets (train_test_split() from scikit-learn). This ensures that the model is trained on a portion of the data and evaluated on unseen data to assess its generalization ability.

**Model Initialization and Training:** Initializing a Logistic Regression model (LogisticRegression() from scikit-learn) and training it using the training data (model.fit()). Logistic Regression is commonly used for binary classification tasks like identifying fraudulent transactions.

**Model Evaluation:** Making predictions (model.predict(X_test)) on the test set and evaluating the model's performance using metrics like accuracy (accuracy_score()), confusion matrix (confusion_matrix()), and classification report (classification_report()). These metrics provide insights into how well the model classifies transactions as legitimate or fraudulent0.

**Visualization:** Plotting the confusion matrix (sns.heatmap() from seaborn and matplotlib) to visually represent true positive, true negative, false positive, and false negative predictions. This visualization helps in understanding the model's performance in detecting fraudulent transactions compared to actual (ground truth) labels.
## Technologies used:
**Python:** The primary programming language for data analysis

**Pandas:** Used for data manipulation and analysis.

**matplotlib:** Employed for creating static, animated and interactive visualization.

**seaborn:** Used for making statistical graphics that are informative and attractive.

# Email-Spam-using-logistic-Regression
This project implements an email spam detector. It loads email data, preprocesses text with TF-IDF, trains a Logistic Regression model, and evaluates performance using a classification report and confusion matrix. A prediction function is provided. Visualizations show data distribution and model results.

Cell 1: Data Loading, Preprocessing, Feature Extraction, Model Training, and Evaluation

This cell contains the core logic of the spam detection model.

Import Libraries: It starts by importing necessary libraries: pandas for data manipulation, train_test_split from sklearn.model_selection for splitting data, TfidfVectorizer from sklearn.feature_extraction.text for converting text to numerical features, LogisticRegression from sklearn.linear_model for the classification model, and classification_report from sklearn.metrics for evaluating the model.
Load Dataset: It loads the spam.csv file into a pandas DataFrame. It then selects the relevant columns ('v1' and 'v2', which represent the label and text of the email) and renames them to 'label' and 'text' for clarity.
Data Preprocessing: The 'label' column is converted from text ('spam', 'ham') to numerical values (1 for spam, 0 for ham). The 'text' column is assigned to X (features) and the 'label' column to y (target variable).
Feature Extraction: A TfidfVectorizer is initialized to convert the email text into a matrix of TF-IDF features. stop_words='english' is used to remove common English words that don't contribute much to the meaning. fit_transform learns the vocabulary and IDF scores from the training data and transforms the text into TF-IDF vectors.
Train-Test Split: The data is split into training and testing sets using train_test_split. 80% of the data is used for training (X_train, y_train) and 20% for testing (X_test, y_test). random_state=42 ensures reproducibility of the split.
Model Training: A LogisticRegression model is initialized with max_iter=1000 to ensure convergence. The model is then trained on the training data using the fit method.
Evaluation: The trained model is used to predict labels on the test set (X_test). The classification_report function is used to print performance metrics (precision, recall, f1-score, support) for both classes (spam and ham).
Prediction Function: A function predict_spam is defined to take a single email text as input, transform it using the trained vectorizer, and use the trained model to predict whether it's spam or not.
Example Usage: An example email is passed to the predict_spam function to demonstrate its usage and print the prediction.
Cell 2: Visualize Spam vs. Ham Distribution

This cell creates a bar plot to show the distribution of spam and ham emails in the dataset.

Import Matplotlib: It imports the matplotlib.pyplot library for plotting.
Plot Distribution: It uses the value_counts() method on the 'label' column to count the occurrences of spam and ham. The result is then plotted as a bar chart using .plot(kind='bar').
Add Labels and Title: Titles and labels are added to the plot for better understanding. plt.xticks is used to label the x-axis ticks as 'Ham' and 'Spam'.
Show Plot: plt.show() displays the generated plot.
Cell 3: Confusion Matrix Visualization

This cell generates and displays a confusion matrix to visualize the model's performance.

Import ConfusionMatrixDisplay: It imports ConfusionMatrixDisplay from sklearn.metrics to easily plot the confusion matrix.
Create Confusion Matrix Display: ConfusionMatrixDisplay.from_estimator calculates and prepares the confusion matrix for plotting using the trained model, the test features (X_test), and the true test labels (y_test). display_labels=['Ham', 'Spam'] sets the labels for the matrix.
Add Title: A title 'Confusion Matrix' is added to the plot.
Show Plot: plt.show() displays the confusion matrix plot.

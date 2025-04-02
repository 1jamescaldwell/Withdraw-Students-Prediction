# Withdraw Predictions Project
# April 2025
# James Caldwell, Natalia Startseva
# What this script does:
    # This script loads in several files of student data, such as UVA GPA, SAT Scores, Year, etc.
    # Processes the files into a single dataframe, "all_data"
    # Splits all data into training and test datasets
    # Builds a random forest model (using the train set) to predict if a student is a higher risk of withdrawing
    # Tests the model's accuracy against the test dataset
    # Output's a CSV of the confusion matrix and a CSV of the predicted withdraw students

# Get filepath of script and get relative filepaths for log and data folders
script_path = os.getcwd()
main_folder_path = os.path.dirname(script_path) # Goes up one folder from script location
data_path = os.path.join(main_folder_path, 'data')
log_path = os.path.join(main_folder_path, 'logs\withdraw.log')

# Create Log that file ran
import sys
import datetime
dt = datetime.datetime.now()
# f = open(r"E:\UBI_data\Withdraw Predictions Project\logs\withdraw.log", "a")
f = open(log_path, "a")
f.write("Starting data Load: %s\n" % (dt))

# Data Processing
import pandas as pd
import numpy as np
import os
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

os.chdir(data_path)
withdraws = pd.read_excel('enrollment all years withdraw.xlsx')
all2324 = pd.read_excel('enrollment 23 and 24 all.xlsx')

all_data = pd.concat([withdraws, all2324])

gpa = pd.read_excel('GPA all.xlsx')
sat = pd.read_excel('SAT all.xlsx')

from collections import Counter

# Withdraw = 1, otherwise = 0
all_data['Withdraw Cancel Reason'] = all_data['Withdraw Cancel Reason'].apply(lambda x: 1 if x == 'Student Initiated' else 0)

all_data.sort_values(by=['Student System ID', 'Withdraw Cancel Reason','Age'],
                     ascending=[True, False, False], 
                     inplace=True)

all_data.drop_duplicates(subset='Student System ID', keep='first', inplace=True)

gpa.sort_values(by=['Student System ID', 'Cumulative GPA'],
                     ascending=[True, False], 
                     inplace=True)
gpa.drop_duplicates(subset='Student System ID', keep='first', inplace=True)

sat.sort_values(by=['Student System ID', 'Score'],
                     ascending=[True, False], 
                     inplace=True)
sat.drop_duplicates(subset='Student System ID', keep='first', inplace=True)

all_data = all_data.merge(gpa, on='Student System ID', how='left')
all_data = all_data.merge(sat, on='Student System ID', how='left')

# Make age, gpa, and test scores numeric
all_data['Age'] = pd.to_numeric(all_data['Age'], errors='coerce')
all_data['Cumulative GPA'] = pd.to_numeric(all_data['Cumulative GPA'], errors='coerce')
all_data['Score'] = pd.to_numeric(all_data['Score'], errors='coerce')

# Split the data into features (X) and target (y)
student_ids = all_data[['Student System ID']] # Store Student System IDs before dropping them
x = all_data.drop(['Withdraw Cancel Reason','Student System ID','Test ID','Test Component Desc'], axis=1)
y = all_data['Withdraw Cancel Reason']

x = pd.get_dummies(x, drop_first=True)
x.drop('Academic Load_No Unit Load', axis=1, inplace=True)
# Split the data into training and test sets
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(x, y, student_ids, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    class_weight={0: 1, 1: 10},
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
dt = datetime.datetime.now()
f.write("Fitting model...: %s\n" % (dt))
rf.fit(X_train, y_train)

y_probs = rf.predict_proba(X_test)[:, 1]

# Make predictions
y_probs = rf.predict_proba(X_test)[:, 1]  # Probability for class 1
threshold = 0.4
y_pred = (y_probs >= threshold).astype(int)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame to display feature names and their importance
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort by importance (highest first)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()

len(all_data['Student System ID'].unique())
import pandas as pd
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred exist in the notebook
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
cm_df.to_csv("confusion_matrix.csv", index=True)

# Identify Student IDs predicted as 1
withdrawn_students = id_test[y_pred == 1]
# Save to a CSV file if needed
withdrawn_students.to_csv("predicted_withdraws.csv", index=False)

dt = datetime.datetime.now()
f.write("Output files saved: %s\n" % (dt))
f.close()

os.chdir(script_path)
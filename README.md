# Withdraw Predictions Project  

## Overview  
This project predicts the likelihood of student withdrawal using a **Random Forest Classifier** based on historical enrollment and academic data. The script processes multiple datasets, trains a model, and outputs prediction results.  

## Authors  
- James Caldwell - Python code
- Natalia Startseva - Integration with Qlik (for automation and dashboarding)

## Data Sources  
The script loads student-related data from CSV files:  
- `enrollment all years withdraw.csv`  
- `enrollment 23 and 24 all.csv`  
- `GPA all.csv`  
- `SAT all.csv`  

## Workflow  
1. **Data Loading**:  
   - Reads multiple CSV files containing enrollment, GPA, and SAT data.  
   - Combines datasets into a single dataframe.  

2. **Preprocessing**:  
   - Converts categorical values into numeric format.  
   - Removes duplicate records, keeping the most recent relevant entry.  

3. **Feature Engineering**:  
   - Merges GPA and SAT scores with enrollment data.  
   - One-hot encodes categorical features.  
   - Drops unnecessary columns.  

4. **Model Training & Evaluation**:  
   - Splits data into training (80%) and testing (20%) sets.  
   - Trains a **Random Forest Classifier** to predict student withdrawal.  
   - Evaluates performance using a confusion matrix and key metrics (accuracy, precision, recall).  

5. **Output Generation**:  
   - Saves the confusion matrix to `confusion_matrix.csv`.  
   - Identifies students predicted to withdraw and saves them to `predicted_withdraws.csv`.  
   - Logs execution details in `withdraw.log`.  

## Output Files  
- **`confusion_matrix.csv`** – Performance summary of predictions.  
- **`predicted_withdraws.csv`** – List of students identified as high risk for withdrawal.  
- **`withdraw.log`** – Log file tracking script execution.  

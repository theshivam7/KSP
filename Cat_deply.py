import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

############################### Analysis of Locations #########################

# Read the data from the CSV file
accident_data = pd.read_csv('accident_data.csv')
valid_severity = ['Grievous Injury', 'Fatal', 'Damage Only', 'Simple Injury']

# Replace any severity not in valid_severity with 'Simple Injury'
accident_data['Severity'] = accident_data['Severity'].apply(
    lambda x: x if x in valid_severity else 'Simple Injury'
)
# Extract relevant columns for analysis
'''landmark_data = accident_data[['Accident_SubLocation', 'Severity', 'Collision_Type']]

# Plot a bar graph showing the Accident_SubLocation and the number of accidents
plt.figure(figsize=(12, 6))
sns.countplot(data=landmark_data, x='Accident_SubLocation', order=landmark_data['Accident_SubLocation'].value_counts().index)
plt.title('Number of Accidents by SubLocation')
plt.xlabel('SubLocation')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.show()

# Encode 'Accident_SubLocation' using LabelEncoder
label_encoder = LabelEncoder()
landmark_data['Accident_SubLocation_Encoded'] = label_encoder.fit_transform(landmark_data['Accident_SubLocation'])

# Plot the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=landmark_data, x='Accident_SubLocation_Encoded', y='Severity')
plt.title('Severity of Accidents by SubLocation')
plt.xlabel('SubLocation')
plt.ylabel('Severity')
plt.xticks(rotation=90)
plt.show()

# Plot a graph of SubLocation vs Collision Type
plt.figure(figsize=(12, 6))
sns.countplot(data=landmark_data, x='Accident_SubLocation', hue='Collision_Type', order=landmark_data['Accident_SubLocation'].value_counts().index)
plt.title('Collision Type by SubLocation')
plt.xlabel('SubLocation')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Collision Type')
plt.show()

############################### VICTIM DATA ANALYSIS (Point 9) ####################

victim_data = pd.read_csv('victim.csv')


victim_data = victim_data[['age', 'Sex', 'InjuryType', 'Injury_Nature']]

# Handle missing values
victim_data.dropna(inplace=True)

gender_count = victim_data['Sex'].value_counts()
gender_percentage = gender_count / gender_count.sum() * 100
age_summary = victim_data['age'].describe()


plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=victim_data, palette='Set2')
plt.title('Distribution of Victims by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(victim_data['age'], bins=20, kde=True, color='blue')
plt.title('Distribution of Victim Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

###################################### Accused Analysis #############################

accused_data = pd.read_csv('Accused_Data.csv')
accused_data.dropna(inplace=True)


accused_analysis = accused_data[['AccusedName', 'age', 'Sex']]
age_stats = accused_analysis['age'].describe()
gender_counts = accused_analysis['Sex'].value_counts()


plt.figure(figsize=(10, 6))
sns.histplot(data=accused_analysis, x='age', bins=20, kde=True)
plt.title('Distribution of Age of Accused')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=accused_analysis, x='Sex')
plt.title('Gender Distribution of Accused')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
'''
######################################### POINTS 9 - 11 ################################

accident_data = pd.read_csv('accident_data.csv', usecols=[
    'DISTRICTNAME', 'UNITNAME', 'Year', 'Accident_Spot', 'Accident_Location', 
    'Accident_SubLocation', 'Main_Cause', 'Severity', 'Road_Type', 'Weather', 
    'Accident_Description', 'Latitude', 'Longitude','Collision_Type'])

# Define the valid severity categories
valid_severity = ['Grievous Injury', 'Fatal', 'Damage Only', 'Simple Injury']

accident_data['Severity'] = accident_data['Severity'].apply(
    lambda x: x if x in valid_severity else 'Simple Injury'
)

merged_data = pd.read_csv('merged_file.csv', usecols=[
    'District_Name', 'UnitName', 'FIRNo', 'Year', 'Month', 'AccusedName', 
    'age', 'Sex_y', 'PresentAddress', 'PermanentAddress'])

merged_data.rename(columns={'District_Name': 'DISTRICTNAME', 'UnitName': 'UNITNAME'}, inplace=True)
merged_data = merged_data.drop_duplicates(subset=['FIRNo'])  # Drop duplicates based on FIRNo

# Merging
merged_df = pd.merge(accident_data, merged_data, on=['DISTRICTNAME', 'UNITNAME', 'Year'])

# NaN with Mean
for column in merged_df.columns:
    if merged_df[column].dtype in [np.float64, np.int64]:
        merged_df[column].fillna(merged_df[column].mean(), inplace=True)
    else:
        merged_df[column].fillna(merged_df[column].mode()[0], inplace=True)

# Mapping
severity_map = {'Fatal': 4, 'Grievous Injury': 3, 'Simple Injury': 2, 'Damage Only': 1}
merged_df['Severity'] = merged_df['Severity'].map(severity_map)

# Causative Factors
X = merged_df[['Road_Type', 'Weather','Accident_Spot','Accident_Location','Collision_Type','Severity','Sex_y']]
y = merged_df['Accident_SubLocation']

# variables into indicator variables
X = pd.get_dummies(X)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost Classifier
catboost_classifier = CatBoostClassifier(iterations=50, depth=10, learning_rate=0.3, random_seed=42, verbose=0)
catboost_classifier.fit(X_train, y_train)



predictions = catboost_classifier.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(f"CatBoost Classifier Model Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
#print(f"Confusion Matrix:\n{conf_matrix}")

total_samples = conf_matrix.sum()
#print(f"Total samples in confusion matrix: {total_samples}")
#print(f"y_test size: {y_test.shape[0]}")

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC Curve
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_probs = catboost_classifier.predict_proba(X_test)

# ROC curve 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(lb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(lb.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CatBoost Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

################################# Deployment Model #################################

# # Assign deployment based on Severity
severity_to_deployment = {4: 4, 3: 3, 2: 2, 1: 1}
merged_df['Deployment_Level'] = merged_df['Severity'].map(severity_to_deployment)

# Features used for deployment prediction
X_deployment = merged_df[['Road_Type', 'Weather', 'Accident_Location','Collision_Type']]

X_deployment = pd.get_dummies(X_deployment)

X_deployment_imputed = imputer.fit_transform(X_deployment)

X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X_deployment_imputed, merged_df['Deployment_Level'], test_size=0.2, random_state=42)

target_imputer = SimpleImputer(strategy='most_frequent')

y_train_dep_imputed = target_imputer.fit_transform(y_train_dep.values.reshape(-1, 1))

y_train_dep_imputed = y_train_dep_imputed.ravel()

y_test_dep_imputed = target_imputer.fit_transform(y_test_dep.values.reshape(-1, 1))

# 2D array to a 1D array
y_test_dep_imputed = y_test_dep_imputed.ravel()

# CatBoost Classifier model for deployment prediction
catboost_deployment_model = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.3, random_seed=42, verbose=0)
catboost_deployment_model.fit(X_train_dep, y_train_dep_imputed)

deployment_predictions = catboost_deployment_model.predict(X_test_dep)

deployment_accuracy = accuracy_score(y_test_dep_imputed, deployment_predictions)
print(f"Deployment Model Accuracy: {deployment_accuracy}")

# Confusion Matrix for Deployment Model
deployment_conf_matrix = confusion_matrix(y_test_dep_imputed, deployment_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(deployment_conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=severity_to_deployment.values(), yticklabels=severity_to_deployment.values())
plt.title('Confusion Matrix for Deployment Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC Curve for Deployment Model
lb_dep = LabelBinarizer()
y_test_dep_bin = lb_dep.fit_transform(y_test_dep_imputed)

y_probs_dep = catboost_deployment_model.predict_proba(X_test_dep)

# ROC curve 
fpr_dep = dict()
tpr_dep = dict()
roc_auc_dep = dict()
for i in range(len(lb_dep.classes_)):
    fpr_dep[i], tpr_dep[i], _ = roc_curve(y_test_dep_bin[:, i], y_probs_dep[:, i])
    roc_auc_dep[i] = auc(fpr_dep[i], tpr_dep[i])

# ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(lb_dep.classes_)):
    plt.plot(fpr_dep[i], tpr_dep[i], label=f'Class {i} (AUC = {roc_auc_dep[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Deployment Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot actual vs predicted values for the deployment model
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test_dep_imputed[:50])), y_test_dep_imputed[:50], marker='o', linestyle='-', color='blue', label='Actual Deployment')
plt.plot(range(len(deployment_predictions[:50])), deployment_predictions[:50], marker='x', linestyle='--', color='red', label='Predicted Deployment')
plt.title('Actual vs Predicted Deployment (Subset of Testing Data)')
plt.xlabel('Data Point')
plt.ylabel('Deployment Level')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

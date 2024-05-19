import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Load datasets
accident_data = pd.read_csv('accident_data.csv', usecols=[
    'DISTRICTNAME', 'UNITNAME', 'Year', 'Accident_Spot', 'Accident_Location', 
    'Accident_SubLocation', 'Main_Cause', 'Severity', 'Road_Type', 'Weather', 
    'Accident_Description', 'Latitude', 'Longitude','Collision_Type'])

merged_data = pd.read_csv('merged_file.csv', usecols=[
    'District_Name', 'UnitName', 'FIRNo', 'Year', 'Month', 'AccusedName', 
    'age', 'Sex_y', 'PresentAddress', 'PermanentAddress'])

# Merge columns from both datasets
merged_data.rename(columns={'District_Name': 'DISTRICTNAME', 'UnitName': 'UNITNAME'}, inplace=True)
merged_data = merged_data.drop_duplicates(subset=['FIRNo'])  # Drop duplicates based on FIRNo

# Merging
merged_df = pd.merge(accident_data, merged_data, on=['DISTRICTNAME', 'UNITNAME', 'Year'])

# Define the severity cases to keep
valid_severity_cases = ['Fatal', 'Grievous Injury', 'Simple Injury', 'Damage Only']

# Replace invalid severity cases with 'Simple Injury'
merged_df.loc[~merged_df['Severity'].isin(valid_severity_cases), 'Severity'] = 'Simple Injury'

# NaN with Mean
for column in merged_df.columns:
    if merged_df[column].dtype in [np.float64, np.int64]:
        merged_df[column].fillna(merged_df[column].mean(), inplace=True)
    else:
        merged_df[column].fillna(merged_df[column].mode()[0], inplace=True)

# Mapping severity to numeric values
severity_map = {'Fatal': 4, 'Grievous Injury': 3, 'Simple Injury': 2, 'Damage Only': 1}
merged_df['Severity'] = merged_df['Severity'].map(severity_map)

# Causative Factors
X = merged_df[['Road_Type', 'Weather', 'Accident_Spot', 'Accident_Location', 'Collision_Type', 'Severity']]
y = merged_df['Accident_SubLocation']

# Convert categorical variables into indicator variables
X = pd.get_dummies(X)

# Handle NaNs
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest for classification
rf_classifier = RandomForestClassifier(n_estimators=20,random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
predictions = rf_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(f"Random Forest Classifier Model Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Assign deployment on Severity
severity_to_deployment = {4: 4, 3: 3, 2: 2, 1: 1}
merged_df['Deployment_Level'] = merged_df['Severity'].map(severity_to_deployment)

# Features used for deployment prediction
X_deployment = merged_df[['Accident_SubLocation', 'Road_Type', 'Weather', 'Accident_Location', 'Accident_Spot', 'Collision_Type']]

# Use ColumnTransformer for encoding
column_transformer_dep = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Accident_SubLocation', 'Road_Type', 'Weather', 'Accident_Location', 'Accident_Spot', 'Collision_Type'])
    ], remainder='passthrough'
)
X_deployment_encoded = column_transformer_dep.fit_transform(X_deployment)

# Handle NaNs
X_deployment_imputed = imputer.fit_transform(X_deployment_encoded)

X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X_deployment_imputed, merged_df['Deployment_Level'], test_size=0.2, random_state=42)

target_imputer = SimpleImputer(strategy='most_frequent')
y_train_dep_imputed = target_imputer.fit_transform(y_train_dep.values.reshape(-1, 1)).ravel()
y_test_dep_imputed = target_imputer.fit_transform(y_test_dep.values.reshape(-1, 1)).ravel()

# Random Forest Classifier model for deployment prediction
rf_deployment_model = RandomForestClassifier(n_estimators=30, max_depth=20, random_state=42)
rf_deployment_model.fit(X_train_dep, y_train_dep_imputed)

# Predictions for deployment levels
deployment_predictions = rf_deployment_model.predict(X_test_dep)

# Model performance
deployment_accuracy = accuracy_score(y_test_dep_imputed, deployment_predictions)
print(f"Deployment Model Accuracy of Random Forest Classifier: {deployment_accuracy}")

# Feature Importance for deployment model
feature_importance_dep = rf_deployment_model.feature_importances_

# Generate feature names for deployment model
feature_names_dep = column_transformer_dep.get_feature_names_out()

# Group the feature importances by the original feature headings
original_features = ['Accident_SubLocation', 'Road_Type', 'Weather', 'Accident_Location', 'Accident_Spot', 'Collision_Type']
feature_group_importance = {feature: 0 for feature in original_features}

for feature_name, importance in zip(feature_names_dep, feature_importance_dep):
    for original_feature in original_features:
        if feature_name.startswith(f'encoder__{original_feature}'):
            feature_group_importance[original_feature] += importance
            break

# Plot grouped feature importance
plt.figure(figsize=(10, 6))
plt.barh(list(feature_group_importance.keys()), list(feature_group_importance.values()))
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.title('Feature Importance Plot')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test[:50])), y_test[:50], marker='o', linestyle='-', color='blue', label='Actual Severity')
plt.plot(range(len(predictions[:50])), predictions[:50], marker='x', linestyle='--', color='red', label='Predicted Severity')
plt.title('Actual vs Predicted Severity (Subset of Testing Data)')
plt.xlabel('Data Point')
plt.ylabel('Severity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Convert labels to binary format for ROC curve
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# Get probabilities for each class
y_probs = rf_classifier.predict_proba(X_test)

# ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(lb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(lb.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()



############################### ROC for Deployment Model ##################



# Convert labels to binary format for ROC curve
lb_dep = LabelBinarizer()
y_test_dep_bin = lb_dep.fit_transform(y_test_dep_imputed)

# Get probabilities for each class
y_probs_dep = rf_deployment_model.predict_proba(X_test_dep)

# ROC curve
fpr_dep = dict()
tpr_dep = dict()
roc_auc_dep = dict()
for i in range(len(lb_dep.classes_)):
    fpr_dep[i], tpr_dep[i], _ = roc_curve(y_test_dep_bin[:, i], y_probs_dep[:, i])
    roc_auc_dep[i] = auc(fpr_dep[i], tpr_dep[i])

# Plot ROC curve for each class
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


deployment_conf_matrix = confusion_matrix(y_test_dep_imputed, deployment_predictions)

# Plot the confusion matrix for deployment levels
plt.figure(figsize=(10, 7))
sns.heatmap(deployment_conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=severity_map.keys(), yticklabels=severity_map.keys())
plt.title('Confusion Matrix for Deployment Level Prediction')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

# Load your data and trained model (for demonstration, we use the existing code and simulated models)
# Dummy data for demonstration (replace with actual data loading and preprocessing)
accident_data = pd.read_csv('accident_data.csv')

# Define the valid severity categories
valid_severity = ['Grievous Injury', 'Fatal', 'Damage Only', 'Simple Injury']
accident_data['Severity'] = accident_data['Severity'].apply(lambda x: x if x in valid_severity else 'Simple Injury')

# Mapping Severity to numerical values for deployment
severity_map = {'Fatal': 4, 'Grievous Injury': 3, 'Simple Injury': 2, 'Damage Only': 1}
accident_data['Severity'] = accident_data['Severity'].map(severity_map)

# Features used for deployment prediction 
X_deployment = accident_data[['Severity','Road_Type', 'Weather', 'Accident_Location', 'Collision_Type', 'Accident_SubLocation']]
X_deployment = pd.get_dummies(X_deployment)

# Impute missing values using training data
imputer = SimpleImputer(strategy='mean')
X_deployment_imputed = imputer.fit_transform(X_deployment)

# Store the columns of the training data
deployment_columns = X_deployment.columns

# Define and train the CatBoost Classifier model for deployment prediction
# Replace the following dummy fit with loading your pre-trained model
catboost_deployment_model = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.6, random_seed=42, verbose=0)
catboost_deployment_model.fit(X_deployment_imputed, accident_data['Severity'])

# If you have a pre-trained model, load it instead
# Example: catboost_deployment_model.load_model('your_model_path.cbm')

# Streamlit app
st.title('Accident Prediction and Deployment Level Prediction')

# Sidebar for user inputs
st.sidebar.header('User Input Features')

road_type = st.sidebar.selectbox('Road Type', options=accident_data['Road_Type'].unique())
weather = st.sidebar.selectbox('Weather', options=accident_data['Weather'].unique())
accident_location = st.sidebar.selectbox('Accident Location', options=accident_data['Accident_Location'].unique())
collision_type = st.sidebar.selectbox('Collision Type', options=accident_data['Collision_Type'].unique())
accident_sublocation = st.sidebar.selectbox('Accident SubLocation', options=accident_data['Accident_SubLocation'].unique())
severity = st.sidebar.selectbox('severity', options=accident_data['Severity'].unique())
# Create a dataframe for the user input
user_input = pd.DataFrame({
    'Road_Type': [road_type],
    'Weather': [weather],
    'Accident_Location': [accident_location],
    'Collision_Type': [collision_type],
    'Accident_SubLocation': [accident_sublocation],
    'Severity' : [severity]
})

# One-hot encode the user input
user_input_encoded = pd.get_dummies(user_input)

# Ensure the user input has the same columns as the training data
user_input_encoded = user_input_encoded.reindex(columns=deployment_columns, fill_value=0)

# Impute missing values in user input
user_input_imputed = imputer.transform(user_input_encoded)

# Predict the deployment level
deployment_prediction = catboost_deployment_model.predict(user_input_imputed)

# Display the prediction
deployment_level = int(deployment_prediction[0])
st.write(f'### Predicted Deployment Level: {deployment_level}')

# Display a corresponding message based on the predicted deployment level
if deployment_level == 4:
    st.write('### More resources needed.')
elif deployment_level == 3:
    st.write('### Significant resources needed.')
elif deployment_level == 2:
    st.write('### Moderate resources needed.')
elif deployment_level == 1:
    st.write('### Fewer resources needed.')
else:
    st.write('### Invalid deployment level.')

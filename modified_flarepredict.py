
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained RandomForest model
model = joblib.load('rf_flare_predictor.pkl')

# Set up the Streamlit app
st.title('Flare Prediction App')
st.write('This application predicts the occurrence of the next flare in days based on your input.')

# Create a form for user input
st.subheader("Input Patient Data")

# Example of input fields for various features
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ('Male', 'Female'))
Disease_Type = st.selectbox('Disease type',(
    'Addisons Disease', 'Graves Disease', 'Hashimotos Thyroiditis', 'Lupus', 
    'Multiple Sclerosis', 'Myasthenia Gravis', 'Psoriasis', 'Rheumatoid Arthritis',
    "Sjögrens Syndrome", 'Type 1 Diabetes'))
Genetic_Markers = st.selectbox('Genetic Markers',(
    'AChR', 'CTLA4', 'HLA DQB1', 'HLA DR3', 'HLA DRB1', 'IL12B', 'IL7R', 'STAT4', 'TSHR'))
Biomarkers = st.selectbox('Bio Markers',(
    'Elevated Acetylcholine Receptor Antibodies', 'Elevated IL 6', 'Elevated TSI', 
    'High ANA', 'High Blood Sugar', 'High CRP', 'High IL 17', 'High RF', 
    'High TPO Antibodies', 'Low Cortisol'))
Environmental_Factors = st.multiselect('Environmental Factors', (
    'Air Pollution', 'High Altitude Living', 'High Stress Job', 'Low Iodine Intake', 
    'Rural Area', 'Smoking', 'Urban Living'))
Medical_History = st.multiselect('Medical History', (
    'Asthma', 'High Blood Pressure', 'Hypertension', 'Obesity', 'Type 2 Diabetes'))

# Convert the user input into a format that the model can process
# One-hot encoding for categorical variables
input_data = {
    'Age': age,
    'Gender': 1 if gender == 'Male' else 0,  # Assuming male=1, female=0
    f'Disease_Type_{Disease_Type}': 1,
    f'Genetic_Markers_{Genetic_Markers}': 1,
    f'Biomarkers_{Biomarkers}': 1
}

# Handle multiselect fields (Environmental Factors and Medical History)
for factor in [
    'Air Pollution', 'High Altitude Living', 'High Stress Job', 'Low Iodine Intake', 
    'Rural Area', 'Smoking', 'Urban Living']:
    input_data[f'Environmental_Factors_{factor}'] = 1 if factor in Environmental_Factors else 0

for history in ['Asthma', 'High Blood Pressure', 'Hypertension', 'Obesity', 'Type 2 Diabetes']:
    input_data[f'Medical_History_{history}'] = 1 if history in Medical_History else 0

# Ensure all other necessary columns are included (set to 0 if not in input)
all_columns = {col: 0 for col in model.feature_names_in_}  # Get all feature names from the model
all_columns.update(input_data)  # Override with actual input values

# Create a DataFrame for model prediction
input_df = pd.DataFrame([all_columns])

# Perform prediction
prediction = model.predict(input_df)
st.subheader('Prediction')
st.write(f'The predicted time until next flare is {prediction[0]} days.')

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
Disease_Type =st.selectbox('Disease type',('Addisons Disease','Graves Disease','Hashimotos Thyroiditis','Lupus','Multiple Sclerosis','Myasthenia Gravis','Psoriasis','Rheumatoid Arthritis','SjÃ¶grens Syndrome','Type 1 Diabetes',))
Genetic_Markers = st.selectbox('Genetic Markers',('AChR','CTLA4','HLA DQB1','HLA DR3','HLA DRB1','IL12B','IL7R','STAT4','TSHR',))
Biomarkers = st.selectbox('Bio Markers',('Elevated Acetylcholine Receptor Antibodies','Elevated IL 6','Elevated TSI','High ANAHigh Blood Sugar','High CRP','High IL 17','High RF','High TPO Antibodies','Low Cortisol'))
Environmental_Factors = st.selectbox('Environmental Factors',('Air Pollution','High Altitude Living','High Stress Job','Low Iodine Intake','Rural Area','Smoking','Urban Living',''))
Medical_History = st.selectbox('Medical History',('Asthma','High Blood Pressure','Hypertension','Obesity','Type 2 Diabetes',''))


# Add more features as per your dataset
# Make sure these features match the features used in your model training

# When the user clicks the 'Predict' button
if st.button('Predict Next Flare'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'gender ': [gender ],
        'Disease_Type': [Disease_Type],
        'Genetic_Markers': [Genetic_Markers],
        'Biomarkers': [Biomarkers],
        'Environmental_Factors': [Environmental_Factors],
        'Medical_History': [Medical_History]
        # Add all necessary features here
    })
    
    # Display the input data
    st.write("Patient Data:")
    st.write(input_data)
    
    # Make predictions using the model
    prediction = model.predict(input_data)
    
    # Display the prediction result
    st.subheader(f"The flare is predicted to occur in {prediction[0]} days.")

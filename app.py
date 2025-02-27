import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb  
# Placeholder for user credentials (replace with secure storage)
valid_credentials = {"passwordapp": "passwordapp"}

def check_credentials(email, password):
    """Checks if the provided email and password are valid."""
    return email in valid_credentials and valid_credentials[email] == password
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_credentials(email, password):
            st.session_state.authenticated = True
            # st.experimental_rerun()  # Rerun to show the main app
        else:
            st.error("Invalid credentials")
else:
    try:
        loaded_model = pickle.load(open('age_xgboost_model.pkl', 'rb'))  # Ensure the model file is in the same directory or provide the correct path.
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'xgboost_model.pkl' is in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()
    
    nhanes_cardiovascular_variables ={'Gender': 'Gender (1 = Male, 2 = Female), [1, 2]',
     'BPXPLS': 'Pulse Rate (beats per minute), [56, 96]',
     'BPXSY1': 'Systolic Blood Pressure 1st Reading (mmHg), [94, 152]',
     'BPXSY2': 'Systolic Blood Pressure 2nd Reading (mmHg), [94, 152]',
     'BPXSY3': 'Systolic Blood Pressure 3rd Reading (mmHg), [94, 152]',
     'BPXDI1': 'Diastolic Blood Pressure 1st Reading (mmHg), [42, 87]',
     'BPXDI2': 'Diastolic Blood Pressure 2nd Reading (mmHg), [42, 86]',
     'BPXDI3': 'Diastolic Blood Pressure 3rd Reading (mmHg), [40, 86]',
     'BMXWAIST': 'Waist Circumference (cm), [62, 126]',
     'BMXBMI': 'Body Mass Index (kg/m^2), [17, 41]',
     'BMXARML': 'Upper Arm Length (cm), [29, 41]'}
    var_median={'Gender': 2.0,
     'BPXPLS': 74.0,
     'BPXSY1': 116.0,
     'BPXSY2': 116.0,
     'BPXSY3': 114.0,
     'BPXDI1': 66.0,
     'BPXDI2': 66.0,
     'BPXDI3': 68.0,
     'BMXWAIST': 92.0,
     'BMXBMI': 26.2,
     'BMXARML': 36.5}
    st.title("XGB Prediction")
    
    user_inputs = {}
    for var in nhanes_cardiovascular_variables:
        if var == 'Gender':
            user_inputs[var] = st.selectbox(f"Select {nhanes_cardiovascular_variables[var]}", [1, 2])
        else:
            user_inputs[var] = st.number_input(f"Enter {nhanes_cardiovascular_variables[var]}", value=var_median[var])
    
    if st.button("Predict"):
        input_data = pd.DataFrame([list(user_inputs.values())], columns=nhanes_cardiovascular_variables.keys())
    
        try:
            prediction = loaded_model.predict(input_data)
            st.success(f"Predicted Age: {prediction[0]:.4f}")  # Display prediction with 4 decimal places
    
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
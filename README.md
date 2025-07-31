# XGBoost Age Prediction App

This Streamlit application allows you to predict age based on various physiological measurements using an XGBoost model. It also provides SHAP (SHapley Additive exPlanations) visualizations to explain the model's predictions.

## Features

User Authentication: Simple password-based authentication to access the prediction interface.

Age Prediction: Input various health metrics to get an age prediction from a pre-trained XGBoost model.

SHAP Explanations: Visualize how each input feature contributes to the predicted age, providing transparency into the model's decision-making.


```
pip install -r requirements.txt

streamlit run app.py

```


## Usage

### Login: 
Upon launching the app, you'll be prompted for an email and password.

Email: passwordapp

Password: passwordapp

### Enter Data: 
After successful login, you will see input fields for various physiological measurements. Fill in the values for:

Gender (1 = Male, 2 = Female)

Pulse Rate (beats per minute)

Systolic Blood Pressure (1st, 2nd, and 3rd readings in mmHg)

Diastolic Blood Pressure (1st, 2nd, and 3rd readings in mmHg)

Waist Circumference (cm)

Body Mass Index (kg/m^2)

Upper Arm Length (cm)

### Get Prediction: 
Click the "Predict" button to see the predicted age and a SHAP waterfall plot explaining the prediction.



## Repository Structure
app.py: The main Streamlit application script.

age_xgboost_model.pkl: The pre-trained XGBoost model for age prediction.

requirements.txt: Lists all the Python dependencies required to run the application.

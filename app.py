import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('reg_model.h5')

with open('my_reg_scaler.pkl','rb') as file:
  scaler = pickle.load(file)

with open('reg_le.pkl','rb') as file:
  le = pickle.load(file)
  
with open('reg_ohe_encoder.pkl','rb') as file:
  ohe = pickle.load(file)
  
  
st.title('Bank Customer Salary Predictor')

geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Is Customer Exited', [0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])



if st.button('Predict'):
 input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [le.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited':[exited]
    }
)

# One-hot encode the geography
 ohe_geo = ohe.transform([[geography]])
 geo_encoded_df = pd.DataFrame(ohe_geo, columns=ohe.get_feature_names_out())

# Combine with input data
 input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure input_data columns match the scaler
 input_data_scaled = scaler.transform(input_data)

# Make prediction
 predict = model.predict(input_data_scaled)[0][0]

# Display result
 st.write(f'The Estimated Salary of Customer is : ₹ {predict}')

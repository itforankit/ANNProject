import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
#from keras.models import load_model

## Load the trained model, scaler pickle and onehot pickle
model=tf.keras.models.load_model('reg_model.h5')

## load the encoder and scaler

with open('label_encoder_gender_reg.pkl','rb') as file:
    label_ennoder_gender =pickle.load(file)

with open('Onehot_encoder_geo_reg.pkl','rb') as file:
    onhot_ennoder_geo =pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler =pickle.load(file)

## streamlit app

st.title('Customer Chur Prediction')

#user inputs

geography=st.selectbox('Geography',onhot_ennoder_geo.categories_[0])
gender=st.selectbox('Gender',label_ennoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure =st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_ennoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited': [exited]
})

#One-hot encode 'Geography'
geo_encoded = onhot_ennoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onhot_ennoder_geo.get_feature_names_out(['Geography']))

#combined with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scaling
input_data_scaled=scaler.transform(input_data)

#Prediction Churn
prediction=model.predict(input_data_scaled)
prediction_salary=prediction[0][0]

st.write(f"Predicted Estimated Salary: {prediction_salary:.2f}")


    # to run go on terminal streamlit run streamlit_reg.py
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.regression import *
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
from pycaret.datasets import get_data

st.title("pycaret model")
filename = st.text_input("write your csv file name, (diabetes.csv) for example")
data=pd.read_csv('diabetes.csv')
data= pd.read_csv(filename)
st.write(data.describe())

fillmethod= st.selectbox("do you wish do fill the missing values with zeros or drop their raws ",['fill', 'drop'])
if fillmethod == 'fill':
    data=data.fillna(0)
if fillmethod == 'drop':
    data = data.dropna()

st.write(data.head())
column = st.selectbox(
    'which column you wish to be predicted',
    (data.columns))
data.drop_duplicates(inplace=True)

coulmntype=data[column].dtype
print(data.dtypes)
if coulmntype == 'float64':
    st.write('your data will do regression analysis')
    s = setup(data, target = column, session_id = 123)
    best = compare_models()
    st.write('the best anaysis way is:' ,str(best))
    predict_model(best)
    predictions = predict_model(best, data=data)
    st.write(predictions)

if coulmntype == 'object':
    st.write('your data will do classification analysis')
    data[column]= label_encoder.fit_transform(data[column]) 
    s = setup(data, target = column, session_id = 123)
    best = compare_models()
    st.write('the best classification model is' , best)
    #plot_model(best, plot = 'auc')
    st.write('the predictions are:')
    st.write(predict_model(best,data=data))


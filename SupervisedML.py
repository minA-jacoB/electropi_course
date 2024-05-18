import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.regression import *
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
from pycaret.datasets import get_data

#####importing data######

st.title("pycaret model")
filename = st.text_input("write your csv file name, (diabetes.csv) for example")
#data=pd.read_csv('diabetes.csv')
if filename:
    try:
        data= pd.read_csv(filename)
        #####A simple EDA#####
        st.write(data.describe())

        #####droping duplicates#####
        data.drop_duplicates(inplace=True)
        st.write('.......................................')

        ######asking the user about the filling method#####

        fillmethod= st.selectbox("do you wish do fill the missing values with zeros or drop their raws ",['fill', 'drop'])
        if fillmethod == 'fill':
            data=data.fillna(0)
        if fillmethod == 'drop':
            data = data.dropna()
        st.write('.......................................')

        #####asking the user about the columns he want to drop#####
        drop_columns = st.multiselect(
            'select the columns you want to drop completely, if you need',
            list(data.columns))
        data.drop(list(drop_columns), axis=1, inplace =True)
        st.write(data.head(2))
        st.write('.......................................')

        ######asking the user about the columns he want to do EDA###
        column_to_eda = st.selectbox(
            'which column you wish to do data analysis to it',
            list(data.columns))

        st.write(data[str(column_to_eda)].describe())

        column_type = data[str(column_to_eda)].dtype
        st.write('the data type of the column is ', str(column_type))

        column_count= data[str(column_to_eda)].count()
        st.write('the length of the column(number of rows is equal to) ', str(column_count))

        column_unique=data[str(column_to_eda)].nunique()
        st.write('the number of unique values in this row is equal to ',str(column_unique) )
        
        st.write('.......................................')
        ########encoding########
        encoding_method= st.selectbox("how do you want to encode the data ,where in hot encoding will encode the whole df, but label encoding will encide only the chosen columns  ",['Hot encoding', 'label encoding'])
        if encoding_method == 'Hot encoding':
            data = pd.get_dummies(data)
        elif encoding_method == 'label encoding':
            columns_encoded = st.multiselect(
            'select the columns you want to encode',
            list(data.columns))
            columns_encoded = list(columns_encoded)
            for i in range(len(columns_encoded)):
                data[columns_encoded[i]]= label_encoder.fit_transform(data[columns_encoded[i]])
        st.write(data.head(3))
        


        #######
        #####asking the user about the column to be predicted###
        target_column = st.selectbox('Select the column to be predicted:', list(data.columns))
        if target_column:
            column_type = data[target_column].dtype
            if np.issubdtype(column_type, np.number):
                st.write('Your data will undergo regression analysis.')
                setup(data, target=target_column, session_id=123)
                best_model = compare_models()
                st.write('The best regression model is:', best_model)
                predictions = predict_model(best_model, data=data)
                st.write('Predictions:')
                st.write(predictions)
            else:
                st.write('Your data will undergo classification analysis.')
                if column_type == 'object':
                    data[target_column] = label_encoder.fit_transform(data[target_column])
                setup(data, target=target_column, session_id=123)
                best_model = compare_models()
                st.write('The best classification model is:', best_model)
                predictions = predict_model(best_model, data=data)
                st.write('Predictions:')
                st.write(predictions)

    except Exception as e:
        st.error(f"An error occurred: {e}")
 

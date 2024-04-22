import streamlit as st
import pandas as pd


#from **pycaret.regression** import *****
df=()

filename = st.text_area("write your csv or excel file name")
file_type = filename.split(".")[-1]
if file_type == "csv":
    df= pd.read_csv(filename)
elif file_type == "xlsx":
    df= pd.read_excel(filename)

st.write(df)

st.write("kdlewer")

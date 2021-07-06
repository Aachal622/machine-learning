
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('kmeansclusterpractical.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('clustering dataset1.csv')
# Extracting independent variable:
X = dataset.iloc[:,1:8].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  predict= model.predict(sc.transform([[Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("cluster number", predict)
  
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Disease Prediction using K-Means Algorithm")
    
    #UserID = st.text_input("UserID","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    #Gender = st.number_input('Insert Gender Male:1 Female:0')
    
    Glucose = st.number_input("Insert of glucose",100,1000)
    BP = st.number_input("Insert BP",0,50)
    SkinThickness = st.number_input("Insert SkinThickness ",0,60)
    Insulin = st.number_input("Insert Insulin ",0,1000)
    BMI = st.number_input("Insert BMI ",0,80)
    PedigreeFunction = st.number_input("Insert PedigreeFunction",0,10)
    Age = st.number_input('Insert a Age',18,60)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Aachal Kala")
      st.subheader("Second mid term practical")
      st.subheader("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()

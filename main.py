import streamlit as st
import numpy as np
import pandas as pd


# ชื่อหน้าเว็บ
st.title("Machine Learning ทำนายโรคเบาหวาน")

# แสดง DataFrame (ตาราง)
import pandas as pd

df = pd.read_csv("diabetes.csv")

st.write("ตารางตัวอย่าง(โรคเบาหวาน):")
st.dataframe(df.head())

#input
st.title("ใส่ค่าเพื่อทำนาย")
Pregnancies = st.number_input("Pregnancies:")
Glucose = st.number_input("Glucose:")
BloodPressure = st.number_input("BloodPressure:")
SkinThickness = st.number_input("SkinThickness:")
Insulin = st.number_input("Insulin:")
Bmi = st.number_input("BMI:")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction:")
Age = st.number_input("Age:")

#model ml
from joblib import load

model_knn = load("model_knn.joblib")
model_svm = load("model_svm.joblib")


knn = model_knn["model"]
svm = model_svm["model"]
if st.button("กดเพื่อแสดงผล"):
    x_pred = np.array([[Pregnancies,
                        Glucose,
                        BloodPressure,
                        SkinThickness,
                        Insulin,
                        Bmi,
                        DiabetesPedigreeFunction,
                        Age]])
    
    y_pred_knn = knn.predict(x_pred)
    y_pred_svm = svm.predict(x_pred)

    st.write("ผลการทำนาย knn :", y_pred_knn[0])
    st.write("ผลการทำนาย svm :", y_pred_svm[0])

#accuracy model
acc_knn = model_knn["accuracy"]
st.write("acc_knn :",acc_knn)

acc_svm = model_svm["accuracy"]
st.write("acc_svm :",acc_svm)
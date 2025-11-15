import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# ================== แสดงรูปเส้นแบ่งของ SVM ==================

st.title("เส้นแบ่งการตัดสินใจของ SVM")

# ฟังก์ชันวาดเส้นแบ่ง (ใช้ Glucose vs BMI)
def plot_svm_boundary(model, df, feat1="Glucose", feat2="BMI"):
    # y = outcome
    y = df["Outcome"].values

    # ชื่อ feature ทั้งหมด (ไม่รวม Outcome)
    feature_names = list(df.drop("Outcome", axis=1).columns)
    i1 = feature_names.index(feat1)
    i2 = feature_names.index(feat2)

    # ช่วงค่าที่จะเอามาวาดแกน X / Y
    x_min, x_max = df[feat1].min() - 1, df[feat1].max() + 1
    y_min, y_max = df[feat2].min() - 1, df[feat2].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # เตรียม X_grid ที่มีทุก feature
    X_mean = df.drop("Outcome", axis=1).mean().values  # ค่าเฉลี่ยของแต่ละคอลัมน์
    X_grid = np.tile(X_mean, (xx.size, 1))

    # แทนค่าของสอง feature ที่เราอยากดูด้วยค่าบน grid
    X_grid[:, i1] = xx.ravel()
    X_grid[:, i2] = yy.ravel()

    # ให้ SVM ทำนาย class ของทุกจุดใน grid
    Z = model.predict(X_grid)
    Z = Z.reshape(xx.shape)

    # วาดรูป
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(df[feat1], df[feat2], c=y, edgecolor="k")

    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title("(Other features fixed at mean)")

    fig.tight_layout()
    return fig


# ปุ่มให้กดแสดงรูป
if st.button("แสดงเส้นแบ่ง SVM (Glucose vs BMI)"):
    fig = plot_svm_boundary(svm, df)   # ใช้โมเดล svm ที่โหลดมาอยู่แล้ว
    st.pyplot(fig)

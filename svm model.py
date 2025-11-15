from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from scipy.stats import expon

#prepare data
df = pd.read_csv("diabetes.csv")
# x is data y is outcome
x = df.drop("Outcome",axis=1).values # axis=1 อ้างอิง colume 
y = df["Outcome"].values

#model
model = SVC(kernel="rbf",class_weight="balanced")

#spit data 80 20
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#train
# random grid
param_dist = {
    "C": expon(scale=10),        # สุ่ม C แถว ๆ 0–30
    "gamma": expon(scale=0.001)  # สุ่ม gamma แถว ๆ 0–0.003
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=1000,        # ยิ่งมากยิ่งละเอียด แต่ช้าขึ้น
    cv=5,
    random_state=0,
    n_jobs=-1
)

random_search.fit(x_train, y_train)

model_best = random_search.best_estimator_

y_pred = model_best.predict(x_test)

print(accuracy_score(y_pred,y_test)*100)
acc = accuracy_score(y_pred,y_test)*100

#save model
import joblib

saved_obj = {
    "model": model_best,
    "accuracy": acc
}
joblib.dump(saved_obj, "model_svm.joblib") 
print("saved model")

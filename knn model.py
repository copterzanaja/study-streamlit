#ทำนายโรคเบาหวาน
#KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

#prepare data
df = pd.read_csv("diabetes.csv")
# x is data y is outcome
x = df.drop("Outcome",axis=1).values # axis=1 อ้างอิง colume 
y = df["Outcome"].values

#spit data 80 20
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#find k to model
k_neighbors = np.arange(1,30) # 1,2,3,4,5,6,7,8
training_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

#find best k value

for i,k in enumerate(k_neighbors):
    #model
    knn = KNeighborsClassifier(n_neighbors=k)
    #train
    knn.fit(x_train,y_train)
    #score
    training_score[i] = knn.score(x_train,y_train)
    test_score[i] = knn.score(x_test,y_test)

plt.title("compare k in model")
plt.plot(k_neighbors,test_score,label="test score")
plt.plot(k_neighbors,training_score,label="train score")
plt.xlabel("k number")
plt.ylabel("test score")
plt.show()

#Model
knn = KNeighborsClassifier(n_neighbors=8)

#train
knn.fit(x_train,y_train)

#predict
y_pred = knn.predict(x_test)

print("ผลการพยากรณ์ :",y_pred)

#วัดผล
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test,y_pred,rownames=["ค่าจริง"],colnames=["ทำนาย"],margins=True))
print(accuracy_score(y_test,y_pred)*100)
acc = accuracy_score(y_test,y_pred)*100

#save model
import joblib 

saved_obj = {
    "model": knn,
    "accuracy": acc
}
joblib.dump(saved_obj, "model_knn.joblib") 
print("saved model")




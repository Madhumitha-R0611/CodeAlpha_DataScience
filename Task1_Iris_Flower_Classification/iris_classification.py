import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("Iris.csv")
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
sns.pairplot(df,hue="Species")
plt.show()

x=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=df["Species"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("\nTraining data size:")
print(x_train.shape)
print("\nTesting data size:")
print(x_test.shape)

model = LogisticRegression()
model.fit(x_train,y_train)
print("\nModel training completed successfully!")

y_pred=model.predict(x_test)
print("\nPredicted values:")
print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("\nModel Accuracy:")
print(accuracy)
print("\nModel Accuracy Percentage:")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",xticklabels=model.classes_,yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Iris Flower Classification - Confusion Matrix")
plt.show()

new_flower=[[5.1,3.5,1.4,0.2]]
prediction=model.predict(new_flower)
print("New Flower Prediction:")
print(prediction[0])
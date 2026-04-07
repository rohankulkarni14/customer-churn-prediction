import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv("churn_data.csv")

# Drop unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Convert target
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Handle categorical data
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))

import matplotlib.pyplot as plt

# Feature importance (basic)
importance = model.coef_[0]
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()

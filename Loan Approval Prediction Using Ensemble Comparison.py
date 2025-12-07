import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, random_state=42)
feature_names = ['Credit_Score', 'Income', 'Loan_Amount', 'Term', 'History', 'Assets', 'Debt', 'Emp_Length', 'Age', 'Education']
df = pd.DataFrame(X, columns=feature_names)
df['Approved'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

print("--- Model Accuracy Results ---")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc * 100:.2f}%")

best_model_name = max(results, key=results.get)
best_model_acc = results[best_model_name]

print(f"\nBest Model: {best_model_name} with {best_model_acc * 100:.2f}% accuracy")

plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green', 'purple'])
plt.ylabel('Accuracy')
plt.title('Loan Approval Model Comparison')
plt.ylim(0.5, 1.0) 
plt.show()

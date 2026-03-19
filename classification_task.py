import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df.drop(columns=['Cabin'], inplace=True)

# Target: Survived (binary 0/1) — Classification problem
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build logistic regression classifier
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate using accuracy
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print("Classification Task — Predicting Survival")
print(f"  Features    : {features}")
print(f"  Target      : Survived (0 = No, 1 = Yes)")
print(f"  Train size  : {len(X_train)}")
print(f"  Test size   : {len(X_test)}")
print(f"  Accuracy    : {acc * 100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
ax.set_yticklabels(['Actual 0', 'Actual 1'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_title(f'Confusion Matrix  (Accuracy = {acc*100:.1f}%)', fontweight='bold')
plt.tight_layout()
plt.savefig('classification_confusion_matrix.png', dpi=150)
plt.show()

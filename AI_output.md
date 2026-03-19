### Types of Machine Learning, Regression, and Classification with Pandas

#### 1. Types of Machine Learning

**Supervised Learning** — The model learns from labelled data (input–output pairs).
- Regression: predict a continuous value (e.g., house price)
- Classification: predict a discrete label (e.g., spam / not spam)

**Unsupervised Learning** — The model finds patterns in unlabelled data.
- Clustering (K-Means), Dimensionality Reduction (PCA)

**Reinforcement Learning** — An agent learns by interacting with an environment and receiving rewards or penalties.
- Examples: game-playing AI (AlphaGo), robot navigation

> *Source: Scikit-learn documentation — scikit-learn.org*

---

#### 2. Regression with Pandas + Scikit-learn

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Regression: predict Fare (continuous target)
X = df[['Pclass', 'Sex', 'Age']]
y = df['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Regression MSE  : {mse:.2f}")
print(f"Regression RMSE : {np.sqrt(mse):.2f}")
```

---

#### 3. Classification with Pandas + Scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Classification: predict Survived (binary target)
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc * 100:.2f}%")
```

---

#### 4. Pandas — Key Operations for ML Prep

```python
# Load and inspect
df = pd.read_csv('titanic.csv')
print(df.isnull().sum())       # check missing values

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df.drop(columns=['Cabin'], inplace=True)

# Filter: survivors only
survivors = df[df['Survived'] == 1]
print(survivors['Fare'].mean())  # avg fare among survivors

# Correlation
print(df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].corr())
```

### Key Differences

| Concept | Regression | Classification |
| :--- | :--- | :--- |
| **Target** | Continuous float | Discrete label |
| **Output** | e.g., £84.50 | e.g., 0 or 1 |
| **Metric** | MSE, RMSE, R² | Accuracy, F1, AUC |
| **Algorithm** | Linear Regression | Logistic Regression |

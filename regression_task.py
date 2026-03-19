import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Target: Fare (continuous) — Regression problem
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate using MSE
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Regression Task — Predicting Fare")
print(f"  Features     : {features}")
print(f"  Target       : Fare (continuous)")
print(f"  Train size   : {len(X_train)}")
print(f"  Test size    : {len(X_test)}")
print(f"  MSE          : {mse:.4f}")
print(f"  RMSE         : {rmse:.4f}")
print(f"\n  Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"    {feat:10s} : {coef:.4f}")
print(f"  Intercept    : {model.intercept_:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='white', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=1.5, label='Perfect prediction')
plt.title('Regression — Actual vs Predicted Fare', fontweight='bold')
plt.xlabel('Actual Fare (£)')
plt.ylabel('Predicted Fare (£)')
plt.legend()
plt.tight_layout()
plt.savefig('regression_actual_vs_pred.png', dpi=150)
plt.show()

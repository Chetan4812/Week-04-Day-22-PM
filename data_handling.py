import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Original shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())

# ── Handle Missing Values ─────────────────────────────────────────────────────

# Age: fill with median (robust to outliers)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: fill with mode (most common port)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabin: too many missing values (77%) — drop the column entirely
df.drop(columns=['Cabin'], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# ── Select Relevant Features ──────────────────────────────────────────────────

# Encode Sex as binary numeric (required for ML models)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Select features useful for both regression (Fare) and classification (Survived)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
df_clean = df[features]

print(f"\nCleaned dataset shape: {df_clean.shape}")
print("\nFirst 5 rows of cleaned dataset:")
print(df_clean.head())

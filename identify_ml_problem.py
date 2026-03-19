import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset: Titanic (891 rows × 12 columns)")
print(df.dtypes)
print()

# ── ML Problem Type Identification ───────────────────────────────────────────

# 1. Supervised vs Unsupervised vs Reinforcement
# The dataset contains labelled examples: each row has a known outcome (Survived = 0 or 1)
# → Supervised Learning

# 2. Regression vs Classification
# Target column: 'Survived' → values are 0 or 1 (discrete binary labels)
# → Classification

# Target column: 'Fare' → values are continuous floats (e.g., 7.25, 71.83, 512.33)
# → Regression

print("── ML Problem Type ──────────────────────────────────────────")
print("Learning Type : Supervised Learning")
print("  Justification: Each row has a labelled outcome.")
print("  'Survived' column provides ground truth for every passenger.\n")

print("If target = 'Survived' (0/1)  → Classification")
print("  Justification: Output is a discrete class label (binary).")
print("  Goal: assign each passenger to 'survived' or 'did not survive'.\n")

print("If target = 'Fare' (float)     → Regression")
print("  Justification: Output is a continuous numeric value.")
print("  Goal: predict exact ticket price for a passenger.\n")

print("── Feature Overview ─────────────────────────────────────────")
print(df[['Survived', 'Fare', 'Age', 'Pclass', 'Sex']].describe())

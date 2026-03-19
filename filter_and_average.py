import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())

# ── Filter dataset where target variable meets a condition ────────────────────

# Example 1: Survived == 1, compute average Fare for survivors
survived = df[df['Survived'] == 1]
avg_fare_survived = survived['Fare'].mean()
print(f"Survivors (Survived == 1)     : {len(survived)} passengers")
print(f"  Average Fare                : £{avg_fare_survived:.2f}")

# Example 2: Did not survive, compute average Age
not_survived = df[df['Survived'] == 0]
avg_age_not_survived = not_survived['Age'].mean()
print(f"\nDid Not Survive (Survived == 0): {len(not_survived)} passengers")
print(f"  Average Age                 : {avg_age_not_survived:.2f} years")

# Example 3: Fare above average, compute survival rate
avg_fare = df['Fare'].mean()
high_fare = df[df['Fare'] > avg_fare]
surv_rate = high_fare['Survived'].mean() * 100
print(f"\nHigh-Fare passengers (Fare > £{avg_fare:.2f}): {len(high_fare)}")
print(f"  Survival Rate               : {surv_rate:.1f}%")

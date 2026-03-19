# Week-04-Day-22-PM

**Dataset used:** Titanic (891 rows) — loaded directly via URL
(`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`)

To use your own CSV replace the `url` line with:
```python
df = pd.read_csv('your_file.csv')
```

---

# Part A — Concept Application (40%)

### 1. Identify ML Problem Type
*   Determine learning type: Supervised / Unsupervised / Reinforcement
*   If supervised, classify as Regression or Classification
*   Justify based on target variable <br>
[Solution](identify_ml_problem.py)

### 2. Data Handling with Pandas
*   Load dataset using Pandas
*   Handle missing values: median-fill Age, mode-fill Embarked, drop Cabin
*   Select and encode relevant features <br>
[Solution](data_handling.py)

### 3. Regression Task
*   Target: `Fare` (continuous)
*   Model: Linear Regression
*   Evaluate using MSE and RMSE <br>
[Solution](regression_task.py)

### 4. Classification Task
*   Target: `Survived` (binary 0/1)
*   Model: Logistic Regression
*   Evaluate using Accuracy and Confusion Matrix <br>
[Solution](classification_task.py)

### 5. Comparison — Regression vs Classification <br>
[Solution](comparison.py)

## Comparison Table

| Aspect | Regression | Classification |
| :--- | :--- | :--- |
| **Output Type** | Continuous float (e.g., £84.50) | Discrete label (e.g., 0 or 1) |
| **Use Cases** | Fare prediction, salary, temperature | Survival, spam, disease diagnosis |
| **Eval Metrics** | MSE, RMSE, MAE, R² | Accuracy, Precision, Recall, F1, AUC |
| **Model** | Linear Regression | Logistic Regression |
| **Loss Function** | Mean Squared Error | Cross-Entropy |

---

## Part B — Stretch Problem (30%)

### 1. Feature Analysis using Pandas
*   Compute correlation matrix between numerical features
*   Identify features most correlated with Survived and Fare <br>
[Solution](feature_analysis.py)

### 2. Improve Model Performance
*   Select features with highest absolute correlation to target
*   Remove low-correlation and irrelevant columns <br>
[Solution](feature_analysis.py)

### 3. How Feature Selection Impacts Models

**Regression (target: Fare):**
*   `Pclass` and `Sex` have the strongest correlation with Fare (negative — higher class = lower number = higher fare).
*   Removing weakly correlated features like `SibSp` and `Parch` reduces noise in the regression surface, improving RMSE.
*   Irrelevant features inflate model complexity without adding predictive power.

**Classification (target: Survived):**
*   `Sex` has the highest correlation with survival (r = −0.54); females had much higher survival rates.
*   `Pclass` and `Fare` capture socioeconomic privilege that influenced lifeboat access.
*   Dropping weakly correlated features like `SibSp` and `Parch` reduces noise and can prevent overfitting on smaller datasets.
*   Feature selection also speeds up training and makes models more interpretable.

---

## Part C — Interview Ready (20%)

**Q1 — What are the types of machine learning? Explain with examples.**

### Supervised Learning
The model learns from labelled input–output pairs. The goal is to learn a mapping function `f(X) → y`.
*   **Regression** — target is continuous: house price prediction, stock forecasting
*   **Classification** — target is discrete: spam detection, disease diagnosis, fraud detection

### Unsupervised Learning
The model finds patterns in unlabelled data — no ground truth is provided.
*   **Clustering** — K-Means groups similar customers for segmentation
*   **Dimensionality Reduction** — PCA compresses high-dimensional data for visualization

### Reinforcement Learning
An agent learns by interacting with an environment. It receives rewards for good actions and penalties for bad ones.
*   **Examples:** AlphaGo (board game), autonomous vehicle lane-keeping, robot arm control

| Type | Labels? | Goal | Example |
| :--- | :--- | :--- | :--- |
| Supervised | ✅ Yes | Learn input → output mapping | Titanic survival prediction |
| Unsupervised | ❌ No | Discover hidden structure | Customer segmentation |
| Reinforcement | 🏆 Rewards | Maximise cumulative reward | Game-playing AI |

**Q2 (Coding) — Filter dataset and compute average of a feature** <br>
[Solution](filter_and_average.py)

**Q3 — What is the difference between regression and classification?**

Regression predicts a **continuous numeric value** — the output can be any real number within a range. The model minimises the distance between predicted and actual values using metrics like MSE.

Classification predicts a **discrete class label** — the output is one of a fixed set of categories. The model learns a decision boundary and is evaluated using accuracy, F1-score, or AUC-ROC.

*   If you ask *"How much will this house sell for?"* → **Regression**
*   If you ask *"Will this passenger survive?"* → **Classification**

The same algorithm can often be adapted for both: Logistic Regression is a classification algorithm despite the name, while Linear Regression can be used as a building block for more complex regression models.

---

## Part D — AI-Augmented Task (10%)

### 1. Prompt AI:
*"Explain types of machine learning, regression, and classification with Python examples using Pandas."*

### 2. Document prompt and output

[AI Output](AI_output.md) for the above prompt

### 3. Evaluate

### Are Concepts Correctly Explained?

*   **Supervised / Unsupervised / Reinforcement:** All three types are correctly defined with appropriate real-world examples.
*   **Regression definition:** Correctly identifies continuous target and uses MSE/RMSE as metrics.
*   **Classification definition:** Correctly identifies discrete target and uses accuracy as metric.
*   **Pandas operations:** `fillna()`, `map()`, `drop()`, `corr()`, and filtering syntax are all correct and idiomatic.

### Is Code Runnable and Meaningful?

*   All snippets are self-contained and run without modification given `pandas`, `numpy`, `scikit-learn` are installed.
*   The AI correctly applied `df['Age'].fillna(df['Age'].median())` before modelling — an important step that avoids NaN errors.
*   The train/test split with `random_state=42` ensures reproducibility — good practice included by the AI.
*   The comparison table at the end is accurate and useful for quick reference.

> **One improvement made:**
> The AI only used `Pclass`, `Sex`, and `Age` as regression features. In our [regression_task.py](regression_task.py) we added `SibSp` and `Parch` as a baseline, then compared against the top-correlated feature set in [feature_analysis.py](feature_analysis.py) — demonstrating the feature selection impact required by Part B.

### Runnability

All code snippets execute without errors. The only dependency beyond the standard library is `scikit-learn` (`pip install scikit-learn pandas matplotlib`).

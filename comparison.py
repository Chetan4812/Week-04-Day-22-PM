import matplotlib.pyplot as plt
import numpy as np

# Visual comparison: Regression output vs Classification output

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Regression side ───────────────────────────────────────────────────────────
x = np.linspace(0, 10, 100)
y = 2.5 * x + np.random.default_rng(42).normal(0, 3, 100)
axes[0].scatter(x, y, color='steelblue', alpha=0.5, s=25)
axes[0].plot(x, 2.5 * x, 'r-', linewidth=2, label='Fitted line (ŷ = 2.5x)')
axes[0].set_title('Regression — Continuous Output', fontweight='bold')
axes[0].set_xlabel('Feature (X)')
axes[0].set_ylabel('Target (y)  — continuous float')
axes[0].legend()

# ── Classification side ───────────────────────────────────────────────────────
x2 = np.linspace(-6, 6, 300)
sigmoid = 1 / (1 + np.exp(-x2))
axes[1].plot(x2, sigmoid, 'purple', linewidth=2.5, label='Sigmoid (probability)')
axes[1].axhline(0.5, linestyle='--', color='gray', linewidth=1.2, label='Threshold = 0.5')
axes[1].fill_between(x2, 0, sigmoid, where=(sigmoid >= 0.5), alpha=0.15,
                     color='green', label='Class 1 (Survived)')
axes[1].fill_between(x2, 0, sigmoid, where=(sigmoid < 0.5), alpha=0.15,
                     color='red', label='Class 0 (Not Survived)')
axes[1].set_title('Classification — Discrete Output', fontweight='bold')
axes[1].set_xlabel('Feature (X)')
axes[1].set_ylabel('Probability → Class Label')
axes[1].legend(fontsize=8)

plt.suptitle('Regression vs Classification', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_vs_classification.png', dpi=150)
plt.show()

# ── Comparison Table (printed) ────────────────────────────────────────────────
print("── Regression vs Classification ─────────────────────────────")
print(f"{'Aspect':<22} {'Regression':<28} {'Classification'}")
print("─" * 72)
rows = [
    ("Output Type",    "Continuous float (e.g., £84.5)",  "Discrete label (e.g., 0 or 1)"),
    ("Use Cases",      "Price, temperature, salary",       "Spam, disease, fraud"),
    ("Eval Metrics",   "MSE, RMSE, MAE, R²",               "Accuracy, F1, AUC-ROC"),
    ("Model Example",  "Linear Regression",                "Logistic Regression"),
    ("Loss Function",  "Mean Squared Error",               "Cross-Entropy / Log Loss"),
]
for aspect, reg, clf in rows:
    print(f"  {aspect:<20} {reg:<28} {clf}")

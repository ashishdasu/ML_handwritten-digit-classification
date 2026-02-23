'''
    Ashish Dasu
    CS6140
    HW2

    Handwriting Classification
'''

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.makedirs("plots", exist_ok=True)

# load data
X_train = pd.read_csv("data/X_train.csv").values.astype(float)
X_test  = pd.read_csv("data/X_test.csv").values.astype(float)
y_train = pd.read_csv("data/y_train.csv").values.ravel().astype(int)
y_test  = pd.read_csv("data/y_test.csv").values.ravel().astype(int)

y_mll_train = pd.read_csv("data/y_MLL_train.csv").values.astype(int)
y_mll_test  = pd.read_csv("data/y_MLL_test.csv").values.astype(int)
labels_mll  = ["is_even", "is_greater_than_5", "is_prime"]

# normalize to [0,1] then standardize
X_train = X_train / 255.0
X_test  = X_test  / 255.0
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Part 1: softmax classification
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train_sc))
test_acc  = accuracy_score(y_test,  clf.predict(X_test_sc))
print(f"\n-- Part 1 --")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test  accuracy: {test_acc:.4f}")

# per-class accuracy
for digit in range(10):
    mask = y_test == digit
    acc  = accuracy_score(y_test[mask], clf.predict(X_test_sc[mask]))
    print(f"  Class {digit}: {acc:.4f}")

# coefficient histogram for class 0 vs class 7
coef_0 = clf.coef_[0]
coef_7 = clf.coef_[7]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(coef_0, bins=20, alpha=0.6, color='red',  label='Class 0')
ax.hist(coef_7, bins=20, alpha=0.6, color='blue', label='Class 7')
ax.set_xlabel("Coefficient value")
ax.set_ylabel("Count")
ax.set_title("Distribution of Softmax Coefficients: Class 0 vs Class 7")
ax.legend()
plt.tight_layout()
plt.savefig("plots/part1_coefficients.png", dpi=150)
plt.close()

# one sample image per class
preds  = clf.predict(X_test_sc)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
shown  = {}
idx    = 0
for ax in axes.flat:
    while idx < len(X_test):
        digit = y_test[idx]
        if digit not in shown:
            shown[digit] = True
            ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
            ax.set_title(f"True: {y_test[idx]}\nPred: {preds[idx]}", fontsize=8)
            ax.axis('off')
            idx += 1
            break
        idx += 1
plt.suptitle("Sample Test Images (one per class)", fontsize=10)
plt.tight_layout()
plt.savefig("plots/part1_images.png", dpi=150)
plt.close()

# Part 2: multi-label from raw pixels
print(f"\n-- Part 2 --")
models_p2 = []
for j, label in enumerate(labels_mll):
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    m.fit(X_train_sc, y_mll_train[:, j])
    tr_acc = accuracy_score(y_mll_train[:, j], m.predict(X_train_sc))
    te_acc = accuracy_score(y_mll_test[:, j],  m.predict(X_test_sc))
    print(f"  {label}: train={tr_acc:.4f}, test={te_acc:.4f}")
    models_p2.append(m)

# Part 3: use softmax probs as features
print(f"\n-- Part 3 --")
X_new_train = clf.predict_proba(X_train_sc)
X_new_test  = clf.predict_proba(X_test_sc)

for j, label in enumerate(labels_mll):
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    m.fit(X_new_train, y_mll_train[:, j])
    tr_acc = accuracy_score(y_mll_train[:, j], m.predict(X_new_train))
    te_acc = accuracy_score(y_mll_test[:, j],  m.predict(X_new_test))
    print(f"  {label}: train={tr_acc:.4f}, test={te_acc:.4f}")

print("\nDone.")

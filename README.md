# Handwriting Classification

Logistic regression-based digit classification on an 8x8 handwritten digit dataset, exploring both multiclass and multi-label setups.

## What it does

- **Multiclass digit recognition:** Multinomial logistic regression (softmax) over raw pixel features to classify digits 0–9. Includes per-class accuracy breakdown and coefficient distribution plots.
- **Multi-label classification from pixels:** Three binary classifiers predicting abstract digit properties — `is_even`, `is_greater_than_5`, and `is_prime` — trained directly on pixel features.
- **Multi-label classification from learned representations:** Same binary classifiers, but using the softmax probability outputs as features rather than raw pixels, demonstrating the value of learned intermediate representations.

## Structure

```
data/               # Train/test CSVs (pixels + labels)
plots/              # Generated figures
handwriting_classification.py   # Main script
hw2_cs6140_adasu.pdf            # Writeup
```

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib
python handwriting_classification.py
```

Output plots are saved to `plots/`.

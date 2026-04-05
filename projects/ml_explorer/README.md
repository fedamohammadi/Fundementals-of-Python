# 🤖 ML Explorer

An interactive, multi-page web application for exploring **Machine Learning** concepts — built entirely in Python with Streamlit. Tune hyperparameters, visualise decision boundaries, compare regression types, and train models live in your browser.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/fedamohammadi/Fundementals-of-Python/main/projects/ml_explorer/app.py)

> **Run it locally in one click** → double-click `run.bat`  
> **Or in your terminal:**
> ```bash
> # from the repo root
> .venv\Scripts\streamlit run projects/ml_explorer/app.py
> ```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pages](#pages)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Deploy to the Cloud](#deploy-to-the-cloud)
- [Project Structure](#project-structure)

---

## Overview

ML Explorer is a self-contained learning tool that covers the full breadth of applied machine learning — from probability distributions and regression mathematics to ensemble methods and model evaluation. Every concept is paired with an **interactive demo** where you control the parameters and see results instantly.

It is designed to be used alongside the `Fundamentals of Python` learning repository as a practical companion to the econometrics and data analysis scripts.

---

## Pages

### 🏠 Home
- What is Machine Learning? — taxonomy and key concepts
- Glossary of core terms (features, overfitting, gradient descent, cross-validation, …)
- Quick-navigation overview of all topics

### 📊 Supervised Learning
Five fully interactive tabs:

| Tab | What you can do |
|-----|----------------|
| **Linear Regression** | Adjust slope, noise, sample size. Toggle residual lines. Live R² and RMSE. |
| **Logistic Regression** | Control class separation and regularisation C. Live decision boundary contour. |
| **Decision Tree** | Tune depth and criterion (Gini / Entropy). Visualise decision regions. |
| **Random Forest** | Set number of trees, depth, feature fraction. Compare against a single tree. |
| **KNN & SVM** | Switch between algorithms. Tune K, kernel, C, gamma. Side-by-side decision boundary. |

### 🔍 Unsupervised Learning
Three tabs:

| Tab | What you can do |
|-----|----------------|
| **K-Means** | Choose K, spread, and sample size. Voronoi regions + elbow chart to select optimal K. |
| **PCA** | Control features and components. Scree plot with cumulative variance. 2D projection scatter. |
| **DBSCAN** | Pick dataset shape (Moons / Circles / Blobs). Tune ε and min_samples. Noise points highlighted. |

### 📈 Probability & Regression
Three tabs:

| Tab | What you can do |
|-----|----------------|
| **Distributions** | Explore Normal, Binomial, Poisson, Exponential, and Uniform. Full parameter control + optional CDF overlay. |
| **Regression Types** | Compare OLS, Ridge, Lasso, and Polynomial on the same noisy data. Tune α, polynomial degree, noise. |
| **Bias-Variance Tradeoff** | Adjust polynomial degree and watch bias² vs variance change across bootstrap samples. Live decomposition chart. |

### 🎮 Model Playground
A full end-to-end training environment:

- **5 datasets** — Iris, Wine, Breast Cancer, Synthetic Classification, Synthetic Regression
- **9 models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, Ridge, Lasso
- **Full hyperparameter control** from the sidebar
- **Evaluation outputs** — Accuracy, Precision, Recall, F1, Cross-validation score, Confusion Matrix, ROC / AUC curve, Feature Importance / Coefficients
- **Dataset preview** table

---

## Features

- **Fully reactive** — every chart updates instantly as you move a slider or change a setting
- **Dark theme** with a purple/cyan/amber colour palette
- **Multi-page navigation** via Streamlit's built-in sidebar
- **No training step needed** — models retrain automatically on every interaction
- **Educational cards** on every page explaining the intuition behind each algorithm
- **Zero configuration** — just install dependencies and run

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io) | Web app framework and reactive UI |
| [Plotly](https://plotly.com/python/) | Interactive charts and visualisations |
| [scikit-learn](https://scikit-learn.org) | All ML models and datasets |
| [NumPy](https://numpy.org) | Numerical computing |
| [Pandas](https://pandas.pydata.org) | Data manipulation and preview table |
| [SciPy](https://scipy.org) | Probability distribution functions |

---

## Installation

Make sure you have Python 3.10+ and a virtual environment set up at the repo root.

```bash
# 1. Clone the repository
git clone https://github.com/fedamohammadi/Fundementals-of-Python.git
cd Fundementals-of-Python

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r projects/ml_explorer/requirements.txt
```

---

## Running the App

**Option 1 — One click (Windows)**

Double-click `projects/ml_explorer/run.bat`

**Option 2 — Terminal**

```bash
# From the repo root
.venv\Scripts\streamlit run projects/ml_explorer/app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deploy to the Cloud

You can host ML Explorer for free on [Streamlit Community Cloud](https://streamlit.io/cloud) so anyone can access it via a permanent public URL — no local setup required.

**Steps:**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Set the following:

   | Field | Value |
   |-------|-------|
   | Repository | `fedamohammadi/Fundementals-of-Python` |
   | Branch | `main` |
   | Main file path | `projects/ml_explorer/app.py` |

4. Click **"Deploy"** — your permanent public URL will be ready in ~2 minutes

Once deployed, the badge at the top of this README will link directly to your live app.

---

## Project Structure

```
projects/ml_explorer/
│
├── app.py                          # Home page (entry point)
│
├── pages/
│   ├── 1_Supervised_Learning.py    # Linear Regression, Logistic, Tree, RF, KNN/SVM
│   ├── 2_Unsupervised_Learning.py  # K-Means, PCA, DBSCAN
│   ├── 3_Probability_Regression.py # Distributions, Regression types, Bias-Variance
│   └── 4_Model_Playground.py       # Full train/evaluate/compare UI
│
├── utils/
│   ├── __init__.py
│   └── styles.py                   # Shared CSS, colour palette, helper functions
│
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
│
├── requirements.txt                # Python dependencies
├── run.bat                         # One-click launcher (Windows)
└── README.md                       # This file
```

---

*Part of the [Fundamentals of Python](https://github.com/fedamohammadi/Fundementals-of-Python) learning repository.*

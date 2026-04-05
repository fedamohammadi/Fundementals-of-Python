"""
Model Playground — pick a dataset, choose a model, tune hyperparameters, evaluate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                               make_classification, make_regression)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, precision_score, r2_score,
                              recall_score, roc_auc_score, roc_curve,
                              root_mean_squared_error)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.styles import (AMBER, CHART_COLORS, CYAN, EMERALD, MUTED, PLOTLY_TEMPLATE,
                          PURPLE, ROSE, TEXT, card, divider, info, inject_css, section)

st.set_page_config(page_title="Model Playground · ML Explorer",
                   page_icon="🎮", layout="wide")
inject_css()

st.markdown('<div class="hero-title">🎮 Model Playground</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Pick a dataset, choose a model, tune hyperparameters, and evaluate — all live.</div>',
            unsafe_allow_html=True)

divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Dataset & Model config
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1rem;font-weight:700;color:{TEXT};margin-bottom:.5rem;'>⚙️ Configuration</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:.85rem;color:{MUTED};font-weight:600;'>DATASET</div>",
                unsafe_allow_html=True)

    dataset_name = st.selectbox("Choose dataset", [
        "Iris (Classification)",
        "Wine (Classification)",
        "Breast Cancer (Classification)",
        "Synthetic Classification",
        "Synthetic Regression",
    ], key="pg_data")

    is_regression = "Regression" in dataset_name

    n_pg: int = 300
    noise_pg: float = 1.0
    if "Synthetic" in dataset_name:
        n_pg = st.slider("Samples", 100, 1000, 300, key="pg_n")
        noise_pg = st.slider("Noise", 0.1, 3.0, 1.0, 0.1, key="pg_noise")
        seed_pg = st.number_input("Seed", 0, 999, 42, key="pg_seed")
    else:
        seed_pg = st.number_input("Seed", 0, 999, 42, key="pg_seed2")

    st.markdown("---")
    st.markdown(f"<div style='font-size:.85rem;color:{MUTED};font-weight:600;'>MODEL</div>",
                unsafe_allow_html=True)

    if is_regression:
        model_name = st.selectbox("Algorithm", [
            "Ridge Regression", "Lasso Regression"
        ], key="pg_model")
    else:
        model_name = st.selectbox("Algorithm", [
            "Logistic Regression", "Decision Tree", "Random Forest",
            "Gradient Boosting", "KNN", "SVM", "Naive Bayes"
        ], key="pg_model")

    st.markdown("---")
    st.markdown(f"<div style='font-size:.85rem;color:{MUTED};font-weight:600;'>HYPERPARAMETERS</div>",
                unsafe_allow_html=True)

    # Model-specific params
    params = {}
    if model_name == "Logistic Regression":
        params["C"]        = st.select_slider("C", [0.01,0.1,0.5,1.0,5.0,10.0], 1.0, key="p_C")
        params["max_iter"] = st.slider("Max iterations", 100, 2000, 500, key="p_iter")

    elif model_name == "Decision Tree":
        params["max_depth"] = st.slider("Max depth", 1, 20, 4, key="p_depth")
        params["criterion"] = st.radio("Criterion", ["gini", "entropy"], key="p_crit")

    elif model_name == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators",  10, 300, 100, key="p_trees")
        params["max_depth"]    = st.slider("Max depth",       1,  20,   5, key="p_depth2")

    elif model_name == "Gradient Boosting":
        params["n_estimators"]  = st.slider("n_estimators",   10, 300, 100, key="p_gb_trees")
        params["learning_rate"] = st.select_slider("Learning rate",
                                                    [0.005,0.01,0.05,0.1,0.2,0.5], 0.1, key="p_lr")
        params["max_depth"]     = st.slider("Max depth", 1, 8, 3, key="p_gb_depth")

    elif model_name == "KNN":
        params["n_neighbors"] = st.slider("K", 1, 25, 5, key="p_k")
        params["weights"]     = st.radio("Weights", ["uniform","distance"], key="p_w")

    elif model_name == "SVM":
        params["C"]      = st.select_slider("C", [0.01,0.1,1.0,5.0,10.0,50.0], 1.0, key="p_svm_C")
        params["kernel"] = st.selectbox("Kernel", ["rbf","linear","poly"], key="p_ker")

    elif model_name in ("Ridge Regression", "Lasso Regression"):
        params["alpha"] = st.select_slider("Alpha α",
                                           [0.001,0.01,0.1,0.5,1.0,5.0,10.0,50.0],
                                           1.0, key="p_alpha")

    st.markdown("---")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="pg_test")
    scale     = st.toggle("Standardise features", value=True, key="pg_scale")

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(name, n=300, noise=1.0, seed=42):
    if name == "Iris (Classification)":
        d = load_iris()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "Wine (Classification)":
        d = load_wine()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "Breast Cancer (Classification)":
        d = load_breast_cancer()
        return d.data, d.target, list(d.feature_names), ["malignant","benign"]
    elif name == "Synthetic Classification":
        X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                                    n_redundant=2, random_state=seed)
        feats = [f"Feature {i+1}" for i in range(X.shape[1])]
        return X, y, feats, ["Class 0","Class 1"]
    else:  # Synthetic Regression
        X, y = make_regression(n_samples=n, n_features=8, n_informative=5,
                                noise=noise * 20, random_state=seed)
        feats = [f"Feature {i+1}" for i in range(X.shape[1])]
        return X, y, feats, []

X, y, feature_names, class_names = get_data(
    dataset_name,
    n=n_pg,
    noise=noise_pg,
    seed=int(seed_pg),
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed_pg), stratify=(y if not is_regression else None))

if scale:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# Build & train model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name, p):
    if name == "Logistic Regression":
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"])
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], criterion=p["criterion"])
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"],
                                       max_depth=p["max_depth"], random_state=0)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=p["n_estimators"],
                                           learning_rate=p["learning_rate"],
                                           max_depth=p["max_depth"], random_state=0)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])
    elif name == "SVM":
        return SVC(C=p["C"], kernel=p["kernel"], probability=True)
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "Ridge Regression":
        return Ridge(alpha=p["alpha"])
    else:
        return Lasso(alpha=p["alpha"], max_iter=5000)

model = build_model(model_name, params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
section(f"Results — {model_name} on {dataset_name.split('(')[0].strip()}")

if is_regression:
    # ── Regression metrics ──────────────────────────────────────────────────
    r2   = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²",   f"{r2:.4f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("MAE",  f"{mae:.3f}")
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
    m4.metric("CV R² (5-fold)", f"{cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    fig_pp = go.Figure()
    fig_pp.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                 marker=dict(color=PURPLE, size=6, opacity=0.7),
                                 name="Predictions"))
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig_pp.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                 line=dict(color=AMBER, dash="dash"), name="Perfect"))
    fig_pp.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          title="Predicted vs Actual",
                          xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig_pp, use_container_width=True)

else:
    # ── Classification metrics ───────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cv_acc = cross_val_score(model, X, y, cv=5)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",          f"{acc:.2%}")
    m2.metric("Precision (w)",     f"{prec:.4f}")
    m3.metric("Recall (w)",        f"{rec:.4f}")
    m4.metric("F1 (weighted)",     f"{f1:.4f}")
    m5.metric("CV Acc (5-fold)",   f"{cv_acc.mean():.2%} ± {cv_acc.std():.2%}")

    col_cm, col_roc = st.columns([1, 1])

    # Confusion matrix
    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        labels_cm = class_names if class_names else [str(i) for i in sorted(set(y))]
        n_lbl = len(labels_cm)
        fig_cm = go.Figure(go.Heatmap(
            z=cm.tolist(),
            x=labels_cm,
            y=list(reversed(labels_cm)),
            colorscale=[[0, "#14142B"], [1, PURPLE]],
            showscale=True))
        for i in range(n_lbl):
            for j in range(n_lbl):
                fig_cm.add_annotation(
                    x=labels_cm[j],
                    y=labels_cm[n_lbl - 1 - i],
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white", size=14, family="monospace"))
        fig_cm.update_layout(template=PLOTLY_TEMPLATE, height=380,
                              title="Confusion Matrix",
                              xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve (binary only; for multiclass show per-class)
    with col_roc:
        n_classes = len(np.unique(y))
        try:
            if n_classes == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                              line=dict(color=PURPLE, width=3),
                                              name=f"ROC  AUC={auc:.3f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                              line=dict(color=MUTED, dash="dash"),
                                              name="Random"))
                fig_roc.update_layout(template=PLOTLY_TEMPLATE, height=380,
                                       title="ROC Curve",
                                       xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                # Multi-class: one-vs-rest AUC bars
                y_bin = label_binarize(y_test, classes=sorted(np.unique(y)))
                y_prob_mc = model.predict_proba(X_test)
                aucs = [roc_auc_score(y_bin[:, i], y_prob_mc[:, i])
                        for i in range(n_classes)]
                lbl = class_names if class_names else [f"Class {i}" for i in range(n_classes)]
                fig_auc = go.Figure(go.Bar(x=lbl, y=aucs,
                                            marker_color=CHART_COLORS[:n_classes]))
                fig_auc.add_hline(y=0.5, line_dash="dash", line_color=MUTED)
                fig_auc.update_layout(template=PLOTLY_TEMPLATE, height=380,
                                       title="Per-class AUC (One-vs-Rest)",
                                       yaxis=dict(range=[0, 1.05]))
                st.plotly_chart(fig_auc, use_container_width=True)
        except Exception:
            st.info("ROC curve requires probability estimates. Not available for this model/dataset combination.")

    # Feature Importance
    divider()
    section("Feature Importance")

    has_importance = hasattr(model, "feature_importances_")
    has_coef       = hasattr(model, "coef_")

    if has_importance:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        fig_fi = go.Figure(go.Bar(
            x=imp[idx], y=[feature_names[i] for i in idx],
            orientation="h",
            marker=dict(color=imp[idx], colorscale="Purples", showscale=False)))
        fig_fi.update_layout(template=PLOTLY_TEMPLATE, height=max(300, len(idx) * 22),
                              title="Feature Importances",
                              xaxis_title="Importance",
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    elif has_coef:
        coef = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
        # Handle multi-class (take mean absolute across classes)
        if model.coef_.ndim > 1:
            coef = np.abs(model.coef_).mean(axis=0)
        idx = np.argsort(np.abs(coef))[::-1][:20]
        fig_coef = go.Figure(go.Bar(
            x=coef[idx], y=[feature_names[i] for i in idx],
            orientation="h",
            marker_color=[EMERALD if v >= 0 else ROSE for v in coef[idx]]))
        fig_coef.update_layout(template=PLOTLY_TEMPLATE, height=max(300, len(idx) * 22),
                                title="Model Coefficients (top features)",
                                xaxis_title="Coefficient value",
                                yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        info("This model type does not expose feature importance or coefficients directly.")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset preview
# ─────────────────────────────────────────────────────────────────────────────
divider()
section("Dataset Preview")

df_prev = pd.DataFrame(X, columns=feature_names)
df_prev["target"] = y
st.dataframe(df_prev.head(20), use_container_width=True, height=320)

c_info1, c_info2, c_info3 = st.columns(3)
c_info1.metric("Total samples",  str(len(X)))
c_info2.metric("Features",       str(X.shape[1]))
c_info3.metric("Training samples", str(len(X_train)))

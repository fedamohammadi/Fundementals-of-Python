"""
Supervised Learning — interactive demos for regression and classification models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.styles import (AMBER, CYAN, EMERALD, PLOTLY_TEMPLATE, PURPLE, ROSE,
                          card, divider, hex_rgba, info, inject_css, section)

st.set_page_config(page_title="Supervised Learning · ML Explorer",
                   page_icon="📊", layout="wide")
inject_css()

st.markdown('<div class="hero-title">📊 Supervised Learning</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Explore regression and classification algorithms with live, interactive demos.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Linear Regression",
    "🔵 Logistic Regression",
    "🌳 Decision Tree",
    "🌲 Random Forest",
    "🤝 KNN & SVM",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section("Linear Regression")
    info("Linear Regression fits a straight line y = β₀ + β₁x to minimise the sum of squared residuals (OLS).")

    c_ctrl, c_plot = st.columns([1, 2])

    with c_ctrl:
        n_pts   = st.slider("Number of points", 30, 300, 100, key="lr_n")
        noise   = st.slider("Noise level",       0.1, 5.0, 1.5, 0.1, key="lr_noise")
        true_slope = st.slider("True slope",    -5.0, 5.0, 2.0, 0.1, key="lr_slope")
        seed    = st.number_input("Random seed", 0, 999, 42, key="lr_seed")
        show_residuals = st.toggle("Show residuals", value=True, key="lr_resid")

    rng = np.random.default_rng(int(seed))
    X_raw = rng.uniform(-3, 3, n_pts)
    y_raw = true_slope * X_raw + rng.normal(0, noise, n_pts)
    model = LinearRegression().fit(X_raw.reshape(-1, 1), y_raw)
    x_line = np.linspace(-3, 3, 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    y_pred = model.predict(X_raw.reshape(-1, 1))

    with c_plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_raw, y=y_raw, mode="markers",
                                 marker=dict(color=CYAN, size=6, opacity=0.7),
                                 name="Data points"))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                 line=dict(color=PURPLE, width=3),
                                 name=f"Fit: ŷ = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"))
        if show_residuals:
            for xi, yi, ypi in zip(X_raw, y_raw, y_pred):
                fig.add_shape(type="line", x0=xi, x1=xi, y0=yi, y1=ypi,
                              line=dict(color=ROSE, width=1, dash="dot"))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          title="Linear Regression Fit",
                          xaxis_title="X", yaxis_title="y",
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slope (β₁)",     f"{model.coef_[0]:.3f}")
    m2.metric("Intercept (β₀)", f"{model.intercept_:.3f}")
    m3.metric("R²",             f"{r2_score(y_raw, y_pred):.4f}")
    m4.metric("RMSE",           f"{root_mean_squared_error(y_raw, y_pred):.3f}")

    divider()
    card("How it works",
         "OLS finds β that minimises <strong>Σ(yᵢ − ŷᵢ)²</strong>. "
         "The closed-form solution is <strong>β = (XᵀX)⁻¹Xᵀy</strong>. "
         "R² measures the fraction of variance explained (1 = perfect, 0 = baseline mean).",
         badge="OLS", badge_style="purple")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section("Logistic Regression")
    info("Logistic Regression models the probability of a binary class using the sigmoid function σ(z) = 1/(1+e⁻ᶻ). Decision boundary where P=0.5.")

    c_ctrl2, c_plot2 = st.columns([1, 2])

    with c_ctrl2:
        n2     = st.slider("Samples",        100, 600, 300, key="log_n")
        sep2   = st.slider("Class separation", 0.5, 3.0, 1.5, 0.1, key="log_sep")
        C_val  = st.select_slider("Regularisation C",
                                  options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                                  value=1.0, key="log_C")
        seed2  = st.number_input("Seed", 0, 999, 7, key="log_seed")

    rng2 = np.random.default_rng(int(seed2))
    X2, y2 = make_classification(n_samples=n2, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1,
                                  class_sep=sep2, random_state=int(seed2))
    scaler = StandardScaler()
    X2s = scaler.fit_transform(X2)
    clf2 = LogisticRegression(C=C_val, max_iter=500).fit(X2s, y2)
    acc2 = accuracy_score(y2, clf2.predict(X2s))

    h = 0.05
    x_min, x_max = X2s[:, 0].min() - 1, X2s[:, 0].max() + 1
    y_min, y_max = X2s[:, 1].min() - 1, X2s[:, 1].max() + 1
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    Z2 = clf2.predict_proba(np.c_[xx2.ravel(), yy2.ravel()])[:, 1].reshape(xx2.shape)

    with c_plot2:
        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=np.arange(x_min, x_max, h),
                                   y=np.arange(y_min, y_max, h),
                                   z=Z2, showscale=True,
                                   colorscale=[[0, hex_rgba(PURPLE, 0.33)],
                                               [0.5, "rgba(255,255,255,0.13)"],
                                               [1, hex_rgba(CYAN, 0.33)]],
                                   contours=dict(showlines=False),
                                   name="P(class=1)"))
        colors = [PURPLE if c == 0 else CYAN for c in y2]
        fig2.add_trace(go.Scatter(x=X2s[:, 0], y=X2s[:, 1], mode="markers",
                                   marker=dict(color=colors, size=5, opacity=0.85,
                                               line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
                                   name="Samples"))
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=400,
                            title=f"Decision Boundary  (C={C_val}, acc={acc2:.2%})",
                            xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig2, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Training Accuracy", f"{acc2:.2%}")
    m2.metric("Coeff[0]", f"{clf2.coef_[0][0]:.3f}")
    m3.metric("Coeff[1]", f"{clf2.coef_[0][1]:.3f}")

    divider()
    card("Key insight — Regularisation C",
         "A <strong>small C</strong> applies strong regularisation → simpler boundary, may underfit. "
         "A <strong>large C</strong> relaxes regularisation → complex boundary, may overfit. "
         "The log-likelihood is maximised via gradient-based optimisation.",
         badge="Classification", badge_style="cyan")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Decision Tree
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section("Decision Tree")
    info("Decision Trees split data recursively on the feature that maximises information gain (or minimises Gini impurity). Depth controls complexity.")

    c_ctrl3, c_plot3 = st.columns([1, 2])

    with c_ctrl3:
        n3      = st.slider("Samples", 150, 600, 300, key="dt_n")
        depth3  = st.slider("Max depth", 1, 12, 3, key="dt_depth")
        seed3   = st.number_input("Seed", 0, 999, 0, key="dt_seed")
        crit3   = st.radio("Criterion", ["gini", "entropy"], key="dt_crit")

    X3, y3 = make_classification(n_samples=n3, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=int(seed3))
    X3s = StandardScaler().fit_transform(X3)
    clf3 = DecisionTreeClassifier(max_depth=depth3, criterion=crit3,
                                   random_state=int(seed3)).fit(X3s, y3)
    acc3 = accuracy_score(y3, clf3.predict(X3s))

    xx3, yy3 = np.meshgrid(np.linspace(X3s[:,0].min()-0.5, X3s[:,0].max()+0.5, 300),
                            np.linspace(X3s[:,1].min()-0.5, X3s[:,1].max()+0.5, 300))
    Z3 = clf3.predict(np.c_[xx3.ravel(), yy3.ravel()]).reshape(xx3.shape)

    with c_plot3:
        fig3 = go.Figure()
        fig3.add_trace(go.Contour(x=np.linspace(X3s[:,0].min()-0.5, X3s[:,0].max()+0.5, 300),
                                   y=np.linspace(X3s[:,1].min()-0.5, X3s[:,1].max()+0.5, 300),
                                   z=Z3.astype(float), showscale=False,
                                   colorscale=[[0, hex_rgba(PURPLE, 0.25)], [1, hex_rgba(CYAN, 0.25)]],
                                   contours=dict(showlines=True,
                                                 coloring="fill")))
        fig3.add_trace(go.Scatter(x=X3s[:, 0], y=X3s[:, 1], mode="markers",
                                   marker=dict(color=[PURPLE if c == 0 else CYAN for c in y3],
                                               size=5, opacity=0.9,
                                               line=dict(width=0.5, color="rgba(0,0,0,0.27)")),
                                   name="Samples"))
        fig3.update_layout(template=PLOTLY_TEMPLATE, height=400,
                            title=f"Decision Tree — depth={depth3}  acc={acc3:.2%}",
                            xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig3, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Training Accuracy", f"{acc3:.2%}")
    m2.metric("Tree Depth",        str(clf3.get_depth()))
    m3.metric("Leaves",            str(clf3.get_n_leaves()))

    divider()
    card("Gini vs Entropy",
         "<strong>Gini impurity</strong>: 1 − Σpᵢ². Faster to compute. Tends to isolate the most frequent class.<br>"
         "<strong>Entropy</strong>: −Σpᵢ log₂(pᵢ). Slightly more balanced splits. Both produce similar trees in practice.",
         badge="Tree", badge_style="amber")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section("Random Forest")
    info("Random Forest builds many decision trees on bootstrapped subsets of data and random feature subsets, then aggregates their votes (bagging).")

    c_ctrl4, c_plot4 = st.columns([1, 2])

    with c_ctrl4:
        n4       = st.slider("Samples",       150, 600, 300, key="rf_n")
        n_trees  = st.slider("n_estimators",   5, 200, 50,  key="rf_trees")
        depth4   = st.slider("Max depth",       1, 15,   5,  key="rf_depth")
        feat_frac = st.slider("Max features %", 20, 100, 70, key="rf_feats")
        seed4    = st.number_input("Seed", 0, 999, 42, key="rf_seed")

    X4, y4 = make_classification(n_samples=n4, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=int(seed4))
    X4s = StandardScaler().fit_transform(X4)
    clf4 = RandomForestClassifier(n_estimators=n_trees, max_depth=depth4,
                                   max_features=feat_frac/100,
                                   random_state=int(seed4)).fit(X4s, y4)
    acc4 = accuracy_score(y4, clf4.predict(X4s))

    xx4, yy4 = np.meshgrid(np.linspace(X4s[:,0].min()-0.5, X4s[:,0].max()+0.5, 250),
                            np.linspace(X4s[:,1].min()-0.5, X4s[:,1].max()+0.5, 250))
    proba4 = clf4.predict_proba(np.c_[xx4.ravel(), yy4.ravel()])[:, 1].reshape(xx4.shape)

    with c_plot4:
        fig4 = go.Figure()
        fig4.add_trace(go.Contour(x=np.linspace(X4s[:,0].min()-0.5, X4s[:,0].max()+0.5, 250),
                                   y=np.linspace(X4s[:,1].min()-0.5, X4s[:,1].max()+0.5, 250),
                                   z=proba4, showscale=True,
                                   colorscale="Purples",
                                   contours=dict(showlines=False),
                                   name="P(class=1)"))
        fig4.add_trace(go.Scatter(x=X4s[:, 0], y=X4s[:, 1], mode="markers",
                                   marker=dict(color=[PURPLE if c == 0 else EMERALD for c in y4],
                                               size=5, opacity=0.85),
                                   name="Samples"))
        fig4.update_layout(template=PLOTLY_TEMPLATE, height=400,
                            title=f"Random Forest — {n_trees} trees  acc={acc4:.2%}",
                            xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig4, use_container_width=True)

    # OOB-style: compare 1 tree vs forest
    single = DecisionTreeClassifier(max_depth=depth4, random_state=int(seed4)).fit(X4s, y4)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forest Accuracy",        f"{acc4:.2%}")
    m2.metric("Single Tree Accuracy",   f"{accuracy_score(y4, single.predict(X4s)):.2%}",
              delta=f"{(acc4-accuracy_score(y4, single.predict(X4s)))*100:+.1f}pp")
    m3.metric("Trees",  str(n_trees))
    m4.metric("Max Depth", str(depth4))

    divider()
    card("Why does bagging help?",
         "Each tree sees a different bootstrap sample → high variance per tree. "
         "By <strong>averaging</strong> many uncorrelated trees, variance cancels out while "
         "bias stays similar. The result is a much more robust model than any single tree.",
         badge="Ensemble", badge_style="green")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — KNN & SVM
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    section("KNN vs SVM")
    info("K-Nearest Neighbours classifies by majority vote among the K closest training points. "
         "SVM finds the maximum-margin hyperplane; the kernel trick enables non-linear boundaries.")

    algo5 = st.radio("Algorithm", ["KNN", "SVM"], horizontal=True, key="knn_svm_algo")

    c_ctrl5, c_plot5 = st.columns([1, 2])

    with c_ctrl5:
        n5   = st.slider("Samples", 150, 600, 300, key="ks_n")
        sep5 = st.slider("Class separation", 0.5, 3.0, 1.2, 0.1, key="ks_sep")
        seed5 = st.number_input("Seed", 0, 999, 21, key="ks_seed")
        if algo5 == "KNN":
            k5 = st.slider("K (neighbours)", 1, 25, 5, key="ks_k")
        else:
            kernel5 = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="ks_ker")
            C5  = st.select_slider("C", options=[0.01,0.1,1.0,5.0,10.0], value=1.0, key="ks_C")
            gam5 = st.select_slider("Gamma", options=["scale","auto",0.01,0.1,1.0],
                                    value="scale", key="ks_gam")

    X5, y5 = make_classification(n_samples=n5, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   class_sep=sep5, random_state=int(seed5))
    X5s = StandardScaler().fit_transform(X5)

    if algo5 == "KNN":
        clf5 = KNeighborsClassifier(n_neighbors=k5).fit(X5s, y5)
    else:
        clf5 = SVC(kernel=kernel5, C=C5, gamma=gam5, probability=True).fit(X5s, y5)

    acc5 = accuracy_score(y5, clf5.predict(X5s))
    xx5, yy5 = np.meshgrid(np.linspace(X5s[:,0].min()-0.5, X5s[:,0].max()+0.5, 250),
                            np.linspace(X5s[:,1].min()-0.5, X5s[:,1].max()+0.5, 250))
    Z5 = clf5.predict(np.c_[xx5.ravel(), yy5.ravel()]).reshape(xx5.shape)

    with c_plot5:
        fig5 = go.Figure()
        fig5.add_trace(go.Contour(x=np.linspace(X5s[:,0].min()-0.5, X5s[:,0].max()+0.5, 250),
                                   y=np.linspace(X5s[:,1].min()-0.5, X5s[:,1].max()+0.5, 250),
                                   z=Z5.astype(float), showscale=False,
                                   colorscale=[[0, hex_rgba(AMBER, 0.27)], [1, hex_rgba(CYAN, 0.27)]],
                                   contours=dict(showlines=True, coloring="fill")))
        fig5.add_trace(go.Scatter(x=X5s[:,0], y=X5s[:,1], mode="markers",
                                   marker=dict(color=[AMBER if c==0 else CYAN for c in y5],
                                               size=5, opacity=0.9),
                                   name="Samples"))
        label = f"K={k5}" if algo5=="KNN" else f"SVM({kernel5})"
        fig5.update_layout(template=PLOTLY_TEMPLATE, height=400,
                            title=f"{algo5} — {label}  acc={acc5:.2%}",
                            xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig5, use_container_width=True)

    st.metric("Training Accuracy", f"{acc5:.2%}")

    divider()
    if algo5 == "KNN":
        card("KNN intuition",
             "No training phase — KNN memorises all training points. "
             "At prediction time it finds the K nearest neighbours and takes a majority vote. "
             "<strong>Small K</strong> → complex, noisy boundary. <strong>Large K</strong> → smooth, biased boundary. "
             "Sensitive to feature scale (always standardise!).",
             badge="Lazy learner", badge_style="amber")
    else:
        card("SVM intuition",
             "SVM maximises the <strong>margin</strong> between classes. Support vectors are the points on the margin. "
             "The <strong>kernel trick</strong> implicitly maps data to higher dimensions where classes become linearly separable. "
             "<strong>RBF kernel</strong>: great general choice. <strong>C</strong>: trade-off between margin width and misclassifications.",
             badge="Max-margin", badge_style="cyan")

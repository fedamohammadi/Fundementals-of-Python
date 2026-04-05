"""
Probability & Regression — distributions, regression types, bias-variance tradeoff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.styles import (AMBER, CHART_COLORS, CYAN, EMERALD, MUTED, PLOTLY_TEMPLATE,
                          PURPLE, ROSE, TEXT, card, divider, hex_rgba, info,
                          inject_css, section)

st.set_page_config(page_title="Probability & Regression · ML Explorer",
                   page_icon="📈", layout="wide")
inject_css()

st.markdown('<div class="hero-title">📈 Probability & Regression</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Probability distributions, regression types, and the bias-variance tradeoff.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎲 Distributions", "📏 Regression Types", "⚖️ Bias-Variance Tradeoff"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Probability Distributions
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section("Probability Distributions")
    info("A probability distribution describes how probability is spread over the values of a random variable. "
         "Understanding distributions is foundational to ML — loss functions, assumptions, and inference all depend on them.")

    dist_name = st.radio("Distribution", ["Normal", "Binomial", "Poisson", "Exponential", "Uniform"],
                          horizontal=True, key="dist_name")

    c_ctrl, c_plot = st.columns([1, 2])

    x_range = np.linspace(-5, 15, 1000)

    with c_ctrl:
        if dist_name == "Normal":
            mu    = st.slider("Mean (μ)",       -5.0, 5.0,  0.0, 0.1, key="d_mu")
            sigma = st.slider("Std Dev (σ)",     0.1, 4.0,  1.0, 0.1, key="d_sig")
            x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            pdf   = stats.norm.pdf(x_range, mu, sigma)
            cdf   = stats.norm.cdf(x_range, mu, sigma)
            mean_d, var_d = mu, sigma**2

        elif dist_name == "Binomial":
            n_bin = st.slider("n (trials)",  1,  50, 20, key="d_n")
            p_bin = st.slider("p (success)", 0.01, 0.99, 0.5, 0.01, key="d_p")
            x_range = np.arange(0, n_bin + 1)
            pdf   = stats.binom.pmf(x_range, n_bin, p_bin)
            cdf   = stats.binom.cdf(x_range, n_bin, p_bin)
            mean_d, var_d = n_bin * p_bin, n_bin * p_bin * (1 - p_bin)

        elif dist_name == "Poisson":
            lam = st.slider("λ (rate)", 0.5, 20.0, 5.0, 0.5, key="d_lam")
            x_range = np.arange(0, int(lam * 3) + 1)
            pdf   = stats.poisson.pmf(x_range, lam)
            cdf   = stats.poisson.cdf(x_range, lam)
            mean_d, var_d = lam, lam

        elif dist_name == "Exponential":
            lam_e = st.slider("λ (rate)", 0.1, 5.0, 1.0, 0.1, key="d_lame")
            x_range = np.linspace(0, 10 / lam_e, 1000)
            pdf   = stats.expon.pdf(x_range, scale=1/lam_e)
            cdf   = stats.expon.cdf(x_range, scale=1/lam_e)
            mean_d, var_d = 1/lam_e, 1/lam_e**2

        else:  # Uniform
            a_u = st.slider("a (lower)", -5.0, 0.0, 0.0, 0.5, key="d_a")
            b_u = st.slider("b (upper)",  0.5, 10.0, 5.0, 0.5, key="d_b")
            if b_u <= a_u:
                b_u = a_u + 0.5
            x_range = np.linspace(a_u - 0.5, b_u + 0.5, 1000)
            pdf   = stats.uniform.pdf(x_range, a_u, b_u - a_u)
            cdf   = stats.uniform.cdf(x_range, a_u, b_u - a_u)
            mean_d, var_d = (a_u + b_u)/2, (b_u - a_u)**2 / 12

        show_cdf = st.toggle("Overlay CDF", value=False, key="d_cdf")

    with c_plot:
        is_discrete = dist_name in ("Binomial", "Poisson")
        fig_d = go.Figure()

        if is_discrete:
            fig_d.add_trace(go.Bar(x=x_range, y=pdf,
                                    marker_color=PURPLE, opacity=0.85, name="PMF"))
        else:
            fig_d.add_trace(go.Scatter(x=x_range, y=pdf, mode="lines",
                                        line=dict(color=PURPLE, width=3), name="PDF",
                                        fill="tozeroy", fillcolor=hex_rgba(PURPLE, 0.19)))
        if show_cdf:
            fig_d.add_trace(go.Scatter(x=x_range, y=cdf, mode="lines",
                                        line=dict(color=AMBER, width=2, dash="dot"),
                                        name="CDF", yaxis="y2"))
            fig_d.update_layout(yaxis2=dict(title="CDF", overlaying="y",
                                            side="right", range=[0, 1.05]))

        fig_d.update_layout(template=PLOTLY_TEMPLATE, height=380,
                             title=f"{dist_name} Distribution",
                             xaxis_title="x",
                             yaxis_title="P(X=x)" if is_discrete else "f(x)")
        st.plotly_chart(fig_d, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Mean", f"{mean_d:.4f}")
    m2.metric("Variance", f"{var_d:.4f}")

    divider()
    descs = {
        "Normal":      ("Symmetric bell curve. Defined by μ and σ. The <strong>Central Limit Theorem</strong> says sample means converge to Normal regardless of the underlying distribution. Fundamental to OLS regression assumptions.", "purple"),
        "Binomial":    ("Models the number of successes in n independent Bernoulli trials with probability p. Basis of Logistic Regression (binary outcome). Converges to Normal for large n.", "cyan"),
        "Poisson":     ("Models the number of events in a fixed interval when events occur at rate λ independently. Used in count regression, queuing theory, and natural language processing.", "amber"),
        "Exponential": ("Models the time between Poisson events. Memoryless property: P(X>s+t|X>s) = P(X>t). Used in survival analysis and reliability engineering.", "green"),
        "Uniform":     ("Equal probability everywhere in [a,b]. Used as a non-informative prior in Bayesian inference and for random initialisation in neural networks.", "rose"),
    }
    body, style = descs[dist_name]
    card(f"About the {dist_name} Distribution", body, badge=dist_name, badge_style=style)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Regression Types
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section("Regression Types: OLS · Ridge · Lasso · Polynomial")
    info("All four models are fit on the same noisy data. Compare how regularisation and polynomial features change the fit.")

    c_ctrl2, c_plot2 = st.columns([1, 2])

    with c_ctrl2:
        n_reg    = st.slider("Samples",        30, 300, 80, key="reg_n")
        noise_r  = st.slider("Noise",          0.1, 5.0, 1.5, 0.1, key="reg_noise")
        true_deg = st.slider("True curve degree", 1, 5, 2, key="reg_true")
        alpha_r  = st.select_slider("Ridge/Lasso α",
                                    options=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                                    value=1.0, key="reg_alpha")
        poly_deg = st.slider("Polynomial degree to fit", 1, 9, 3, key="reg_poly")
        seed_r   = st.number_input("Seed", 0, 999, 7, key="reg_seed")
        models_on = st.multiselect("Show models",
                                    ["OLS", "Ridge", "Lasso", "Polynomial"],
                                    default=["OLS", "Ridge", "Lasso", "Polynomial"],
                                    key="reg_models")

    rng_r = np.random.default_rng(int(seed_r))
    coefs_true = rng_r.standard_normal(true_deg + 1)
    X_r = rng_r.uniform(-3, 3, n_reg)
    y_r = sum(coefs_true[i] * X_r**i for i in range(true_deg + 1))
    y_r = y_r + rng_r.normal(0, noise_r, n_reg)

    X_r_2d  = X_r.reshape(-1, 1)
    X_poly  = PolynomialFeatures(poly_deg).fit_transform(X_r_2d)
    x_line  = np.linspace(-3, 3, 300)
    x_line2 = x_line.reshape(-1, 1)
    X_line_poly = PolynomialFeatures(poly_deg).fit_transform(x_line2)

    fitted_models = {
        "OLS":        LinearRegression().fit(X_r_2d, y_r),
        "Ridge":      make_pipeline(StandardScaler(), Ridge(alpha=alpha_r)).fit(X_r_2d, y_r),
        "Lasso":      make_pipeline(StandardScaler(), Lasso(alpha=alpha_r, max_iter=5000)).fit(X_r_2d, y_r),
        "Polynomial": LinearRegression().fit(X_poly, y_r),
    }

    colors_reg = {"OLS": CYAN, "Ridge": EMERALD, "Lasso": AMBER, "Polynomial": ROSE}

    with c_plot2:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=X_r, y=y_r, mode="markers",
                                    marker=dict(color=MUTED, size=5, opacity=0.6),
                                    name="Data"))
        for name, mdl in fitted_models.items():
            if name not in models_on:
                continue
            if name == "Polynomial":
                y_hat = mdl.predict(X_line_poly)
            else:
                y_hat = mdl.predict(x_line2)
            r2 = r2_score(y_r, (fitted_models[name].predict(X_poly)
                                 if name == "Polynomial"
                                 else fitted_models[name].predict(X_r_2d)))
            fig_r.add_trace(go.Scatter(x=x_line, y=y_hat, mode="lines",
                                        line=dict(color=colors_reg[name], width=2.5),
                                        name=f"{name}  R²={r2:.3f}"))
        fig_r.update_layout(template=PLOTLY_TEMPLATE, height=420,
                             title="Regression Comparison",
                             xaxis_title="X", yaxis_title="y",
                             yaxis=dict(range=[y_r.min()-2, y_r.max()+2]))
        st.plotly_chart(fig_r, use_container_width=True)

    if models_on:
        cols_m = st.columns(len(models_on))
        for col, name in zip(cols_m, models_on):
            mdl = fitted_models[name]
            y_p = (mdl.predict(X_poly) if name == "Polynomial"
                   else mdl.predict(X_r_2d))
            col.metric(f"{name} R²",   f"{r2_score(y_r, y_p):.4f}")

    divider()

    cr1, cr2 = st.columns(2)
    with cr1:
        card("OLS vs Ridge vs Lasso",
             "<strong>OLS</strong>: minimises Σ(y−ŷ)². No constraint on coefficients — can overfit with many features.<br>"
             "<strong>Ridge (L2)</strong>: adds λΣβ² penalty. Shrinks all coefficients towards 0, keeps all features.<br>"
             "<strong>Lasso (L1)</strong>: adds λΣ|β| penalty. Performs automatic feature selection by setting some β to exactly 0.",
             badge="Regularisation", badge_style="amber")
    with cr2:
        card("Polynomial Regression",
             "A linear model in the polynomial feature space. Features: 1, x, x², x³, ... "
             "Captures non-linear curves while remaining a linear model mathematically. "
             "High degree → risk of extreme overfitting. Regularise with Ridge or Lasso.",
             badge="Polynomial", badge_style="rose")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Bias-Variance Tradeoff
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section("Bias-Variance Tradeoff")
    info("Every ML model faces a fundamental tension: a model that is too simple (high bias) underfits, "
         "while a model that is too complex (high variance) overfits. The goal is to find the sweet spot.")

    c_ctrl3, c_plot3 = st.columns([1, 2])

    with c_ctrl3:
        poly_bv   = st.slider("Polynomial degree", 1, 12, 1, key="bv_deg")
        n_bv      = st.slider("Training samples",  10, 100, 30, key="bv_n")
        noise_bv  = st.slider("Noise",            0.1, 3.0, 1.0, 0.1, key="bv_noise")
        n_trials  = st.slider("Bootstrap trials", 10, 60, 30, key="bv_trials")
        seed_bv   = st.number_input("Seed", 0, 999, 0, key="bv_seed")

    rng_bv = np.random.default_rng(int(seed_bv))

    def true_f(x):
        return np.sin(x) + 0.3 * x

    x_test = np.linspace(-3.5, 3.5, 200)
    y_true_bv = true_f(x_test)

    preds_bv = []
    for _ in range(n_trials):
        x_tr = rng_bv.uniform(-3, 3, n_bv)
        y_tr = true_f(x_tr) + rng_bv.normal(0, noise_bv, n_bv)
        try:
            pipe = make_pipeline(PolynomialFeatures(poly_bv), LinearRegression())
            pipe.fit(x_tr.reshape(-1, 1), y_tr)
            preds_bv.append(pipe.predict(x_test.reshape(-1, 1)))
        except Exception:
            pass

    preds_bv = np.array(preds_bv)  # (n_trials, 200)
    mean_pred = preds_bv.mean(axis=0)
    bias2     = (mean_pred - y_true_bv) ** 2
    variance  = preds_bv.var(axis=0)

    with c_plot3:
        fig_bv = go.Figure()

        # Individual model fits
        for i, pred in enumerate(preds_bv):
            fig_bv.add_trace(go.Scatter(x=x_test, y=pred, mode="lines",
                                         line=dict(color=PURPLE, width=0.8),
                                         opacity=0.15,
                                         showlegend=(i == 0),
                                         name=f"{n_trials} bootstrapped fits"))
        # Mean prediction
        fig_bv.add_trace(go.Scatter(x=x_test, y=mean_pred, mode="lines",
                                     line=dict(color=AMBER, width=3, dash="dash"),
                                     name="Mean prediction"))
        # True function
        fig_bv.add_trace(go.Scatter(x=x_test, y=y_true_bv, mode="lines",
                                     line=dict(color=EMERALD, width=2.5),
                                     name="True function"))
        fig_bv.update_layout(template=PLOTLY_TEMPLATE, height=360,
                              title=f"Degree-{poly_bv} polynomial across {n_trials} random datasets",
                              xaxis_title="X", yaxis_title="y",
                              yaxis=dict(range=[-5, 5]))
        st.plotly_chart(fig_bv, use_container_width=True)

    # Bias / Variance breakdown
    fig_bv2 = go.Figure()
    fig_bv2.add_trace(go.Scatter(x=x_test, y=bias2, fill="tozeroy",
                                  fillcolor=hex_rgba(ROSE, 0.27),
                                  line=dict(color=ROSE, width=1.5),
                                  name="Bias²"))
    fig_bv2.add_trace(go.Scatter(x=x_test, y=variance, fill="tozeroy",
                                  fillcolor=hex_rgba(PURPLE, 0.27),
                                  line=dict(color=PURPLE, width=1.5),
                                  name="Variance"))
    fig_bv2.update_layout(template=PLOTLY_TEMPLATE, height=250,
                           title="Bias² and Variance across input space",
                           xaxis_title="X", yaxis_title="Error",
                           yaxis=dict(range=[0, min(10, float(bias2.max() + variance.max()) * 1.2)]))
    st.plotly_chart(fig_bv2, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Bias²",    f"{bias2.mean():.4f}")
    m2.metric("Mean Variance", f"{variance.mean():.4f}")
    m3.metric("Total Error",   f"{(bias2 + variance).mean():.4f}")

    divider()
    card("The fundamental equation",
         "Expected prediction error = <strong>Bias²</strong> + <strong>Variance</strong> + <strong>Irreducible Noise</strong><br>"
         "A low-degree polynomial: high bias (underfits), low variance (stable across datasets).<br>"
         "A high-degree polynomial: low bias, high variance (wildly different across datasets).<br>"
         "The optimal model minimises their sum. Regularisation is one way to trade variance for acceptable bias.",
         badge="MSE = Bias² + Var + σ²", badge_style="amber")

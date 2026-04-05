"""
ML Explorer — Home
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.styles import (AMBER, CYAN, EMERALD, MUTED, PURPLE, ROSE, TEXT,
                          card, divider, inject_css, section)

st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.5rem;">
    <div class="hero-title">ML Explorer 🤖</div>
    <div class="hero-sub">
        An interactive guide to Machine Learning — models, math, and intuition.<br>
        Explore, tune hyperparameters, and see results live.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick-nav cards ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;border-color:{PURPLE}66;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-weight:700;color:{TEXT};margin:.4rem 0 .2rem;">Supervised</div>
        <div style="font-size:.82rem;color:{MUTED};">Regression · Classification<br>Trees · SVM · KNN</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;border-color:{CYAN}66;">
        <div style="font-size:2rem;">🔍</div>
        <div style="font-weight:700;color:{TEXT};margin:.4rem 0 .2rem;">Unsupervised</div>
        <div style="font-size:.82rem;color:{MUTED};">K-Means · PCA<br>DBSCAN · Autoencoders</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;border-color:{AMBER}66;">
        <div style="font-size:2rem;">📈</div>
        <div style="font-weight:700;color:{TEXT};margin:.4rem 0 .2rem;">Probability</div>
        <div style="font-size:.82rem;color:{MUTED};">Distributions · OLS<br>Ridge · Lasso · Poly</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;border-color:{EMERALD}66;">
        <div style="font-size:2rem;">🎮</div>
        <div style="font-weight:700;color:{TEXT};margin:.4rem 0 .2rem;">Playground</div>
        <div style="font-size:.82rem;color:{MUTED};">Pick a dataset · Train any model<br>Compare · Evaluate</div>
    </div>""", unsafe_allow_html=True)

divider()

# ── ML Taxonomy ───────────────────────────────────────────────────────────────
section("What is Machine Learning?")

st.markdown(f"""
<div class="info-box">
Machine Learning is a field of AI where systems <strong>learn patterns from data</strong>
to make decisions or predictions — without being explicitly programmed for each task.
The core idea: <em>instead of writing rules, you feed examples and let the algorithm
discover the rules itself.</em>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    card("🎓 Supervised Learning",
         "The model learns from <strong>labeled data</strong> (input → known output). "
         "Goal: predict labels on unseen inputs.<br><br>"
         "<strong>Examples:</strong> spam detection, house price prediction, "
         "image classification.",
         badge="Has labels", badge_style="purple")

with col_b:
    card("🔎 Unsupervised Learning",
         "The model finds <strong>hidden structure</strong> in <em>unlabeled</em> data. "
         "No correct answer provided — the algorithm clusters or compresses on its own.<br><br>"
         "<strong>Examples:</strong> customer segmentation, anomaly detection, topic modeling.",
         badge="No labels", badge_style="cyan")

with col_c:
    card("🕹️ Reinforcement Learning",
         "An agent learns by <strong>interacting with an environment</strong>, receiving "
         "rewards or penalties. Learns to maximise cumulative reward over time.<br><br>"
         "<strong>Examples:</strong> game-playing AIs, robotic control, recommendation engines.",
         badge="Reward signal", badge_style="amber")

divider()

# ── Key concepts ──────────────────────────────────────────────────────────────
section("Core Concepts at a Glance")

concepts = [
    ("Feature (X)",        "An input variable used to make a prediction. Also called a predictor or independent variable.", "purple"),
    ("Label / Target (y)", "The output the model tries to predict (in supervised learning).", "cyan"),
    ("Training set",       "Data the model learns from. Usually 70–80% of available data.", "amber"),
    ("Test set",           "Held-out data used to evaluate final model performance. Never seen during training.", "green"),
    ("Overfitting",        "Model memorises training data but fails on new data — too complex.", "rose"),
    ("Underfitting",       "Model is too simple to capture the underlying pattern — high bias.", "rose"),
    ("Hyperparameter",     "A setting chosen before training (e.g. learning rate, tree depth). Tuned by the developer.", "purple"),
    ("Loss function",      "Measures how wrong the model's predictions are. Training minimises this.", "cyan"),
    ("Gradient Descent",   "Iterative optimisation algorithm that updates model weights in the direction that reduces loss.", "amber"),
    ("Cross-validation",   "Technique to estimate model performance on unseen data by splitting training data into k folds.", "green"),
]

r1, r2 = st.columns(2)
for i, (term, desc, style) in enumerate(concepts):
    target = r1 if i % 2 == 0 else r2
    with target:
        card(term, desc, badge_style=style)

divider()

# ── Stats strip ───────────────────────────────────────────────────────────────
section("ML by the Numbers")

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Common algorithms covered", "15+")
s2.metric("Interactive demos", "12")
s3.metric("Probability distributions", "5")
s4.metric("Regression types", "5")
s5.metric("Model comparison metrics", "8")

divider()

# ── Sidebar guide ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:.5rem 0 1.2rem;">
        <span style="font-size:2rem;">🤖</span><br>
        <span style="font-size:1.1rem;font-weight:700;color:{TEXT};">ML Explorer</span><br>
        <span style="font-size:.78rem;color:{MUTED};">Interactive Machine Learning Guide</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{MUTED}33;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:.82rem;color:{MUTED};line-height:1.9;">
    📊 <strong style="color:{TEXT};">Supervised Learning</strong><br>
    &nbsp;&nbsp;Linear & Logistic Regression<br>
    &nbsp;&nbsp;Trees, Random Forest, SVM, KNN<br><br>
    🔍 <strong style="color:{TEXT};">Unsupervised Learning</strong><br>
    &nbsp;&nbsp;K-Means Clustering<br>
    &nbsp;&nbsp;PCA, DBSCAN<br><br>
    📈 <strong style="color:{TEXT};">Probability & Regression</strong><br>
    &nbsp;&nbsp;Distributions, OLS, Ridge, Lasso<br>
    &nbsp;&nbsp;Polynomial, Bias-Variance<br><br>
    🎮 <strong style="color:{TEXT};">Model Playground</strong><br>
    &nbsp;&nbsp;Train · Tune · Evaluate · Compare
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{MUTED}33;'>", unsafe_allow_html=True)
    st.caption("Use the navigation above to explore each topic.")

"""
Unsupervised Learning — K-Means, PCA, DBSCAN interactive demos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.styles import (AMBER, CHART_COLORS, CYAN, EMERALD, MUTED, PLOTLY_TEMPLATE,
                          PURPLE, ROSE, TEXT, card, divider, hex_rgba, info,
                          inject_css, section)

st.set_page_config(page_title="Unsupervised Learning · ML Explorer",
                   page_icon="🔍", layout="wide")
inject_css()

st.markdown('<div class="hero-title">🔍 Unsupervised Learning</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Discover hidden structure in data — no labels required.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔵 K-Means Clustering", "📐 PCA", "🌀 DBSCAN"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — K-Means
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section("K-Means Clustering")
    info("K-Means assigns each point to the nearest centroid, then re-computes centroids, repeating until convergence. K must be chosen upfront.")

    c_ctrl, c_plot = st.columns([1, 2])

    with c_ctrl:
        k        = st.slider("Number of clusters K", 2, 8, 3, key="km_k")
        n_km     = st.slider("Samples",              100, 600, 300, key="km_n")
        spread   = st.slider("Cluster spread",       0.3, 3.0, 1.0, 0.1, key="km_spread")
        seed_km  = st.number_input("Seed", 0, 999, 42, key="km_seed")
        show_vor = st.toggle("Show Voronoi-style regions", value=True, key="km_vor")

    X_km, y_true = make_blobs(n_samples=n_km, centers=k, cluster_std=spread,
                               random_state=int(seed_km))
    X_km_s = StandardScaler().fit_transform(X_km)

    km = KMeans(n_clusters=k, random_state=int(seed_km), n_init=10).fit(X_km_s)
    labels_km = km.labels_
    centers   = km.cluster_centers_
    inertia   = km.inertia_

    with c_plot:
        fig_km = go.Figure()

        if show_vor:
            h = 0.05
            xmin, xmax = X_km_s[:,0].min()-0.5, X_km_s[:,0].max()+0.5
            ymin, ymax = X_km_s[:,1].min()-0.5, X_km_s[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
            Z_km = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig_km.add_trace(go.Contour(
                x=np.arange(xmin, xmax, h), y=np.arange(ymin, ymax, h),
                z=Z_km.astype(float), showscale=False,
                colorscale=[[i/(k-1 or 1), hex_rgba(CHART_COLORS[i % len(CHART_COLORS)], 0.19)]
                            for i in range(k)],
                contours=dict(showlines=True, coloring="fill")))

        for i in range(k):
            mask = labels_km == i
            fig_km.add_trace(go.Scatter(
                x=X_km_s[mask, 0], y=X_km_s[mask, 1], mode="markers",
                marker=dict(color=CHART_COLORS[i % len(CHART_COLORS)], size=6, opacity=0.8),
                name=f"Cluster {i}"))

        fig_km.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode="markers",
            marker=dict(symbol="x", size=14, color="white",
                        line=dict(width=2, color="white")),
            name="Centroids"))

        fig_km.update_layout(template=PLOTLY_TEMPLATE, height=400,
                              title=f"K-Means  (K={k}, inertia={inertia:.1f})",
                              xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig_km, use_container_width=True)

    # Elbow chart
    st.markdown("#### Elbow Method — choosing K")
    max_k = min(10, n_km // 20)
    ks    = range(1, max_k + 1)
    inertias = [KMeans(n_clusters=ki, random_state=0, n_init=5).fit(X_km_s).inertia_
                for ki in ks]

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(ks), y=inertias, mode="lines+markers",
                                    line=dict(color=PURPLE, width=2),
                                    marker=dict(color=CYAN, size=8)))
    fig_elbow.add_vline(x=k, line_color=AMBER, line_dash="dash",
                         annotation_text=f"K={k}", annotation_position="top right")
    fig_elbow.update_layout(template=PLOTLY_TEMPLATE, height=280,
                             title="Inertia vs K  (elbow = optimal K)",
                             xaxis_title="K", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)

    divider()
    card("K-Means limitations",
         "Assumes <strong>spherical, equal-sized</strong> clusters. Sensitive to outliers. "
         "Requires K upfront. Use the elbow method or silhouette score to pick K. "
         "For non-convex shapes, try DBSCAN or Gaussian Mixture Models instead.",
         badge="Centroid-based", badge_style="purple")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PCA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section("Principal Component Analysis (PCA)")
    info("PCA finds the directions (principal components) of maximum variance in the data. "
         "It projects high-dimensional data to a lower-dimensional space, preserving as much variance as possible.")

    c_ctrl2, c_plot2 = st.columns([1, 2])

    with c_ctrl2:
        n_feat   = st.slider("Original features",  3, 20, 8, key="pca_feat")
        n_samp   = st.slider("Samples",           100, 500, 200, key="pca_n")
        n_comp   = st.slider("Components to keep", 1, min(n_feat, 10), 2, key="pca_comp")
        noise_pca = st.slider("Feature noise",    0.1, 2.0, 0.5, 0.1, key="pca_noise")
        seed_pca  = st.number_input("Seed", 0, 999, 3, key="pca_seed")

    rng_pca = np.random.default_rng(int(seed_pca))
    # Create structured data (a few latent factors + noise)
    n_latent = max(2, n_feat // 3)
    latent = rng_pca.standard_normal((n_samp, n_latent))
    W = rng_pca.standard_normal((n_latent, n_feat))
    X_pca = latent @ W + rng_pca.normal(0, noise_pca, (n_samp, n_feat))
    X_pca_s = StandardScaler().fit_transform(X_pca)

    pca_full = PCA().fit(X_pca_s)
    pca_red  = PCA(n_components=n_comp).fit_transform(X_pca_s)
    ev_ratio = pca_full.explained_variance_ratio_
    cumvar   = np.cumsum(ev_ratio)

    with c_plot2:
        # Scree plot
        fig_scree = go.Figure()
        fig_scree.add_bar(x=[f"PC{i+1}" for i in range(len(ev_ratio))],
                          y=ev_ratio * 100,
                          marker_color=CHART_COLORS[:len(ev_ratio)],
                          name="Individual")
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(ev_ratio))],
            y=cumvar * 100, mode="lines+markers",
            line=dict(color=AMBER, width=2),
            marker=dict(color=AMBER, size=7),
            name="Cumulative", yaxis="y2"))
        fig_scree.add_hline(y=90, line_color=ROSE, line_dash="dash",
                             annotation_text="90% threshold")
        fig_scree.add_vline(x=n_comp - 0.5, line_color=CYAN, line_dash="dot",
                             annotation_text=f"Keep {n_comp}")
        fig_scree.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            title="Scree Plot — Explained Variance per Component",
            xaxis_title="Component", yaxis_title="Variance Explained (%)",
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                        range=[0, 105]),
            legend=dict(x=0.5, y=1.15, orientation="h"))
        st.plotly_chart(fig_scree, use_container_width=True)

    # 2D projection scatter
    if n_comp >= 2:
        fig_proj = px.scatter(x=pca_red[:, 0], y=pca_red[:, 1],
                              labels={"x": "PC1", "y": "PC2"},
                              title=f"2D Projection onto PC1 & PC2  ({cumvar[1]:.1%} variance)",
                              template=PLOTLY_TEMPLATE,
                              color_discrete_sequence=[PURPLE])
        fig_proj.update_traces(marker=dict(size=5, opacity=0.7))
        fig_proj.update_layout(height=320)
        st.plotly_chart(fig_proj, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Variance in {n_comp} PCs", f"{cumvar[n_comp-1]:.1%}")
    m2.metric("PCs for 90% variance",
              str(np.searchsorted(cumvar, 0.90) + 1))
    m3.metric("Compression ratio",
              f"{n_feat} → {n_comp}  ({100*(1-n_comp/n_feat):.0f}% reduction)")

    divider()
    card("PCA is a linear transformation",
         "Each principal component is an <strong>eigenvector</strong> of the covariance matrix, "
         "sorted by eigenvalue (variance). PCA is unsupervised — it ignores labels. "
         "Use it for <strong>visualisation</strong>, noise reduction, and speeding up downstream models. "
         "For non-linear structure, try t-SNE or UMAP.",
         badge="Dimensionality Reduction", badge_style="cyan")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DBSCAN
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section("DBSCAN")
    info("DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together, "
         "marking points in low-density regions as outliers. Works on any cluster shape.")

    c_ctrl3, c_plot3 = st.columns([1, 2])

    with c_ctrl3:
        dataset3 = st.selectbox("Dataset shape", ["Moons", "Circles", "Blobs"], key="db_data")
        n3       = st.slider("Samples",    100, 500, 250, key="db_n")
        eps3     = st.slider("ε (radius)", 0.05, 1.0, 0.3, 0.01, key="db_eps")
        min_s3   = st.slider("min_samples", 2, 20, 5, key="db_min")
        noise3   = st.slider("Noise",     0.01, 0.3, 0.05, 0.01, key="db_noise")
        seed3    = st.number_input("Seed", 0, 999, 0, key="db_seed")

    if dataset3 == "Moons":
        X3, _ = make_moons(n_samples=n3, noise=noise3, random_state=int(seed3))
    elif dataset3 == "Circles":
        X3, _ = make_circles(n_samples=n3, noise=noise3, factor=0.5,
                              random_state=int(seed3))
    else:
        X3, _ = make_blobs(n_samples=n3, centers=4, cluster_std=noise3*3,
                            random_state=int(seed3))

    X3s = StandardScaler().fit_transform(X3)
    db  = DBSCAN(eps=eps3, min_samples=min_s3).fit(X3s)
    labels3   = db.labels_
    n_clusters3 = len(set(labels3)) - (1 if -1 in labels3 else 0)
    n_noise3   = (labels3 == -1).sum()

    palette = CHART_COLORS + ["#ffffff"] * 20

    with c_plot3:
        fig3 = go.Figure()
        for lbl in sorted(set(labels3)):
            mask = labels3 == lbl
            if lbl == -1:
                fig3.add_trace(go.Scatter(
                    x=X3s[mask, 0], y=X3s[mask, 1], mode="markers",
                    marker=dict(color=ROSE, size=6, symbol="x", opacity=0.6),
                    name="Noise / Outliers"))
            else:
                fig3.add_trace(go.Scatter(
                    x=X3s[mask, 0], y=X3s[mask, 1], mode="markers",
                    marker=dict(color=palette[lbl % len(palette)], size=6, opacity=0.85),
                    name=f"Cluster {lbl}"))
        fig3.update_layout(template=PLOTLY_TEMPLATE, height=420,
                            title=f"DBSCAN — {n_clusters3} clusters, {n_noise3} noise points  (ε={eps3}, min={min_s3})",
                            xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig3, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters found",  str(n_clusters3))
    m2.metric("Noise points",    str(n_noise3))
    m3.metric("Noise %",         f"{100*n_noise3/n3:.1f}%")

    divider()
    card("DBSCAN vs K-Means",
         "DBSCAN does <strong>not</strong> require K upfront and can find arbitrarily shaped clusters. "
         "It also identifies <strong>outliers</strong> as noise. "
         "<strong>ε</strong>: neighbourhood radius — too small → everything is noise; too large → one big cluster. "
         "<strong>min_samples</strong>: minimum points to form a core point. "
         "Tip: use a k-distance plot to choose ε.",
         badge="Density-based", badge_style="rose")

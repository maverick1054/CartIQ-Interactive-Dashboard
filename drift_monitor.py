"""
CartIQ Module 7: Data Drift & Concept Drift Monitoring Engine
Supports:
1. Population Stability Index (PSI) for continuous & categorical features
2. Kolmogorov-Smirnov (KS) 2-sample test for numerical feature drift
3. Chi-square distribution tests for categorical feature shift
4. Interactive drift report generation and visual profiling
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

COLOR_PRIMARY = "#2563EB"
COLOR_SUCCESS = "#10B981"
COLOR_DANGER = "#EF4444"
COLOR_WARNING = "#F59E0B"
PLOT_BG = "#FFFFFF"
GRID_COLOR = "#E2E8F0"

def calculate_psi_numeric(reference_series, current_series, num_bins=10, epsilon=1e-4):
    ref = reference_series.dropna().values
    curr = current_series.dropna().values

    if len(ref) == 0 or len(curr) == 0:
        return 0.0, []

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0, []

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    ref_pct = (ref_counts / len(ref)) + epsilon
    curr_pct = (curr_counts / len(curr)) + epsilon

    ref_pct = ref_pct / np.sum(ref_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    total_psi = float(np.sum(psi_values))

    return total_psi, []

def calculate_psi_categorical(reference_series, current_series, epsilon=1e-4):
    ref_counts = reference_series.value_counts(dropna=False)
    curr_counts = current_series.value_counts(dropna=False)

    all_cats = list(set(ref_counts.index).union(set(curr_counts.index)))
    ref_total = len(reference_series)
    curr_total = len(current_series)

    if ref_total == 0 or curr_total == 0:
        return 0.0, []

    ref_pcts = np.array([(ref_counts.get(cat, 0) / ref_total) + epsilon for cat in all_cats])
    curr_pcts = np.array([(curr_counts.get(cat, 0) / curr_total) + epsilon for cat in all_cats])

    ref_pcts = ref_pcts / np.sum(ref_pcts)
    curr_pcts = curr_pcts / np.sum(curr_pcts)

    psi_values = (curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts)
    total_psi = float(np.sum(psi_values))

    return total_psi, []

def detect_feature_drift(reference_df, current_df, feature_name):
    is_numeric = pd.api.types.is_numeric_dtype(reference_df[feature_name])

    if is_numeric:
        psi_val, _ = calculate_psi_numeric(reference_df[feature_name], current_df[feature_name])
        r_clean = reference_df[feature_name].dropna().values
        c_clean = current_df[feature_name].dropna().values
        ks_stat, p_value = stats.ks_2samp(r_clean, c_clean)
        test_type = "Kolmogorov-Smirnov"
    else:
        psi_val, _ = calculate_psi_categorical(reference_df[feature_name], current_df[feature_name])
        ref_vc = reference_df[feature_name].value_counts()
        curr_vc = current_df[feature_name].value_counts()
        common_idx = ref_vc.index.intersection(curr_vc.index)
        if len(common_idx) > 1:
            obs = curr_vc.reindex(common_idx, fill_value=1).values
            exp = ref_vc.reindex(common_idx, fill_value=1).values
            exp = exp / np.sum(exp) * np.sum(obs)
            chi_res = stats.chisquare(f_obs=obs, f_exp=exp)
            ks_stat = float(chi_res.statistic)
            p_value = float(chi_res.pvalue)
        else:
            ks_stat, p_value = 0.0, 1.0
        test_type = "Chi-Square"

    if psi_val < 0.1:
        status = "Stable (No Drift)"
    elif psi_val < 0.25:
        status = "Moderate Shift (Warning)"
    else:
        status = "Significant Drift (Alert)"

    return {
        "feature": feature_name,
        "type": "Numeric" if is_numeric else "Categorical",
        "psi": float(psi_val),
        "test_name": test_type,
        "test_statistic": float(ks_stat),
        "p_value": float(p_value),
        "status": status
    }

def run_dataset_drift_report(reference_df, current_df, features=None):
    if features is None:
        features = [c for c in reference_df.columns if c in current_df.columns and not c.startswith("is_")]

    results = []
    drifted_count = 0
    for f in features:
        res = detect_feature_drift(reference_df, current_df, f)
        results.append(res)
        if res["psi"] >= 0.1:
            drifted_count += 1

    ratio = drifted_count / max(len(features), 1)
    status_label = "CRITICAL DRIFT: Retraining Triggered" if ratio >= 0.4 else ("WARNING: Moderate Drift" if ratio > 0.1 else "HEALTHY: Stable Distribution")

    return {
        "total_features": len(features),
        "drifted_count": drifted_count,
        "overall_status": status_label,
        "feature_reports": results
    }

def plot_drift_psi_summary(report):
    feats = [r["feature"] for r in report["feature_reports"]]
    psis = [r["psi"] for r in report["feature_reports"]]
    colors = [COLOR_SUCCESS if p < 0.1 else (COLOR_WARNING if p < 0.25 else COLOR_DANGER) for p in psis]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=psis, y=feats, orientation="h", marker=dict(color=colors), text=[f"PSI: {p:.4f}" for p in psis], textposition="outside"))
    fig.add_vline(x=0.1, line_dash="dash", line_color=COLOR_WARNING, annotation_text="Warning (0.1)")
    fig.add_vline(x=0.25, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Action Required (0.25)")

    fig.update_layout(
        title="CartIQ Feature Population Stability Index (PSI) Profile",
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(title="PSI Score", gridcolor=GRID_COLOR),
        yaxis=dict(title="Feature Name", gridcolor=GRID_COLOR),
        height=450
    )
    return fig

def plot_feature_drift_distribution(reference_df, current_df, feature_name):
    is_numeric = pd.api.types.is_numeric_dtype(reference_df[feature_name])
    fig = go.Figure()

    if is_numeric:
        fig.add_trace(go.Histogram(x=reference_df[feature_name].dropna(), name="Baseline Reference", opacity=0.6, histnorm="probability density", marker=dict(color=COLOR_PRIMARY)))
        fig.add_trace(go.Histogram(x=current_df[feature_name].dropna(), name="Production Batch", opacity=0.6, histnorm="probability density", marker=dict(color="#F97316")))
        fig.update_layout(barmode="overlay")
    else:
        ref_pct = reference_df[feature_name].value_counts(normalize=True) * 100
        curr_pct = current_df[feature_name].value_counts(normalize=True) * 100
        cats = sorted(list(set(ref_pct.index).union(set(curr_pct.index))))
        fig.add_trace(go.Bar(x=cats, y=[ref_pct.get(c, 0) for c in cats], name="Baseline (%)", marker=dict(color=COLOR_PRIMARY)))
        fig.add_trace(go.Bar(x=cats, y=[curr_pct.get(c, 0) for c in cats], name="Production Batch (%)", marker=dict(color="#F97316")))
        fig.update_layout(barmode="group")

    fig.update_layout(
        title=f"Feature Distribution Drift: {feature_name}",
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(title=feature_name, gridcolor=GRID_COLOR),
        yaxis=dict(title="Density / Share (%)", gridcolor=GRID_COLOR),
        height=420
    )
    return fig

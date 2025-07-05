# --- SME OVERHAUL: Definitive, Compliance-Focused, and Unabridged Version (PMA-Focused V4-Final) ---
"""
Plotting utilities for creating standardized, publication-quality visualizations.

This module contains functions that generate various Plotly figures
used throughout the GenomicsDx dashboard. It is augmented with a comprehensive
suite of specialized functions for creating plots essential for analytical
validation (AV), clinical validation (CV), quality control (QC), and process
monitoring for a genomic diagnostic, ensuring a consistent, professional,
and compliant visual style.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Optional, Tuple, Any
import io

# --- Third-party Imports ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats

# --- Setup Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# --- MODULE-LEVEL CONFIGURATION CONSTANTS ---
# ==============================================================================
_PLOT_LAYOUT_CONFIG: Dict[str, Any] = {
    "margin": dict(l=50, r=30, t=80, b=50),
    "title_x": 0.5,
    "font": {"family": "Arial, sans-serif", "size": 12},
    "template": "plotly_white"
}

_RISK_CONFIG: Dict[str, Any] = {
    'levels': {
        (1, 1): 'Low', (1, 2): 'Low', (1, 3): 'Medium', (1, 4): 'Medium', (1, 5): 'High',
        (2, 1): 'Low', (2, 2): 'Low', (2, 3): 'Medium', (2, 4): 'High', (2, 5): 'High',
        (3, 1): 'Medium', (3, 2): 'Medium', (3, 3): 'High', (3, 4): 'High', (3, 5): 'Unacceptable',
        (4, 1): 'Medium', (4, 2): 'High', (4, 3): 'High', (4, 4): 'Unacceptable', (4, 5): 'Unacceptable'},
    'colors': {
        'Unacceptable': 'rgba(139, 0, 0, 0.9)', 'High': 'rgba(214, 39, 40, 0.9)',
        'Medium': 'rgba(255, 127, 14, 0.9)', 'Low': 'rgba(44, 160, 44, 0.9)'},
    'order': ['Low', 'Medium', 'High', 'Unacceptable']
}

def _create_placeholder_figure(text: str, title: str, icon: str = "ℹ️") -> go.Figure:
    """Creates a standardized, empty figure with an icon and text annotation."""
    fig = go.Figure()
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[{'text': f"{icon}<br>{text}", 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16, 'color': '#7f7f7f'}}],
        height=300, **_PLOT_LAYOUT_CONFIG
    )
    return fig

# --- REUSABLE DATA CLEANING UTILITY ---
def _clean_df_for_model(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """A robust, reusable function to clean a DataFrame before statistical modeling."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df_clean = df.copy()
    for col in required_cols:
        if col not in df_clean.columns:
            logger.warning(f"Required column '{col}' for model cleaning not found in DataFrame.")
            return pd.DataFrame()
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(subset=required_cols, inplace=True)
    return df_clean

# ==============================================================================
# --- GENERAL PURPOSE PLOTTING FUNCTIONS ---
# ... (These functions are unchanged and omitted for brevity) ...
def create_risk_profile_chart(hazards_df: pd.DataFrame) -> go.Figure: pass
def create_action_item_chart(actions_df: pd.DataFrame) -> go.Figure: pass
# ==============================================================================


# ==============================================================================
# --- SPECIALIZED GENOMICS, QC & ML PLOTS ---
# ==============================================================================

def create_roc_curve(df: pd.DataFrame, score_col: str, truth_col: str, title: str = "Receiver Operating Characteristic (ROC) Curve", fig: Optional[go.Figure] = None, name: Optional[str] = None) -> go.Figure:
    """Generates or adds to a ROC curve. A cornerstone of any PMA submission."""
    try:
        fpr, tpr, _ = roc_curve(df[truth_col], df[score_col])
        roc_auc = auc(fpr, tpr)
        
        if fig is None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='navy', width=2, dash='dash')))

        trace_name = f'AUC = {roc_auc:.4f}' if name is None else f'{name} (AUC={roc_auc:.3f})'
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=trace_name))
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title='1 - Specificity (False Positive Rate)',
            yaxis_title='Sensitivity (True Positive Rate)',
            xaxis=dict(range=[-0.01, 1.01]), yaxis=dict(range=[-0.01, 1.01]),
            legend=dict(x=0.5, y=0.1, xanchor='center', yanchor='bottom', bgcolor='rgba(255,255,255,0.6)'),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating ROC curve: {e}", exc_info=True)
        return _create_placeholder_figure("ROC Curve Error", title, icon="⚠️")

def create_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, title: str = "Precision-Recall Curve", fig: Optional[go.Figure] = None, name: Optional[str] = None) -> go.Figure:
    """Generates or adds to a Precision-Recall curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)
        
        if fig is None: fig = go.Figure()
            
        trace_name = f'AP = {avg_precision:.4f}' if name is None else f'{name} (AP={avg_precision:.3f})'
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=trace_name))
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title='Recall (Sensitivity)', yaxis_title='Precision (PPV)',
            xaxis=dict(range=[-0.01, 1.01]), yaxis=dict(range=[-0.01, 1.01]),
            legend=dict(x=0.1, y=0.1, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.6)'),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating PR curve: {e}", exc_info=True)
        return _create_placeholder_figure("PR Curve Error", title, icon="⚠️")

def create_lod_probit_plot(df: pd.DataFrame, conc_col: str, hit_rate_col: str, title: str = "Limit of Detection (LoD) by Probit Analysis") -> go.Figure:
    """Generates a Probit regression plot to determine the Limit of Detection (LoD)."""
    try:
        df_filtered = df.dropna(subset=[conc_col, hit_rate_col]).copy()
        df_filtered = df_filtered[df_filtered[conc_col] > 0]
        if len(df_filtered) < 2: return _create_placeholder_figure("Insufficient data for Probit plot.", title)

        df_filtered['probit_hit_rate'] = stats.norm.ppf(df_filtered[hit_rate_col].clip(0.001, 0.999))
        log_conc = np.log10(df_filtered[conc_col])
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_conc, df_filtered['probit_hit_rate'])
        
        lod_95_probit = stats.norm.ppf(0.95)
        lod_95 = 10**((lod_95_probit - intercept) / slope)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[conc_col], y=df_filtered[hit_rate_col], mode='markers', name='Observed Data', marker=dict(size=10, color='blue')))
        
        x_fit = np.logspace(np.log10(df_filtered[conc_col].min() * 0.5), np.log10(df_filtered[conc_col].max() * 2), 100)
        y_fit = stats.norm.cdf(intercept + slope * np.log10(x_fit))
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'Probit Fit (R²={r_val**2:.3f})', line=dict(color='red', width=3)))
        
        fig.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="95% Hit Rate", annotation_position="bottom right")
        fig.add_vline(x=lod_95, line_dash="dash", line_color="black", annotation_text=f"LoD = {lod_95:.4f}", annotation_position="top left")
        
        fig.update_layout(title=f"<b>{title}</b>", xaxis_title=f"Analyte Concentration ({conc_col})", yaxis_title="Detection Rate (Hit Rate)", xaxis_type="log", yaxis=dict(range=[0, 1.05], tickformat=".0%"), **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating LoD Probit plot: {e}", exc_info=True)
        return _create_placeholder_figure("Probit Plot Error", title, icon="⚠️")

def create_levey_jennings_plot(spc_data: Dict[str, Any]) -> go.Figure:
    """Creates a Levey-Jennings chart for laboratory quality control monitoring."""
    title = "<b>Levey-Jennings Chart: Assay Control Monitoring</b>"
    try:
        if not spc_data or 'measurements' not in spc_data: return _create_placeholder_figure("SPC data is incomplete or missing.", title, "📊")
        meas = np.array(spc_data['measurements']); mu = spc_data.get('target', meas.mean()); sigma = spc_data.get('stdev', meas.std())
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=meas, name='Control Value', mode='lines+markers', line=dict(color='#1f77b4'), marker=dict(size=8)))
        fig.add_hline(y=mu, line_dash="solid", line_color="green", annotation_text=f"Mean: {mu:.2f}")
        for i, color in zip([1, 2, 3], ["orange", "orange", "red"]):
            fig.add_hline(y=mu + i*sigma, line_dash="dash", line_color=color, annotation_text=f"+{i}SD")
            fig.add_hline(y=mu - i*sigma, line_dash="dash", line_color=color, annotation_text=f"-{i}SD")
        fig.update_layout(title=title, yaxis_title="Measured Value", xaxis_title="Run Number", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating Levey-Jennings chart: {e}", exc_info=True)
        return _create_placeholder_figure("Levey-Jennings Chart Error", title, icon="⚠️")

def create_bland_altman_plot(df: pd.DataFrame, method1_col: str, method2_col: str, title: str = "Bland-Altman Agreement Plot") -> go.Figure:
    """Generates a Bland-Altman plot to assess agreement between two measurement methods."""
    try:
        df_val = df[[method1_col, method2_col]].dropna().copy()
        if len(df_val) < 2: return _create_placeholder_figure("Insufficient data for Bland-Altman plot.", title)
        df_val['average'] = (df_val[method1_col] + df_val[method2_col]) / 2; df_val['difference'] = df_val[method1_col] - df_val[method2_col]
        mean_diff = df_val['difference'].mean(); std_diff = df_val['difference'].std(); upper_loa = mean_diff + 1.96 * std_diff; lower_loa = mean_diff - 1.96 * std_diff
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_val['average'], y=df_val['difference'], mode='markers', name='Differences', marker=dict(color='rgba(31, 119, 180, 0.7)')))
        fig.add_hline(y=mean_diff, line=dict(color='blue', width=3, dash='dash'), name=f'Mean Diff: {mean_diff:.3f}')
        fig.add_hline(y=upper_loa, line=dict(color='red', width=2, dash='dash'), name=f'Upper LoA: {upper_loa:.3f}')
        fig.add_hline(y=lower_loa, line=dict(color='red', width=2, dash='dash'), name=f'Lower LoA: {lower_loa:.3f}')
        fig.update_layout(title=f"<b>{title}</b><br>({method1_col} vs. {method2_col})", xaxis_title="Average of Measurements", yaxis_title="Difference between Measurements", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating Bland-Altman plot: {e}", exc_info=True)
        return _create_placeholder_figure("Bland-Altman Plot Error", title, icon="⚠️")

def create_pareto_chart(df: pd.DataFrame, category_col: str, title: str) -> go.Figure:
    """Creates a Pareto chart for identifying the most frequent categories."""
    try:
        if df.empty or category_col not in df.columns: return _create_placeholder_figure("No data for Pareto analysis.", title)
        counts = df[category_col].value_counts(); pareto_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
        pareto_df = pareto_df.sort_values(by='Count', ascending=False); pareto_df['Cumulative Percentage'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df['Category'], y=pareto_df['Count'], name='Count', marker_color='cornflowerblue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df['Category'], y=pareto_df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)
        fig.update_layout(title_text=f"<b>{title}</b>", **_PLOT_LAYOUT_CONFIG); fig.update_yaxes(title_text="Count", secondary_y=False); fig.update_yaxes(title_text="Cumulative Percentage (%)", range=[0, 101], secondary_y=True)
        return fig
    except Exception as e:
        logger.error(f"Error creating Pareto chart: {e}", exc_info=True)
        return _create_placeholder_figure("Pareto Chart Error", title, icon="⚠️")
        
def create_gauge_rr_plot(df: pd.DataFrame, part_col: str, operator_col: str, value_col: str) -> Tuple[go.Figure, pd.DataFrame]:
    """Performs Gauge R&R analysis and returns a summary plot and results table."""
    title = "<b>Measurement System Analysis (Gauge R&R)</b>"; results_df = pd.DataFrame(columns=['Source', 'Variance Component', '% Contribution']).set_index('Source')
    try:
        ### BUG FIX: Apply robust data cleaning before modeling
        required_cols = [part_col, operator_col, value_col]
        df_clean = _clean_df_for_model(df, required_cols)
        if len(df_clean) < 4: return _create_placeholder_figure("Insufficient clean data for Gauge R&R.", title, "⚠️"), results_df

        formula = f"`{value_col}` ~ C(`{operator_col}`) + C(`{part_col}`) + C(`{operator_col}`):C(`{part_col}`)"; model = ols(formula, data=df_clean).fit()
        anova_table = anova_lm(model, typ=2)
        
        ms_op = anova_table.loc[f"C(`{operator_col}`)", 'mean_sq']; ms_part = anova_table.loc[f"C(`{part_col}`)", 'mean_sq']; ms_interact = anova_table.loc[f"C(`{operator_col}`):C(`{part_col}`)", 'mean_sq']; ms_error = anova_table.loc['Residual', 'mean_sq']
        n_parts = df_clean[part_col].nunique(); n_ops = df_clean[operator_col].nunique(); reps = len(df_clean) / (n_parts * n_ops)
        
        var_repeat = ms_error; var_repro = max(0, (ms_op - ms_interact) / (n_parts * reps)); var_interact = max(0, (ms_interact - ms_error) / reps); var_repro += var_interact; var_part = max(0, (ms_part - ms_interact) / (n_ops * reps))
        var_grr = var_repeat + var_repro; total_var = var_grr + var_part
        
        results_data = {'Source': ['Total Gauge R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total Variation'],'Variance Component': [var_grr, var_repeat, var_repro, var_part, total_var],
                        '% Contribution': [(v / total_var * 100) if total_var > 0 else 0 for v in [var_grr, var_repeat, var_repro, var_part, total_var]]}
        results_df = pd.DataFrame(results_data).set_index('Source')
        
        fig = go.Figure(go.Bar(x=results_df['% Contribution'], y=results_df.index, orientation='h', marker_color=['crimson', 'lightcoral', 'lightsalmon', 'lightseagreen', 'grey']))
        fig.update_layout(title_text=title, xaxis_title="% Contribution to Total Variation", **_PLOT_LAYOUT_CONFIG)
        return fig, results_df.round(4)
    except Exception as e:
        logger.error(f"Error creating Gauge R&R plot: {e}", exc_info=True)
        return _create_placeholder_figure("Gauge R&R Error", title, "⚠️"), results_df

def create_tost_plot(a: np.ndarray, b: np.ndarray, low: float, high: float) -> Tuple[go.Figure, float]:
    """Performs TOST and returns a plot and the max p-value."""
    title = "<b>Equivalence Test (TOST) Results</b>"; p_value = 1.0
    try:
        diff = a.mean() - b.mean(); n1, n2 = len(a), len(b); s_pool = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2)); se_diff = s_pool * np.sqrt(1/n1 + 1/n2)
        t_stat1 = (diff - low) / se_diff; t_stat2 = (diff - high) / se_diff
        p1 = stats.t.sf(t_stat1, df=n1 + n2 - 2); p2 = stats.t.cdf(t_stat2, df=n1 + n2 - 2); p_value = max(p1, p2)
        ci_90 = stats.t.interval(0.90, df=n1+n2-2, loc=diff, scale=se_diff)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[ci_90[0], ci_90[1]], y=[1, 1], mode='lines', line=dict(color='blue', width=5), name='90% CI of Difference'))
        fig.add_trace(go.Scatter(x=[diff], y=[1], mode='markers', marker=dict(color='blue', size=12, symbol='x'), name='Mean Difference'))
        fig.add_shape(type='line', x0=low, y0=0, x1=low, y1=2, line=dict(color='red', dash='dash'), name='Lower Equivalence Bound')
        fig.add_shape(type='line', x0=high, y0=0, x1=high, y1=2, line=dict(color='red', dash='dash'), name='Upper Equivalence Bound')
        fig.update_layout(title=title, yaxis_visible=False, xaxis_title="Difference in Means", **_PLOT_LAYOUT_CONFIG); fig.update_xaxes(range=[min(low*1.2, ci_90[0]*1.2), max(high*1.2, ci_90[1]*1.2)])
        return fig, p_value
    except Exception as e:
        logger.error(f"Error creating TOST plot: {e}", exc_info=True)
        return _create_placeholder_figure("TOST Plot Error", title, "⚠️"), p_value

def create_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """Creates a professional heatmap for a confusion matrix."""
    try:
        fig = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, text=[[str(y) for y in x] for x in cm], texttemplate="%{text}", colorscale='Blues', showscale=False))
        fig.update_layout(title_text="<b>Confusion Matrix</b>", xaxis_title="Predicted Label", yaxis_title="True Label", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating confusion matrix heatmap: {e}", exc_info=True)
        return _create_placeholder_figure("Confusion Matrix Error", "Confusion Matrix", "⚠️")

def create_shap_summary_plot(shap_values: np.ndarray, features: pd.DataFrame) -> Optional[io.BytesIO]:
    """Creates a SHAP summary plot and returns it as an in-memory PNG image buffer."""
    try:
        import shap; import matplotlib.pyplot as plt
        # Defensive check
        if shap_values.shape[1] != features.shape[1]:
            logger.error(f"SHAP plot error: Mismatch in shapes. SHAP values have {shap_values.shape[1]} features, data has {features.shape[1]}.")
            return None
        fig, ax = plt.subplots(); shap.summary_plot(shap_values, features, show=False); plt.title("SHAP Feature Importance Summary", fontsize=16); plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {e}", exc_info=True)
        return None

def create_forecast_plot(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Creates a time series forecast plot."""
    title = "<b>Time Series Forecast</b>"
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df.index, y=history_df.iloc[:, 0], name='Historical Data', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name='Forecast', line=dict(color='darkorange', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
        fig.update_layout(title_text=title, xaxis_title="Date", yaxis_title="Value", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating forecast plot: {e}", exc_info=True)
        return _create_placeholder_figure("Forecast Plot Error", title, "⚠️")

### NEW PLOTTING FUNCTIONS ###

def create_power_analysis_plot(effect_size: float, alpha: float) -> go.Figure:
    """Generates a power curve plot showing power vs. sample size."""
    try:
        power_analysis = TTestIndPower()
        sample_sizes = np.arange(10, 501, 10)
        powers = [power_analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0, alternative='two-sided') for n in sample_sizes]
        fig = go.Figure(data=go.Scatter(x=sample_sizes, y=powers, mode='lines+markers', name=f'Power Curve (d={effect_size}, α={alpha})'))
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% Power Target")
        fig.update_layout(title="<b>Power vs. Sample Size</b>", xaxis_title="Sample Size (per group)", yaxis_title="Statistical Power (1 - β)", yaxis=dict(range=[0,1.05]), **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating power analysis plot: {e}", exc_info=True)
        return _create_placeholder_figure("Power Plot Error", "Power vs. Sample Size", "⚠️")

def create_kaplan_meier_plot(kmf1: Any, kmf2: Any, p_value: float) -> Optional[io.BytesIO]:
    """Creates a Kaplan-Meier plot and returns it as an in-memory PNG buffer."""
    try:
        from lifelines.plotting import add_at_risk_counts
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        kmf1.plot(ax=ax, ci_show=True)
        kmf2.plot(ax=ax, ci_show=True)
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.title(f"Kaplan-Meier Survival Curves (Log-Rank p={p_value:.3f})")
        plt.xlabel("Time (Days)")
        plt.ylabel("Survival Probability")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        logger.error(f"Error creating Kaplan-Meier plot: {e}", exc_info=True)
        return None

def create_distribution_comparison_plot(dist1: np.ndarray, dist2: np.ndarray, feature_name: str) -> go.Figure:
    """Creates a histogram/KDE plot comparing two distributions for data drift analysis."""
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dist1, name='Training Data', histnorm='probability density', opacity=0.75, marker_color='#1f77b4'))
        fig.add_trace(go.Histogram(x=dist2, name='Production Data', histnorm='probability density', opacity=0.75, marker_color='#ff7f0e'))
        fig.update_layout(barmode='overlay', title=f"<b>Data Drift Analysis for '{feature_name}'</b>", xaxis_title="Feature Value", yaxis_title="Density", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating distribution comparison plot: {e}", exc_info=True)
        return _create_placeholder_figure("Distribution Plot Error", "Data Drift Analysis", "⚠️")

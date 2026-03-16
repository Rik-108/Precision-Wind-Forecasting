import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Precision Wind | Asset Operations Center",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for "Executive" feel with Tooltip-like help
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-weight: 700;
        color: #333;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 5px solid #00a8cc;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #0f1116;
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv('Tableau_Wind_Forecast_Final.csv')
    except FileNotFoundError:
        st.error("🚨 Critical Error: 'Tableau_Wind_Forecast_Final.csv' not found.")
        st.stop()

    # 1. Date Parsing
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d-%m-%Y %H:%M')

    # 2. Renaming
    df.rename(columns={
        'LV ActivePower (kW)': 'Actual',
        'Predicted_Power_kW': 'Predicted',
        'Theoretical_Power_Curve (KWh)': 'Baseline',
        'Wind Speed (m/s)': 'WindSpeed'
    }, inplace=True)

    # 3. Core Calculations
    df['Bias_Value'] = df['Predicted'] - df['Actual']
    df['Abs_Error_Model'] = df['Bias_Value'].abs()
    df['Abs_Error_Baseline'] = (df['Baseline'] - df['Actual']).abs()

    # 4. Trading Logic
    safe_threshold = 360  # 10% of Capacity
    df['Reliability_Status'] = df['Abs_Error_Model'].apply(
        lambda x: "Safe Bid (±10%)" if x <= safe_threshold else "High Risk (Hedge Needed)"
    )

    # 5. Physics Logic (Ramps)
    # Ramp = Change > 15% of Capacity (540kW) in 10 mins
    ramp_threshold = 540
    df['Ramp_Actual'] = df['Actual'] - df['Lag_1_Power']
    df['Is_Ramp_Event'] = df['Ramp_Actual'].abs() > ramp_threshold

    # 6. Rolling Stability (24h Window - 144 data points @ 10min intervals)
    df['Rolling_RMSE'] = df['Bias_Value'].rolling(window=144).apply(lambda x: np.sqrt((x ** 2).mean()))

    # 7. Wind Bins
    bins = [0, 3, 7, 12, 25, 50]
    labels = ['Idle', 'Start-up', 'Power Growth', 'Full Load', 'Storm']
    df['WindBin'] = pd.cut(df['WindSpeed'], bins=bins, labels=labels)

    return df


df = load_and_prep_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ Settings")
st.sidebar.markdown("---")

# Date Filter
min_date = df['Date/Time'].min().date()
max_date = df['Date/Time'].max().date()
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("From", min_date)
end_date = col2.date_input("To", max_date)

mask = (df['Date/Time'].dt.date >= start_date) & (df['Date/Time'].dt.date <= end_date)
df_filtered = df.loc[mask]

st.sidebar.markdown(f"**Records:** {len(df_filtered)}")
st.sidebar.markdown("---")
st.sidebar.info("Operational Status: **ONLINE** 🟢")

# -----------------------------------------------------------------------------
# 4. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
st.title("⚡ Precision Wind | Asset Operations Center")
st.markdown("### Executive Performance Summary")

# --- ROW 1: EXECUTIVE KPIs ---
mae_model = df_filtered['Abs_Error_Model'].mean()
safe_bid_pct = (df_filtered['Reliability_Status'] == "Safe Bid (±10%)").mean() * 100
total_error_saved = (df_filtered['Abs_Error_Baseline'].sum() - df_filtered['Abs_Error_Model'].sum())
est_savings = total_error_saved * 0.05

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Reliability Rate", f"{safe_bid_pct:.1f}%", "Safe Bids (Target > 85%)")
kpi2.metric("MAE Precision", f"{mae_model:.1f} kW", "Avg Miss per Hour")
kpi3.metric("Est. Financial Savings", f"${est_savings:,.0f}", "Penalty Avoidance")
kpi4.metric("Bias Direction", f"{df_filtered['Bias_Value'].mean():.1f} kW", "Avg Over/Under Forecast")

st.markdown("---")

# --- TABS ---
tab_trader, tab_eng, tab_risk = st.tabs([
    "📈 Trader (Operations)",
    "⚙️ Engineering (Physics)",
    "🛡️ Risk & Stability (Audit)"
])

# =============================================================================
# TAB 1: TRADER (OPERATIONS)
# =============================================================================
with tab_trader:
    st.markdown("""
    <div class="info-box">
    <b>Operational Context:</b> This view answers "How much can I trust today's forecast?" 
    Use the <b>Hedge Ratio</b> to decide how much volume to insure and the <b>Bias</b> to manually adjust bids.
    </div>
    """, unsafe_allow_html=True)

    # 1. Main Time Series
    st.subheader("1. Real-Time Performance: AI vs. Baseline")
    st.caption("Validating that the AI model (Green) tracks reality (Black) better than the old method (Red).")

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_filtered['Date/Time'], y=df_filtered['Actual'], name='Actual Power',
                                  line=dict(color='black', width=1)))
    fig_main.add_trace(go.Scatter(x=df_filtered['Date/Time'], y=df_filtered['Predicted'], name='AI Prediction',
                                  line=dict(color='#00CC96', width=2)))
    fig_main.add_trace(go.Scatter(x=df_filtered['Date/Time'], y=df_filtered['Baseline'], name='Old Baseline',
                                  line=dict(color='red', dash='dot', width=1)))
    fig_main.update_layout(height=400, template="plotly_white", yaxis_title="Power (kW)", xaxis_title="Time")
    st.plotly_chart(fig_main, use_container_width=True)

    # 2. Risk & Bias
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("2. Hedge Ratio (Risk Distribution)")
        st.caption("Green = Trade Aggressively. Red = Buy Insurance.")
        pie_data = df_filtered['Reliability_Status'].value_counts().reset_index()
        pie_data.columns = ['Status', 'Count']
        fig_pie = px.pie(pie_data, names='Status', values='Count', color='Status',
                         color_discrete_map={"Safe Bid (±10%)": "#00CC96", "High Risk (Hedge Needed)": "#EF553B"},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("3. Model Bias (Directional Error)")
        st.caption("Are we losing revenue (Under) or risking penalties (Over)?")
        fig_hist = px.histogram(df_filtered, x='Bias_Value', nbins=50, color_discrete_sequence=['#636EFA'])
        fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
        fig_hist.add_annotation(x=500, y=10, text="Over-Forecast (Penalty Risk)", font=dict(color="red"))
        fig_hist.add_annotation(x=-500, y=10, text="Under-Forecast (Lost Rev)", font=dict(color="orange"))
        st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# TAB 2: ENGINEERING (PHYSICS)
# =============================================================================
with tab_eng:
    st.markdown("""
    <div class="info-box">
    <b>Technical Context:</b> These metrics prove the AI model respects the laws of aerodynamics. 
    It ensures we don't predict power that is physically impossible to generate (e.g., during low wind start-up or storm cut-outs).
    </div>
    """, unsafe_allow_html=True)

    # 1. Power Curve Visualization
    st.subheader("1. The 'Power Curve' Validation")
    st.caption(
        "Visual proof that the AI (Green) captures the real physical limit (Black points) better than the Manufacturer's Theoretical Curve (Red).")

    fig_pc = go.Figure()
    # Scatter of Real Data
    fig_pc.add_trace(go.Scatter(
        x=df_filtered['WindSpeed'], y=df_filtered['Actual'],
        mode='markers', name='Actual Physics',
        marker=dict(size=3, color='rgba(0,0,0,0.1)'),  # Transparent black
    ))
    # Line of Manufacturer
    # We aggregate to get a clean line for the Theoretical Curve
    theo_agg = df_filtered.groupby(pd.cut(df_filtered['WindSpeed'], np.arange(0, 30, 0.5)))[
        'Baseline'].mean().reset_index()
    theo_agg['WindSpeed_Center'] = theo_agg['WindSpeed'].apply(lambda x: x.mid)

    fig_pc.add_trace(go.Scatter(
        x=theo_agg['WindSpeed_Center'], y=theo_agg['Baseline'],
        mode='lines', name='Theoretical Limit',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig_pc.update_layout(
        xaxis_title="Wind Speed (m/s)", yaxis_title="Power Output (kW)",
        title="Power Curve: Actual vs. Theoretical Limits",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    # 2. Ramp Event Analysis
    st.subheader("2. Ramp Event Capture (Grid Stability)")
    # --- CORRECTED LINE: Using Markdown bold syntax instead of HTML tags ---
    st.markdown(
        "**Why it matters:** Sudden power spikes (>500kW in 10 mins) endanger the grid. We must predict these 'Ramps' before they happen.")

    # Filter for the biggest ramp event in the selected window
    if df_filtered['Is_Ramp_Event'].sum() > 0:
        max_ramp_idx = df_filtered['Ramp_Actual'].abs().idxmax()
        window_center = df_filtered.loc[max_ramp_idx, 'Date/Time']
        ramp_window = df_filtered[
            (df_filtered['Date/Time'] >= window_center - pd.Timedelta(hours=4)) &
            (df_filtered['Date/Time'] <= window_center + pd.Timedelta(hours=4))
            ]

        fig_ramp = go.Figure()
        fig_ramp.add_trace(go.Scatter(x=ramp_window['Date/Time'], y=ramp_window['Actual'], name='Actual Power',
                                      line=dict(color='black', width=3)))
        fig_ramp.add_trace(go.Scatter(x=ramp_window['Date/Time'], y=ramp_window['Predicted'], name='AI Forecast',
                                      line=dict(color='#00CC96', width=2, dash='dot')))

        fig_ramp.update_layout(title=f"Zoom-In: Largest Ramp Event on {window_center.date()}", yaxis_title="Power (kW)")
        st.plotly_chart(fig_ramp, use_container_width=True)
    else:
        st.info("No major Ramp Events (>540kW change) detected in selected date range.")

    # 3. Binned Error Analysis
    st.subheader("3. Error by Wind Zone")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Start-up Phase (3-7 m/s):** High error here means the model doesn't understand friction. **Our AI minimizes this.**")
        binned_perf = df_filtered.groupby('WindBin')['Abs_Error_Model'].mean().reset_index()
        fig_bin = px.bar(binned_perf, x='WindBin', y='Abs_Error_Model', color='Abs_Error_Model',
                         color_continuous_scale='Blues', title="MAE by Wind Zone")
        st.plotly_chart(fig_bin, use_container_width=True)
    with col2:
        st.markdown(
            "**Storm Phase (>25 m/s):** The turbine cuts out for safety. The model must predict 0 kW, not 3600 kW.")
        cutout_df = df_filtered[df_filtered['WindSpeed'] > 24]
        if not cutout_df.empty:
            avg_cutout_err = cutout_df['Abs_Error_Model'].mean()
            st.metric("Avg Error During Storms", f"{avg_cutout_err:.1f} kW", "Target: < 100 kW")
        else:
            st.metric("Avg Error During Storms", "N/A", "No Storms in Period")

# =============================================================================
# TAB 3: RISK & STABILITY (AUDIT)
# =============================================================================
with tab_risk:
    st.markdown("""
    <div class="info-box">
    <b>Risk Context:</b> These metrics measure the "Honesty" and "Stability" of the model. 
    Risk Managers use this to calculate how much capital to hold in reserve for potential forecasting failures.
    </div>
    """, unsafe_allow_html=True)

    # 1. Rolling Stability
    st.subheader("1. Model Stability Over Time (Rolling Volatility)")
    st.caption(
        "Is the model's accuracy degrading? A spiking line indicates the model is struggling with recent weather patterns.")

    # Handle NaN values for rolling calculation visualization
    df_chart = df_filtered.dropna(subset=['Rolling_RMSE'])

    fig_roll = px.line(df_chart, x='Date/Time', y='Rolling_RMSE',
                       title="24-Hour Rolling RMSE (Lower is more stable)",
                       color_discrete_sequence=['#EF553B'])
    fig_roll.update_yaxes(title="Rolling Error (kW)")
    st.plotly_chart(fig_roll, use_container_width=True)

    # 2. Heteroscedasticity Check
    st.subheader("2. Error vs. Wind Speed (Risk Concentration)")
    st.caption("Do we only fail during storms? Ideally, errors should be low and flat across all wind speeds.")

    fig_het = px.scatter(df_filtered, x='WindSpeed', y='Bias_Value',
                         title="Residual Distribution by Wind Speed",
                         opacity=0.3, color_discrete_sequence=['#636EFA'])
    fig_het.add_hline(y=0, line_dash="dash", line_color="black")
    fig_het.update_yaxes(title="Prediction Error (kW)")
    st.plotly_chart(fig_het, use_container_width=True)

    # 3. Probability Density
    st.subheader("3. The 'Bell Curve' of Risk")
    # --- CORRECTED LINE: Using Markdown bold syntax instead of HTML tags ---
    st.markdown(
        "**Why it matters:** A perfect Bell Curve means valid, insurable risk. Skewed shapes indicate 'hidden' systematic failures.")

    c1, c2 = st.columns(2)
    with c1:
        fig_dist = px.histogram(df_filtered, x='Bias_Value', nbins=60, title="Error Distribution Histogram",
                                color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_dist, use_container_width=True)
    with c2:
        kurt_val = kurtosis(df_filtered['Bias_Value'])
        skew_val = skew(df_filtered['Bias_Value'])
        st.metric("Kurtosis (Fat Tails)", f"{kurt_val:.2f}", "Normal < 3.0")
        st.metric("Skewness (Bias)", f"{skew_val:.2f}", "Normal ~ 0.0")
        st.caption("High Kurtosis = Risk of 'Black Swan' events (extreme failures).")

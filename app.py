import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import custom project modules
from validator import run_data_qa_pipeline
from index_engine import BespokeIndexEngine

# =====================================================================
# 1. PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="SPDJI | Bespoke Index Engineering Sandbox",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 2. CACHED DATA INGESTION & FALLBACK ENGINE
# =====================================================================
@st.cache_data(ttl=3600, show_spinner="Fetching institutional market data...")
def load_market_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical close prices from yfinance.
    Automatically falls back to a synthetic dataset if offline or if API rate-limits occur.
    """
    try:
        raw_df = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(raw_df, pd.Series):
            raw_df = raw_df.to_frame()
        # Drop columns that returned entirely empty
        raw_df = raw_df.dropna(how="all", axis=1)
        if not raw_df.empty and len(raw_df) > 10:
            return raw_df
    except Exception as e:
        st.sidebar.warning(f"Live API unavailable ({e}). Switched to synthetic data.")
    
    # Synthetic fallback for offline development & testing
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(42)
    synth_data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + (i * 0.0001)
        vol = 0.012 + (i * 0.003)
        synth_data[t] = 100 * np.cumprod(1 + np.random.normal(drift, vol, len(dates)))
    
    df_synth = pd.DataFrame(synth_data, index=dates)
    # Inject synthetic anomalies to demonstrate the QA Audit Log feature
    if len(df_synth.columns) > 1:
        df_synth.iloc[15:18, 0] = np.nan   # 3 missing days
        df_synth.iloc[50, 1] *= 1.25       # 25% single-day price spike
    return df_synth

# =====================================================================
# 3. SIDEBAR CONTROLS
# =====================================================================
st.sidebar.title("🛠️ Index Configuration")
st.sidebar.markdown("Configure constituent basket and rebalancing rules.")

# A. Constituent Selection
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "PG", "JNJ"]
tickers_input = st.sidebar.text_input(
    "Constituents (Comma-separated tickers)",
    value=", ".join(default_tickers)
)
selected_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# B. Historical Backtest Window
end_date_default = date.today()
start_date_default = end_date_default - timedelta(days=730) # 2-year history
date_range = st.sidebar.date_input(
    "Historical Backtest Window",
    value=(start_date_default, end_date_default)
)

# C. Rebalance Schedule
rebal_freq = st.sidebar.selectbox(
    "Rebalancing Schedule",
    options=["QE", "ME", "YE"],
    format_func=lambda x: "Quarter-End (QE)" if x == "QE" else ("Month-End (ME)" if x == "ME" else "Year-End (YE)")
)

# D. Base Level
initial_level = st.sidebar.number_input("Base Index Level", value=1000.0, step=100.0)

# =====================================================================
# 4. MAIN DASHBOARD HEADER & DATA PIPELINE
# =====================================================================
st.title("S&P Dow Jones Indices | Bespoke Index Engineering Sandbox")
st.markdown(
    """
    **Interactive Quantitative Workbench:** Validate structured constituent price feeds, 
    model rebalancing turnover, and compare **Equal-Weight (1/N)** vs. **Low-Volatility Smart Beta** weighting schemes.
    """
)

# Date input validation
if len(date_range) != 2:
    st.info("Please select both a start and end date in the sidebar.")
    st.stop()

start_str, end_str = date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d")

# 1. Fetch & Clean Market Data
raw_prices = load_market_data(selected_tickers, start_str, end_str)
cleaned_prices, audit_log = run_data_qa_pipeline(raw_prices)

# 2. Run Index Math & Extract Weight Histories
engine = BespokeIndexEngine(cleaned_prices, initial_index_level=initial_level)

ew_index, ew_weights_df = engine.run_rebalanced_index(schedule=rebal_freq, scheme="equal")
iv_index, iv_weights_df = engine.run_rebalanced_index(schedule=rebal_freq, scheme="inv_vol")

# Combine headline index levels for simple comparison
comparison_df = pd.DataFrame({
    "Equal-Weight Index (1/N)": ew_index["Index_Level"],
    "Smart Beta Index (Inv-Vol)": iv_index["Index_Level"]
})

# =====================================================================
# 5. TABBED WORKSPACE
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Index Performance & Alpha", 
    "⚙️ Divisor & Rebalancing Audit", 
    "🥧 Rebalance Weight Explorer", 
    "🛡️ Structured Data QA Log"
])

# ---------------------------------------------------------------------
# TAB 1: INDEX PERFORMANCE & ALPHA
# ---------------------------------------------------------------------
with tab1:
    ew_total_ret = ((ew_index["Index_Level"].iloc[-1] / initial_level) - 1.0) * 100
    iv_total_ret = ((iv_index["Index_Level"].iloc[-1] / initial_level) - 1.0) * 100
    alpha_spread = iv_total_ret - ew_total_ret

    col1, col2, col3 = st.columns(3)
    col1.metric("Equal-Weight Return", f"{ew_total_ret:.2f}%", f"{ew_index['Index_Level'].iloc[-1]:.2f} pts")
    col2.metric("Smart Beta (Inv-Vol) Return", f"{iv_total_ret:.2f}%", f"{iv_index['Index_Level'].iloc[-1]:.2f} pts")
    col3.metric("Smart Beta vs. EW Spread", f"{alpha_spread:.2f}%", delta_color="normal")

    st.markdown("---")
    
    st.subheader("Normalized Index Performance Comparison")
    st.line_chart(comparison_df, use_container_width=True)
    
    # Calculate & Display Tracking Error
    daily_returns = comparison_df.pct_change().dropna()
    tracking_diff = daily_returns["Smart Beta Index (Inv-Vol)"] - daily_returns["Equal-Weight Index (1/N)"]
    annualized_te = np.sqrt(252) * tracking_diff.std() * 100
    st.caption(f"**Annualized Tracking Error (Smart Beta vs. Equal-Weight):** {annualized_te:.2f}%")

# ---------------------------------------------------------------------
# TAB 2: DIVISOR & REBALANCING AUDIT
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Index Divisor Continuity Log")
    st.markdown(
        """
        When constituent weights are rebalanced, the **Index Divisor** must adjust dynamically so the 
        headline index level does not experience an artificial jump due to portfolio turnover.
        """
    )
    
    # Extract rebalance dates where divisor changed
    ew_divisors = ew_index["Divisor"]
    rebalance_events = ew_divisors[ew_divisors.diff().abs() > 1e-6].to_frame(name="New Equal-Weight Divisor")
    rebalance_events["New Smart-Beta Divisor"] = iv_index.loc[rebalance_events.index, "Divisor"]
    rebalance_events.index = rebalance_events.index.strftime("%Y-%m-%d")
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.dataframe(rebalance_events, use_container_width=True)
    with col_right:
        st.info(
            """
            **Index Engineering Note:**  
            Notice how the divisor changes on each scheduled rebalance date while the closing 
            index level remains continuous across trading sessions.
            """
        )
    
    st.subheader("Divisor Drift Over Time")
    divisor_df = pd.DataFrame({
        "Equal-Weight Divisor": ew_index["Divisor"],
        "Smart-Beta Divisor": iv_index["Divisor"]
    })
    st.line_chart(divisor_df, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: REBALANCE WEIGHT EXPLORER (PLOTLY)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Constituent Allocation on Rebalance Dates")
    st.markdown(
        """
        Compare how **Equal-Weight (1/N)** resets all constituents identically, while 
        **Smart Beta (Inverse-Volatility)** tilts heavier toward defensive, low-volatility names.
        """
    )
    
    # Date dropdown for rebalance events
    available_dates = iv_weights_df.index.strftime("%Y-%m-%d").tolist()
    selected_date_str = st.selectbox(
        "Select Rebalance Event Date:", 
        options=available_dates, 
        index=len(available_dates)-1 # Default to most recent
    )
    
    selected_dt = pd.to_datetime(selected_date_str)
    ew_slice = ew_weights_df.loc[selected_dt]
    iv_slice = iv_weights_df.loc[selected_dt]
    
    # Side-by-Side Donut Charts
    fig_donut = make_subplots(
        rows=1, cols=2, 
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=[
            f"Equal-Weight (1/N) — {selected_date_str}", 
            f"Smart Beta (Inv-Vol) — {selected_date_str}"
        ]
    )
    
    fig_donut.add_trace(
        go.Pie(
            labels=ew_slice.index, 
            values=ew_slice.values, 
            name="Equal-Weight",
            hole=0.4,
            textinfo="label+percent"
        ), 
        row=1, col=1
    )
    
    fig_donut.add_trace(
        go.Pie(
            labels=iv_slice.index, 
            values=iv_slice.values, 
            name="Smart Beta",
            hole=0.4,
            textinfo="label+percent"
        ), 
        row=1, col=2
    )
    
    fig_donut.update_layout(
        height=400,
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("---")
    
    # Stacked Historical Drift Chart
    st.subheader("Smart Beta Constituent Drift Across All Rebalance Periods")
    st.caption("Shows how Inverse-Volatility allocations dynamically adapt to changing market conditions.")
    
    iv_weights_reset = iv_weights_df.copy()
    iv_weights_reset.index = iv_weights_reset.index.strftime("%Y-%m-%d")
    iv_weights_reset.index.name = "Rebalance Date"
    
    fig_stacked = px.bar(
        iv_weights_reset, 
        x=iv_weights_reset.index, 
        y=iv_weights_reset.columns,
        title="Historical Smart Beta Constituent Weight Allocations",
        labels={"value": "Weight Allocation", "variable": "Ticker"},
        barmode="stack"
    )
    
    fig_stacked.update_layout(
        yaxis=dict(tickformat=".0%"),
        height=450,
        hovermode="x unified",
        legend_title_text="Constituents"
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: STRUCTURED DATA QA LOG
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Automated Data Quality Assurance (QA) Audit")
    st.markdown(
        """
        Directly targets the JD requirement to **'review and validate structured data outputs'**. 
        This automated pipeline checks raw constituent data for stale prices, missing bars, and unadjusted corporate actions before calculation.
        """
    )
    
    flagged_count = (audit_log["qa_status"] == "FLAGGED").sum()
    if flagged_count > 0:
        st.warning(f"⚠️ Anomaly Warning: Found **{flagged_count}** constituent(s) with data quality flags.")
    else:
        st.success("✅ Clean Feed: All constituents passed automated data quality validation.")
    
    def highlight_qa_status(val):
        color = "#ff4b4b" if val == "FLAGGED" else "#09ab3b"
        return f"color: {color}; font-weight: bold;"
    
    st.dataframe(
        audit_log.style.map(highlight_qa_status, subset=["qa_status"]),
        use_container_width=True
    )
    
    st.markdown("### Raw vs. Cleaned Data Inspection")
    selected_inspect_ticker = st.selectbox("Select constituent to inspect:", selected_tickers)
    
    inspect_df = pd.DataFrame({
        "Raw Price (Uncleaned)": raw_prices[selected_inspect_ticker],
        "Cleaned & Imputed Price": cleaned_prices[selected_inspect_ticker]
    })
    st.line_chart(inspect_df, use_container_width=True)
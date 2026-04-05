import os
import sys
import time
import json
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta as ta_lib
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-card h4 {
        color: #a5b4fc;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-card .change-up { color: #34d399; }
    .metric-card .change-down { color: #f87171; }

    /* Analysis Box */
    .analysis-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        line-height: 1.7;
        color: #cbd5e1;
    }
    .analysis-box h1, .analysis-box h2, .analysis-box h3 {
        color: #a5b4fc;
    }
    .analysis-box strong { color: #e2e8f0; }

    /* Signal Badge */
    .signal-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .signal-buy {
        background: linear-gradient(135deg, #059669, #34d399);
        color: #022c22;
    }
    .signal-sell {
        background: linear-gradient(135deg, #dc2626, #f87171);
        color: #450a0a;
    }
    .signal-hold {
        background: linear-gradient(135deg, #d97706, #fbbf24);
        color: #451a03;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stDateInput label {
        color: #a5b4fc !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #312e81 0%, #1e1b4b 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a78bfa, #818cf8);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def load_api_key():
    """Load Gemini key from env or .env file."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("GOOGLE_API_KEY"):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y"):
    """Fetch stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    info = stock.info
    return hist, info


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe."""
    df = df.copy()
    # Moving Averages
    df['SMA_20'] = ta_lib.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta_lib.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta_lib.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta_lib.trend.ema_indicator(df['Close'], window=26)
    # RSI
    df['RSI'] = ta_lib.momentum.rsi(df['Close'], window=14)
    # MACD
    macd = ta_lib.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    # Bollinger Bands
    bb = ta_lib.volatility.BollingerBands(df['Close'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    # Volume SMA
    df['Vol_SMA'] = ta_lib.trend.sma_indicator(df['Volume'], window=20)
    return df


def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build an interactive candlestick chart with indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f'{ticker} Price & Bollinger Bands',
            'RSI (14)',
            'MACD',
            'Volume'
        ]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price',
        increasing_line_color='#34d399', decreasing_line_color='#f87171'
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
        line=dict(color='rgba(99,102,241,0.3)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
        line=dict(color='rgba(99,102,241,0.3)', width=1),
        fill='tonexty', fillcolor='rgba(99,102,241,0.05)'), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
        line=dict(color='#fbbf24', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
        line=dict(color='#fb923c', width=1.5)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f87171", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#34d399", row=2, col=1)

    # MACD
    colors = ['#34d399' if v >= 0 else '#f87171' for v in df['MACD_Hist'].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
        marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='#818cf8', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
        line=dict(color='#fbbf24', width=1.5)), row=3, col=1)

    # Volume
    vol_colors = ['#34d399' if c >= o else '#f87171'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
        marker_color=vol_colors, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_SMA'], name='Vol SMA',
        line=dict(color='#fbbf24', width=1)), row=4, col=1)

    fig.update_layout(
        height=900,
        template='plotly_dark',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', color='#94a3b8'),
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=20, l=50, r=20)
    )
    fig.update_xaxes(gridcolor='rgba(148,163,184,0.08)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.08)')

    return fig


def build_analysis_prompt(ticker, info, df, period_label):
    """Build a single comprehensive prompt for Gemini."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price_change = latest['Close'] - prev['Close']
    pct_change = (price_change / prev['Close']) * 100 if prev['Close'] != 0 else 0

    # Summarize fundamentals
    fundamentals = []
    for key in ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield',
                'profitMargins', 'revenueGrowth', 'debtToEquity',
                'returnOnEquity', 'totalRevenue', 'totalDebt',
                'totalCash', 'freeCashflow', 'operatingMargins',
                'earningsGrowth', 'currentRatio', 'quickRatio',
                'bookValue', 'priceToBook', 'enterpriseToRevenue',
                'enterpriseToEbitda']:
        val = info.get(key)
        if val is not None:
            fundamentals.append(f"  - {key}: {val}")

    fund_text = "\n".join(fundamentals) if fundamentals else "  Not available"

    # Detect currency
    cur = "₹" if (".NS" in ticker or ".BO" in ticker) else "$"

    # Technical summary
    tech_text = f"""
  Latest Close: {cur}{latest['Close']:.2f}
  Change: {cur}{price_change:+.2f} ({pct_change:+.2f}%)
  RSI(14): {latest['RSI']:.1f}
  MACD: {latest['MACD']:.4f}
  MACD Signal: {latest['MACD_Signal']:.4f}
  MACD Histogram: {latest['MACD_Hist']:.4f}
  SMA 20: {cur}{latest['SMA_20']:.2f}
  SMA 50: {cur}{latest['SMA_50']:.2f}
  Bollinger Upper: {cur}{latest['BB_Upper']:.2f}
  Bollinger Lower: {cur}{latest['BB_Lower']:.2f}
  Volume: {latest['Volume']:,.0f}
  Avg Volume (20d): {latest['Vol_SMA']:,.0f}
"""

    # Recent price action (last 10 days)
    recent = df.tail(10)[['Close', 'Volume', 'RSI']].to_string()

    market_context = "Indian stock market (NSE/BSE). Use ₹ (Indian Rupees) for all prices." if (".NS" in ticker or ".BO" in ticker) else "US stock market. Use $ (USD) for all prices."

    prompt = f"""You are an expert financial analyst. Analyze {ticker} ({info.get('longName', ticker)}) and provide a comprehensive trading analysis.

## Market
{market_context}

## Company Info
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Country: {info.get('country', 'N/A')}
- Employees: {info.get('fullTimeEmployees', 'N/A')}
- Summary: {info.get('longBusinessSummary', 'N/A')[:500]}

## Fundamental Data (Balance Sheet & Financials)
{fund_text}

## Technical Indicators (as of {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else 'latest'})
{tech_text}

## Recent Price Action (Last 10 Trading Days)
{recent}

## Analysis Period: {period_label}

---

Please provide your analysis in the following structure:

### 📊 EXECUTIVE SUMMARY
A 2-3 sentence overview of the stock's current position.

### 📈 TECHNICAL ANALYSIS
- Analyze RSI, MACD, Moving Averages, Bollinger Bands, and Volume trends.
- Identify key support/resistance levels.
- Note any chart patterns or divergences.

### 💰 FUNDAMENTAL ANALYSIS
- Evaluate the company's financial health using the data above.
- Comment on valuation (P/E, P/B, EV/EBITDA).
- Assess debt levels, cash flow, and profitability margins.
- Is the company fundamentally strong or weak?

### 📰 MARKET SENTIMENT & RISK FACTORS
- What market-wide or sector-specific risks exist?
- Any upcoming catalysts (earnings, product launches, etc.)?

### 🎯 TRADING RECOMMENDATION
- **Signal:** BUY / SELL / HOLD
- **Confidence:** High / Medium / Low
- **Time Horizon:** Short-term (1-2 weeks) / Medium-term (1-3 months) / Long-term (3-12 months)
- **Entry Price:** (if applicable)
- **Target Price:** (if applicable)
- **Stop Loss:** (if applicable)

### ⚠️ DISCLAIMER
Always include a brief disclaimer that this is AI-generated analysis and not financial advice.

Be specific with numbers. Use the actual data provided above. Do not make up data.
"""
    return prompt


def run_gemini_analysis(prompt: str, api_key: str):
    """Call Gemini API with a single comprehensive prompt."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text


def format_large_number(num, currency="$"):
    """Format large numbers (market cap, revenue, etc.)"""
    if num is None:
        return "N/A"
    if abs(num) >= 1e12:
        return f"{currency}{num/1e12:.2f}T"
    if abs(num) >= 1e9:
        return f"{currency}{num/1e9:.2f}B"
    if abs(num) >= 1e6:
        return f"{currency}{num/1e6:.2f}M"
    if abs(num) >= 1e5:
        return f"{currency}{num/1e5:.2f}L"
    return f"{currency}{num:,.0f}"


# ─────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <h1>📈 AI Trading Assistant</h1>
    <p>Real-time stock analysis powered by Yahoo Finance data & Google Gemini AI — built to work reliably on the free tier.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    market = st.selectbox("Market", ["🇺🇸 US (NYSE/NASDAQ)", "🇮🇳 India NSE", "🇮🇳 India BSE"], index=0)

    if "US" in market:
        default_ticker = "AAPL"
        ticker_help = "e.g., AAPL, GOOGL, TSLA, MSFT, NVDA"
        currency = "$"
        suffix = ""
    elif "NSE" in market:
        default_ticker = "RELIANCE"
        ticker_help = "e.g., RELIANCE, TCS, INFY, HDFCBANK, TATAMOTORS"
        currency = "₹"
        suffix = ".NS"
    else:
        default_ticker = "RELIANCE"
        ticker_help = "e.g., RELIANCE, TCS, INFY, HDFCBANK, TATAMOTORS"
        currency = "₹"
        suffix = ".BO"

    raw_ticker = st.text_input("Stock Ticker", value=default_ticker, help=ticker_help).upper().strip()
    # Build the Yahoo Finance ticker (auto-append suffix)
    ticker = raw_ticker + suffix if suffix and not raw_ticker.endswith(suffix) else raw_ticker
    display_ticker = raw_ticker  # Clean name for display

    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    period_label = st.selectbox("Analysis Period", list(period_options.keys()), index=3)
    period = period_options[period_label]

    st.markdown("---")
    run_ai = st.button("🚀 Run Full AI Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0.8rem; border-radius:12px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2);">
        <p style="color:#a5b4fc; font-size:0.75rem; margin:0;"><strong>HOW IT WORKS</strong></p>
        <p style="color:#94a3b8; font-size:0.72rem; margin:0.3rem 0 0 0;">
        1️⃣ Fetches live data from Yahoo Finance<br>
        2️⃣ Computes RSI, MACD, Bollinger Bands<br>
        3️⃣ Reads balance sheets & fundamentals<br>
        4️⃣ Sends everything to Gemini AI (1 call)<br>
        5️⃣ Gets a complete analysis in ~15 seconds
        </p>
    </div>
    """, unsafe_allow_html=True)
    if suffix:
        st.caption(f"Yahoo Finance ticker: `{ticker}`")


# ─────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────

if ticker:
    try:
        with st.spinner(f"Fetching {ticker} data from Yahoo Finance..."):
            hist, info = fetch_stock_data(ticker, period)

        if hist.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
            st.stop()

        # Compute technical indicators
        df = compute_technicals(hist)

        # ── Top Metrics Row ──
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        price_change = latest['Close'] - prev['Close']
        pct_change = (price_change / prev['Close']) * 100 if prev['Close'] != 0 else 0
        change_class = "change-up" if price_change >= 0 else "change-down"
        arrow = "▲" if price_change >= 0 else "▼"

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current Price</h4>
                <div class="value">{currency}{latest['Close']:.2f}</div>
                <div class="{change_class}" style="font-size:0.85rem;">{arrow} {currency}{abs(price_change):.2f} ({pct_change:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            rsi_val = latest['RSI']
            rsi_color = "change-up" if 30 <= rsi_val <= 70 else "change-down"
            rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            st.markdown(f"""
            <div class="metric-card">
                <h4>RSI (14)</h4>
                <div class="value">{rsi_val:.1f}</div>
                <div class="{rsi_color}" style="font-size:0.85rem;">{rsi_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Market Cap</h4>
                <div class="value">{format_large_number(info.get('marketCap'), currency)}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">{info.get('sector', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            pe = info.get('trailingPE')
            st.markdown(f"""
            <div class="metric-card">
                <h4>P/E Ratio</h4>
                <div class="value">{f'{pe:.1f}' if pe else 'N/A'}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">Trailing 12M</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            vol = latest['Volume']
            avg_vol = latest['Vol_SMA']
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1
            vol_class = "change-up" if vol_ratio > 1.2 else ("change-down" if vol_ratio < 0.8 else "")
            st.markdown(f"""
            <div class="metric-card">
                <h4>Volume</h4>
                <div class="value">{vol/1e6:.1f}M</div>
                <div class="{vol_class}" style="font-size:0.85rem;">{vol_ratio:.1f}x avg</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Interactive Chart ──
        st.markdown("### 📊 Interactive Technical Chart")
        chart = build_price_chart(df, ticker)
        st.plotly_chart(chart, use_container_width=True)

        # ── AI Analysis ──
        if run_ai:
            api_key = load_api_key()
            if not api_key:
                st.error("❌ Google API key not found. Add `GOOGLE_API_KEY=your_key` to the `.env` file in this directory.")
                st.stop()

            st.markdown("### 🤖 AI-Powered Analysis")

            progress = st.progress(0, text="Preparing data...")
            time.sleep(0.5)
            progress.progress(20, text="Building comprehensive analysis prompt...")
            
            prompt = build_analysis_prompt(ticker, info, df, period_label)
            
            progress.progress(40, text="Sending to Google Gemini AI (single API call)...")

            try:
                analysis = run_gemini_analysis(prompt, api_key)
                progress.progress(90, text="Formatting results...")
                time.sleep(0.3)
                progress.progress(100, text="✅ Analysis complete!")
                time.sleep(0.5)
                progress.empty()

                # Determine signal for badge
                analysis_lower = analysis.lower()
                if "signal:** buy" in analysis_lower or "signal: buy" in analysis_lower:
                    badge_class = "signal-buy"
                    badge_text = "📈 BUY SIGNAL"
                elif "signal:** sell" in analysis_lower or "signal: sell" in analysis_lower:
                    badge_class = "signal-sell"
                    badge_text = "📉 SELL SIGNAL"
                else:
                    badge_class = "signal-hold"
                    badge_text = "⏸️ HOLD SIGNAL"

                st.markdown(f'<div style="text-align:center; margin:1rem 0;"><span class="signal-badge {badge_class}">{badge_text}</span></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)

                st.success("✅ Analysis complete! This used only **1 API call** — your free tier is safe.")

            except Exception as e:
                progress.empty()
                st.error(f"❌ Gemini API Error: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("You've hit the rate limit. Wait 60 seconds and try again. This app only makes 1 API call, so this is very rare.")
                st.info("Make sure your API key is valid at https://aistudio.google.com/apikey")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.info("Please check that the ticker symbol is correct (e.g., AAPL, GOOGL, MSFT, TSLA)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem; color:#475569; font-size:0.75rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. AI-generated analysis is not financial advice. Always do your own research before making investment decisions.</p>
    <p>Data from Yahoo Finance • AI by Google Gemini • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

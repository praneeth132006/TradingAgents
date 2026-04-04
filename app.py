import os
import sys
import streamlit as st

# Setup path to import TradingAgents correctly
sys.path.append(os.path.dirname(__file__))

# Set environment variables so the framework picks up the API key if it's set in process
# Note: It usually loads from .env, but Streamlit can sometimes be finicky 
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

st.set_page_config(page_title="AI Trading Assistant", page_icon="📈", layout="wide")

st.title("📈 AI Trading Assistant")
st.markdown("Powered by `TauricResearch/TradingAgents` and Google Gemini AI")

st.info("This is a multi-agent framework. When you run the analysis, a team of virtual AI analysts (Fundamental, Technical, Sentiment, and News) will automatically fetch data, read the news, calculate indicators, and debate to reach a trading decision.")

# Sidebar for inputs
with st.sidebar:
    st.header("Trading Configuration")
    ticker = st.text_input("Stock Ticker (e.g., AAPL, NVDA, TSLA)", value="AAPL").upper()
    date = st.date_input("Analysis Date")
    
    if st.button("Run AI Analysis", type="primary"):
        st.session_state['run_analysis'] = True

if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
    st.write(f"### 🔍 Researching {ticker} on {date}...")
    
    st.info("⚠️ **Note:** Real-time AI analysis takes **2 to 4 minutes**. The system is downloading 5 years of stock prices, checking balance sheets, and making multiple AIs debate the best action. Please do not refresh the page.")
    
    progress_text = "The AI Team is analyzing financials, reading news, and debating the best move..."
    
    # Run the TradingAgents logic
    with st.spinner(progress_text):
        try:
            config = DEFAULT_CONFIG.copy()
            config["llm_provider"] = "google"
            config["backend_url"] = None  # Prevent it from routing Google to OpenAI!
            
            # Using stable models provided by Gemini
            config["deep_think_llm"] = "gemini-3.1-pro-preview"
            config["quick_think_llm"] = "gemini-2.5-flash"
            
            # Initialize the agent graph
            ta = TradingAgentsGraph(debug=False, config=config)
            
            import time
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Execute the agent simulation
                    result, decision = ta.propagate(ticker, str(date))
                    
                    st.success("✅ Analysis Complete!")
                    st.subheader("Final Trading Decision")
                    
                    if hasattr(decision, 'dict'):
                        st.json(decision.dict())
                    else:
                        st.write(decision)
                        
                    st.balloons()
                    break # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        if attempt < max_retries - 1:
                            st.warning(f"⚠️ Google's free-tier speed limit hit! The AIs are working too fast. Automatically pausing for 60 seconds and then retrying... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(60)
                            st.info("🔄 Retrying analysis now...")
                        else:
                            st.error("❌ Rate limit still exceeded after auto-retry. Please wait a few minutes and click Run again.")
                    else:
                        st.error(f"❌ An error occurred during analysis: {e}")
                        st.write("Ensure your API key is valid and you have internet access.")
                        break # Not a rate limit error, break out of loop
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
    
    # Prevent re-running unless button is clicked again
    st.session_state['run_analysis'] = False
